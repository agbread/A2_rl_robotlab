"""Opt-in straight-stair task components. Original Go2/A2 tasks are unchanged."""

from __future__ import annotations

import math
import numpy as np
import torch
import trimesh
from isaaclab.managers import CommandTerm
from isaaclab.terrains import SubTerrainBaseCfg, TerrainGeneratorCfg, MeshPlaneTerrainCfg
from isaaclab.utils import configclass
import isaaclab.utils.math as math_utils
from ..stair_geometry import get_stair_spec, build_stair_boxes, curriculum_spec, validate_stair_boxes
from ..stair_logic import StairCourseTracker, StairCohortCurriculum, resolve_stair_completion
from .commands import A2RLGymCommand, A2RLGymCommandCfg
from .events import reset_root_state_uniform


def spec_for_terrain(cfg, level=0):
    common = dict(num_steps=cfg.num_steps, rise=cfg.rise, tread=cfg.tread,
                  approach=cfg.approach, landing=cfg.landing, width=cfg.width)
    if cfg.curriculum_mode == "fixed":
        return get_stair_spec(**common)
    return curriculum_spec(level, mode=cfg.curriculum_mode, step_counts=cfg.step_counts,
                           heights=cfg.heights, **common)


def straight_stairs_terrain(difficulty, cfg):
    levels = len(cfg.step_counts) if cfg.curriculum_mode == "steps" else len(cfg.heights)
    spec = spec_for_terrain(cfg, min(int(float(difficulty) * levels), levels - 1))
    boxes = build_stair_boxes(spec)
    validate_stair_boxes(spec, boxes)
    origin = np.array([cfg.start_x, cfg.size[1] / 2, 0.0])
    if spec.top_end + origin[0] >= cfg.size[0] - cfg.border_width:
        raise ValueError("Stair course does not fit inside its tile")
    if spec.width > cfg.size[1] - 2 * cfg.border_width:
        raise ValueError("Stair course is wider than its tile")
    floor = trimesh.creation.box((cfg.size[0], cfg.size[1], 0.1),
            trimesh.transformations.translation_matrix((cfg.size[0]/2, cfg.size[1]/2, -0.05)))
    meshes = [floor]
    for box in boxes:
        meshes.append(trimesh.creation.box(box.size,
                      trimesh.transformations.translation_matrix(np.array(box.center) + origin)))
    return meshes, origin


@configclass
class StraightStairsTerrainCfg(SubTerrainBaseCfg):
    function = straight_stairs_terrain
    num_steps: int = 12
    rise: float = 0.15
    tread: float = 0.20
    approach: float = 1.5
    landing: float = 1.2
    width: float = 3.0
    start_x: float = 1.0
    border_width: float = 0.5
    curriculum_mode: str = "fixed"
    step_counts: tuple[int, ...] = (6, 8, 10, 12)
    heights: tuple[float, ...] = (0.12, 0.135, 0.15)


def make_stair_terrain_cfg(curriculum_mode="steps"):
    if curriculum_mode not in ("steps", "height"):
        raise ValueError("curriculum_mode must be steps or height")
    return TerrainGeneratorCfg(
        seed=42, size=(12.0, 9.0), border_width=20.0,
        num_rows=4 if curriculum_mode == "steps" else 3, num_cols=20,
        curriculum=True, difficulty_range=(0.0, 1.0), use_cache=False,
        horizontal_scale=0.1, vertical_scale=0.005,
        sub_terrains={
            "target_12": StraightStairsTerrainCfg(proportion=0.40),
            "stair_curriculum": StraightStairsTerrainCfg(proportion=0.20, curriculum_mode=curriculum_mode),
            "near_low": StraightStairsTerrainCfg(proportion=0.05, rise=0.14),
            "near_high": StraightStairsTerrainCfg(proportion=0.05, rise=0.16),
            "near_narrow": StraightStairsTerrainCfg(proportion=0.05, tread=0.18),
            "near_wide": StraightStairsTerrainCfg(proportion=0.05, tread=0.22),
            "flat": MeshPlaneTerrainCfg(proportion=0.20),
        })


class StairVelocityCommand(A2RLGymCommand):
    """Existing flat commands plus constant-episode forward stair commands."""

    def _init_terrain_infos(self):
        terrain = self._env.scene.terrain
        cfg = terrain.cfg.terrain_generator
        self.terrain_types = list(cfg.sub_terrains)
        self.terrain_type2idx = {name: i for i, name in enumerate(self.terrain_types)}
        proportions = np.array([v.proportion for v in cfg.sub_terrains.values()], dtype=float)
        cumulative = np.cumsum(proportions / proportions.sum())
        # Match the actual generator; the legacy helper uses a different rounding rule.
        columns = np.array([np.flatnonzero(i / cfg.num_cols + 0.001 < cumulative)[0]
                            for i in range(cfg.num_cols)])
        self.terrain_idxs = torch.as_tensor(columns, device=self.device)[terrain.terrain_types]
        self.terrain_length = cfg.size[0]
        self.stair_mask = self.terrain_idxs != self.terrain_type2idx["flat"]
        self.target_mask = self.terrain_idxs == self.terrain_type2idx["target_12"]
        self.curriculum_mask = self.terrain_idxs == self.terrain_type2idx["stair_curriculum"]
        self.near_mask = self.stair_mask & ~self.target_mask & ~self.curriculum_mask
        table = np.zeros((cfg.num_rows, cfg.num_cols, 6), dtype=np.float32)
        table[:] = [1, 0.15, 0.2, 1.5, 1.2, 3.0]
        for col, type_index in enumerate(columns):
            sub_cfg = cfg.sub_terrains[self.terrain_types[type_index]]
            if not isinstance(sub_cfg, StraightStairsTerrainCfg):
                continue
            for row in range(cfg.num_rows):
                s = spec_for_terrain(sub_cfg, row)
                table[row, col] = [s.num_steps, s.rise, s.tread, s.approach, s.landing, s.width]
        self.spec_table = torch.as_tensor(table, device=self.device)
        self.specs = self.spec_table[terrain.terrain_levels, terrain.terrain_types].clone()
        expected = dict(target_12=8, stair_curriculum=4, near_low=1, near_high=1, near_narrow=1, near_wide=1, flat=4)
        for name, count in expected.items():
            if np.count_nonzero(columns == self.terrain_type2idx[name]) != count:
                raise ValueError(f"Incorrect terrain column allocation for {name}")
        print("[A2Stairs] terrain columns:", {name: np.flatnonzero(columns == i).tolist()
                                               for name, i in self.terrain_type2idx.items()})

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        self.foot_ids, _ = self.robot.find_bodies(foot_names, preserve_order=True)
        self.contact_sensor = env.scene["contact_forces"]
        self.contact_ids, _ = self.contact_sensor.find_bodies(foot_names, preserve_order=True)
        if len(self.foot_ids) != 4 or len(self.contact_ids) != 4:
            raise ValueError("Stair completion requires all four A2 feet")
        self.course = StairCourseTracker(self.num_envs, self.device)
        self.cohort = StairCohortCurriculum(env.scene.terrain.max_terrain_level, self.device,
                                           cfg.curriculum_window, cfg.curriculum_success_rate)
        self.completed_episodes = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.successful_episodes = torch.zeros_like(self.completed_episodes)
        self.settled_episodes = torch.zeros_like(self.completed_episodes)
        self.ascent_episodes = torch.zeros_like(self.completed_episodes)
        self.failed_episodes = torch.zeros_like(self.completed_episodes)
        self.assigned_seconds_total = torch.zeros(self.num_envs, device=self.device)
        self.contact_seconds_total = torch.zeros_like(self.assigned_seconds_total)
        self.foot_contact_seconds_total = torch.zeros_like(self.assigned_seconds_total)
        self._last_course_step = -1

    def _resample(self, env_ids):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        if len(env_ids) == 0:
            return
        flat_ids = env_ids[~self.stair_mask[env_ids]]
        if len(flat_ids):
            super()._resample(flat_ids)
        stair_ids = env_ids[self.stair_mask[env_ids]]
        if len(stair_ids):
            self.commands[stair_ids] = 0.0
            self.commands[stair_ids, 0] = torch.empty(len(stair_ids), device=self.device).uniform_(*self.cfg.forward_speed)
            self.time_left[stair_ids] = self._env.max_episode_length_s + 1.0

    def update_course(self):
        # Called by termination terms before resets, once per control step.
        step = self._env.common_step_counter
        if self._last_course_step == step:
            return
        self._last_course_step = step
        origins = self._env.scene.env_origins
        feet = self.robot.data.body_pos_w[:, self.foot_ids] - origins[:, None, :]
        root = self.robot.data.root_pos_w - origins
        forces = self.contact_sensor.data.net_forces_w_history[:, :self._env.cfg.decimation, self.contact_ids, 2]
        self.course.update(feet, forces, root, self.robot.data.projected_gravity_b,
                           self.specs, self.stair_mask, self._env.step_dt)
        self.commands[self.course.stop & self.stair_mask] = 0.0

    def _update_command(self):
        super()._update_command()
        if hasattr(self, "course"):
            self.commands[self.course.stop & self.stair_mask] = 0.0

    def completed(self):
        return resolve_stair_completion(self.course.ascent_completed, self.course.success,
                                        self.course.failed, self.cfg.require_settled_completion)

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        if not isinstance(ids, torch.Tensor):
            ids = torch.as_tensor(ids, device=self.device, dtype=torch.long)
        # Curriculum and root reset precede this call; use cached episode measurements.
        measured_ids = ids[self._env.episode_length_buf[ids] > 0]
        self.completed_episodes[measured_ids] += 1
        completed = self.completed()
        self.successful_episodes[measured_ids] += completed[measured_ids].long()
        self.settled_episodes[measured_ids] += self.course.success[measured_ids].long()
        self.ascent_episodes[measured_ids] += self.course.ascent_completed[measured_ids].long()
        failure = self.course.failed | (getattr(self._env, "reset_terminated", torch.zeros_like(self.stair_mask)) & self.stair_mask)
        self.failed_episodes[measured_ids] += failure[measured_ids].long()
        self.assigned_seconds_total[measured_ids] += self.course.elapsed_s[measured_ids]
        self.contact_seconds_total[measured_ids] += self.course.contact_time_s[measured_ids]
        self.foot_contact_seconds_total[measured_ids] += self.course.foot_contact_time_s[measured_ids]
        extras = {}
        groups = dict(stairs=self.stair_mask, target=self.target_mask,
                      curriculum=self.curriculum_mask, near=self.near_mask)
        policies = {"all": torch.ones_like(self.stair_mask)}
        for name in ("teacher", "student"):
            mask = getattr(self._env, f"{name}_env_mask", None)
            if mask is not None:
                policies[name] = mask.to(device=self.device)
        for group_name, mask in groups.items():
            for policy_name, policy_mask in policies.items():
                selected = measured_ids[(mask & policy_mask)[measured_ids]]
                if not len(selected):
                    continue
                prefix = f"stairs/{group_name}/{policy_name}"
                extras[f"{prefix}/episodes"] = len(selected)
                extras[f"{prefix}/success_count"] = completed[selected].sum().item()
                extras[f"{prefix}/settled_success_count"] = self.course.success[selected].sum().item()
                extras[f"{prefix}/ascent_count"] = self.course.ascent_completed[selected].sum().item()
                extras[f"{prefix}/failure_count"] = failure[selected].sum().item()
                all_ids = (mask & policy_mask).nonzero().flatten()
                for name, values in (("episodes_total", self.completed_episodes), ("success_total", self.successful_episodes),
                                     ("settled_success_total", self.settled_episodes),
                                     ("ascent_total", self.ascent_episodes), ("failure_total", self.failed_episodes),
                                     ("assigned_seconds_total", self.assigned_seconds_total),
                                     ("contact_seconds_total", self.contact_seconds_total),
                                     ("foot_contact_seconds_total", self.foot_contact_seconds_total)):
                    extras[f"{prefix}/{name}"] = values[all_ids].sum().item()
                for key, values in (("success", completed.float()), ("settled_success", self.course.success.float()),
                                    ("ascent_completed", self.course.ascent_completed.float()),
                                    ("failure", failure.float()),
                                    ("progress_risers", self.course.progress_steps),
                                    ("contact_seconds", self.course.contact_time_s),
                                    ("foot_contact_seconds", self.course.foot_contact_time_s),
                                    ("assigned_seconds", self.course.elapsed_s)):
                    extras[f"{prefix}/{key}"] = values[selected].mean().item()
                for i, stage in enumerate(("early", "middle", "late")):
                    seen = selected[self.course.stage_samples[selected, i] > 0]
                    if len(seen):
                        denom = self.course.stage_samples[seen, i]
                        extras[f"{prefix}/{stage}_roll_deg"] = (self.course.stage_roll_sum[seen, i] / denom).mean().item() * 180 / math.pi
                        extras[f"{prefix}/{stage}_abs_roll_deg"] = (self.course.stage_roll_abs_sum[seen, i] / denom).mean().item() * 180 / math.pi
        for level in range(self.cohort.history.shape[0]):
            count = min(int(self.cohort.counts[level].item()), self.cohort.window_size)
            extras[f"stairs/cohort/level_{level}_episodes"] = int(self.cohort.counts[level].item())
            if count:
                extras[f"stairs/cohort/level_{level}_success"] = self.cohort.history[level, :count].float().mean().item()
        extras["stairs/cohort/unlocked_level"] = self.cohort.unlocked_level
        extras["stairs/require_settled_completion"] = int(self.cfg.require_settled_completion)
        self.course.reset(ids)
        terrain = self._env.scene.terrain
        self.specs[ids] = self.spec_table[terrain.terrain_levels[ids], terrain.terrain_types[ids]]
        self.commands_xy_accumulation[ids] = 0
        self.max_move_distance[ids] = 0
        self.last_is_limit_vel[ids] = False
        extras.update(CommandTerm.reset(self, ids))
        return extras


@configclass
class StairVelocityCommandCfg(A2RLGymCommandCfg):
    class_type: type = StairVelocityCommand
    forward_speed: tuple[float, float] = (0.3, 0.5)
    require_settled_completion: bool = False
    curriculum_window: int = 100
    curriculum_success_rate: float = 0.90
    command_range_curriculum: list = []
    zero_command_curriculum: dict = {"start_iter": 0, "end_iter": 1, "start_value": 0.1, "end_value": 0.1}
    ranges = A2RLGymCommandCfg.Ranges(lin_vel_x=(-2.0, 2.0), lin_vel_y=(-1.0, 1.0), ang_vel_yaw=(-2.0, 2.0))
    terrain_max_command_ranges: dict = {
        name: {"lin_vel_x": (-2.0, 2.0), "lin_vel_y": (-1.0, 1.0), "ang_vel_yaw": (-2.0, 2.0)}
        for name in ("target_12", "stair_curriculum", "near_low", "near_high", "near_narrow", "near_wide", "flat")}


def reset_stair_root(env, env_ids, pose_range, velocity_range):
    command = env.command_manager.get_term("base_velocity")
    flat = env_ids[~command.stair_mask[env_ids]]
    if len(flat):
        reset_root_state_uniform(env, flat, pose_range, velocity_range)
    ids = env_ids[command.stair_mask[env_ids]]
    if not len(ids):
        return
    asset = env.scene["robot"]
    state = asset.data.default_root_state[ids].clone()
    state[:, :3] += env.scene.env_origins[ids]
    state[:, 0] += torch.empty(len(ids), device=env.device).uniform_(-0.10, 0.10)
    state[:, 1] += torch.empty(len(ids), device=env.device).uniform_(-0.10, 0.10)
    yaw = torch.empty(len(ids), device=env.device).uniform_(-math.pi / 36, math.pi / 36)
    zeros = torch.zeros_like(yaw)
    state[:, 3:7] = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    state[:, 7:13] = 0.0
    asset.write_root_pose_to_sim(state[:, :7], env_ids=ids)
    asset.write_root_velocity_to_sim(state[:, 7:13], env_ids=ids)


def stair_completed(env):
    command = env.command_manager.get_term("base_velocity")
    command.update_course()
    return command.completed()


def stair_failed(env):
    command = env.command_manager.get_term("base_velocity")
    command.update_course()
    return command.course.failed


def stair_completion_curriculum(env, env_ids):
    command = env.command_manager.get_term("base_velocity")
    terrain = env.scene.terrain
    ids = env_ids[command.curriculum_mask[env_ids] & (env.episode_length_buf[env_ids] > 0)]
    if len(ids):
        command.cohort.record(terrain.terrain_levels[ids], command.completed()[ids])
        # Move at most one level on reset; do not teleport live episodes or randomize the final level.
        terrain.terrain_levels[ids] = torch.minimum(terrain.terrain_levels[ids] + 1,
                                                    torch.full_like(terrain.terrain_levels[ids], command.cohort.unlocked_level))
        terrain.env_origins[ids] = terrain.terrain_origins[terrain.terrain_levels[ids], terrain.terrain_types[ids]]
    return float(command.cohort.unlocked_level)


def legacy_critic_contact_forces(env, sensor_cfg):
    """Preserve the 219k critic's max + three-frame contact observation (16 values)."""
    forces = env.scene.sensors[sensor_cfg.name].data.net_forces_w_history[:, :3, sensor_cfg.body_ids]
    norms = torch.linalg.vector_norm(forces, dim=-1)
    return torch.cat((norms.amax(dim=1, keepdim=True), norms), dim=1).flatten(start_dim=-2)
