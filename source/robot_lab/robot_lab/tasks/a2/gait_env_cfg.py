"""Opt-in gait tasks. Import only after starting Isaac Sim."""
from collections import deque
import torch
from isaaclab.managers import RewardTermCfg as RewTerm, TerminationTermCfg as DoneTerm, CurriculumTermCfg as CurrTerm
from isaaclab.utils import configclass
from .stairs_env_cfg import A2StairsEnvCfg
from .env.stairs_env import A2StairsEnv
from .gait_logic import GaitStatistics
from .mdp.stairs import StairVelocityCommand, StairVelocityCommandCfg, make_stair_terrain_cfg
from .mdp.commands import A2RLGymCommand

STAGES = {"stairs_2_5": (2, .05), "stairs_4_10": (4, .10), "stairs_6_15": (6, .15),
          "stairs_8_15": (8, .15), "stairs_10_15": (10, .15), "stairs_12_15": (12, .15)}
MIX = dict(target_12=.20, stair_curriculum=.10, near_low=.05, near_high=.05,
           near_narrow=.05, near_wide=.05, flat=.50)


class GaitVelocityCommand(StairVelocityCommand):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.gait_stats = GaitStatistics(self.num_envs, env.step_dt, self.device)
        self.gait_step = -1
        self.gait_losses = None
        self.recent_student_stairs = deque(maxlen=200)

    def _resample(self, env_ids):
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        if not len(env_ids):
            return
        flat = env_ids[~self.stair_mask[env_ids]]
        if len(flat):
            # Preserve the legacy distribution for the 20% replay component.
            A2RLGymCommand._resample(self, flat)
            straight = flat[torch.rand(len(flat), device=self.device) < self.cfg.straight_probability]
            self.commands[straight] = 0
            self.commands[straight, 0] = torch.empty(len(straight), device=self.device).uniform_(*self.cfg.flat_speed)
            self.time_left[straight] = 5.
        stairs = env_ids[self.stair_mask[env_ids]]
        self.commands[stairs] = 0
        self.commands[stairs, 0] = torch.empty(len(stairs), device=self.device).uniform_(*self.cfg.forward_speed)
        self.time_left[stairs] = self._env.max_episode_length_s + 1
        if hasattr(self, "gait_stats"):
            self.gait_stats.reset(env_ids)

    def reset(self, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        if hasattr(self, "recent_student_stairs"):
            student = getattr(self._env, "student_env_mask", torch.zeros_like(self.stair_mask))
            measured = ids[(self._env.episode_length_buf[ids] > 0) & self.stair_mask[ids] & student[ids]]
            self.recent_student_stairs.extend(self.completed()[measured].detach().cpu().tolist())
            self.gait_stats.reset(ids)
        return super().reset(env_ids)

    def gait_penalties(self):
        if self.gait_step != self._env.common_step_counter:
            force = self.contact_sensor.data.net_forces_w_history[:, :self._env.cfg.decimation, self.contact_ids, 2]
            self.gait_losses = self.gait_stats.update((force > 10).float().mean(1), self.command, ~self.stair_mask)
            self.gait_step = self._env.common_step_counter
        updates = self._env.common_step_counter / 24 + self._env.cfg.gait_update_offset
        ramp = min(1., updates / self._env.cfg.gait_ramp_updates)
        return tuple(value * ramp for value in self.gait_losses)


@configclass
class GaitVelocityCommandCfg(StairVelocityCommandCfg):
    class_type: type = GaitVelocityCommand
    flat_speed: tuple[float, float] = (.5, 1.2)
    straight_probability: float = .8
    expected_columns: dict = {name: round(p * 20) for name, p in MIX.items()}


def swing_balance(env):
    return env.command_manager.get_term("base_velocity").gait_penalties()[0]


def duty_balance(env):
    return env.command_manager.get_term("base_velocity").gait_penalties()[1]


def flat_boundary(env):
    term = env.command_manager.get_term("base_velocity")
    xy = env.scene["robot"].data.root_pos_w[:, :2] - env.scene.env_origins[:, :2]
    size = env.scene.terrain.cfg.terrain_generator.size
    return ~term.stair_mask & ((xy.abs() > xy.new_tensor(size) / 2 - 1.).any(-1))


def gait_reward_schedule(env, env_ids, term_name, initial, final, end_update):
    update = env.common_step_counter // 24 + env.cfg.gait_update_offset
    weight = initial + (final - initial) * min(1., update / end_update)
    term = env.reward_manager.get_term_cfg(term_name)
    term.weight = weight
    env.reward_manager.set_term_cfg(term_name, term)
    return weight


def configure_gait(cfg, branch, phase, update_offset=0):
    if phase not in ("mixed", "flat_slow", "flat_fast", *STAGES):
        raise ValueError(f"Unknown gait phase: {phase}")
    if branch not in ("warm", "scratch") or (branch == "warm" and phase != "mixed"):
        raise ValueError("Invalid gait branch/phase")
    cfg.gait_branch, cfg.gait_phase = branch, phase
    cfg.gait_update_offset = int(update_offset)
    cfg.gait_ramp_updates = 200 if branch == "warm" else 1000
    cfg.commands.base_velocity = GaitVelocityCommandCfg()
    cfg.scene.terrain.terrain_generator = terrain = make_stair_terrain_cfg()
    proportions = dict(MIX)
    if phase.startswith("flat"):
        proportions = {name: float(name == "flat") for name in MIX}
    elif phase in STAGES:
        proportions = {name: .5 if name in ("flat", "stair_curriculum") else 0. for name in MIX}
        stair = terrain.sub_terrains["stair_curriculum"]
        stair.curriculum_mode = "fixed"
        stair.num_steps, stair.rise = STAGES[phase]
    for name, proportion in proportions.items():
        terrain.sub_terrains[name].proportion = proportion
    cfg.commands.base_velocity.expected_columns = {name: round(p * 20) for name, p in proportions.items()}
    cfg.commands.base_velocity.flat_speed = (.2, .5) if phase == "flat_slow" else (.5, 1.2)
    cfg.commands.base_velocity.straight_probability = .8 if phase == "mixed" else 1.
    if phase != "mixed":
        cfg.curriculum.terrain_levels = None
    cfg.rewards.gait_swing_balance = RewTerm(func=swing_balance, weight=-.2)
    cfg.rewards.gait_duty_balance = RewTerm(func=duty_balance, weight=-1.)
    cfg.terminations.flat_boundary = DoneTerm(func=flat_boundary, time_out=True)
    if branch == "scratch":
        cfg.curriculum.base_linear_velocity = CurrTerm(func=gait_reward_schedule, params=dict(
            term_name="lin_vel_z_l2", initial=-2., final=0., end_update=1500))
        cfg.curriculum.base_height_l2 = CurrTerm(func=gait_reward_schedule, params=dict(
            term_name="base_height_l2", initial=-1., final=-10., end_update=5000))
        cfg.rewards.lin_vel_z_l2.weight = -2*max(0., 1-update_offset/1500)
        cfg.rewards.base_height_l2.weight = -1-9*min(1., update_offset/5000)


class A2GaitEnv(A2StairsEnv):
    def get_training_state(self):
        state = super().get_training_state()
        state["gait"] = dict(branch=self.cfg.gait_branch, phase=self.cfg.gait_phase,
                             update_offset=self.cfg.gait_update_offset,
                             recent_student_stairs=list(self.command_manager.get_term("base_velocity").recent_student_stairs))
        return state

    def set_training_state(self, state):
        gait = state.get("gait", {})
        if (gait.get("branch"), gait.get("phase"), gait.get("update_offset")) != (
                self.cfg.gait_branch, self.cfg.gait_phase, self.cfg.gait_update_offset):
            raise ValueError("Gait resume phase/offset differs: use an explicit phase transition")
        super().set_training_state(state)
        term = self.command_manager.get_term("base_velocity")
        term.recent_student_stairs.clear()
        term.recent_student_stairs.extend(gait["recent_student_stairs"])


@configclass
class A2GaitEnvCfg(A2StairsEnvCfg):
    gait_branch: str = "warm"
    gait_phase: str = "mixed"
    gait_update_offset: int = 0
    gait_ramp_updates: int = 200

    def __post_init__(self):
        super().__post_init__()
        configure_gait(self, "warm", "mixed")


@configclass
class A2GaitScratchEnvCfg(A2GaitEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_gait(self, "scratch", "flat_slow")
