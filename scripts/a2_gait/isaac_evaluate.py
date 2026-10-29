"""Remote-only nominal evaluation, independent of training population/curriculum."""
from pathlib import Path
import numpy as np
import torch
from tensordict import TensorDict
import yaml

from common import SPEEDS
from metrics import summarize


class Loader(yaml.SafeLoader):
    pass


Loader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node)))


def reset_evaluation(env, env_ids):
    robot = env.scene["robot"]
    root = robot.data.default_root_state[env_ids].clone()
    root[:, :3] += env.scene.env_origins[env_ids]
    root[:, 3:7] = root.new_tensor([1, 0, 0, 0])
    generator = torch.Generator().manual_seed(env.cfg.seed)
    perturbation = (torch.rand(64, robot.num_joints, generator=generator) * 2 - 1) * .02
    perturbation[0] = 0
    joints = robot.data.default_joint_pos[env_ids] + perturbation.to(env.device)[env_ids % 64]
    robot.write_root_pose_to_sim(root[:, :7], env_ids=env_ids)
    robot.write_root_velocity_to_sim(torch.zeros_like(root[:, 7:]), env_ids=env_ids)
    robot.write_joint_state_to_sim(joints, torch.zeros_like(joints), env_ids=env_ids)


def evaluate(request, app):
    from isaaclab.envs.mdp.commands import UniformVelocityCommand, UniformVelocityCommandCfg
    from isaaclab.managers import EventTermCfg
    from isaaclab.utils import configclass
    from isaaclab.utils.math import euler_xyz_from_quat
    from robot_lab.tasks.a2.env_cfg import A2EnvCfg
    from robot_lab.tasks.a2.env.a2_env import A2Env
    from robot_lab.tasks.a2.stairs_env_cfg import A2StairsEnvCfg
    from robot_lab.tasks.a2.gait_env_cfg import GaitVelocityCommand, GaitVelocityCommandCfg
    from robot_lab.tasks.a2.mdp.stairs import legacy_critic_contact_forces
    from rsl_rl.modules import ActorCriticMoECTS

    speeds = request.get("speeds", SPEEDS)
    per_speed = 64
    count = len(speeds)*per_speed

    class FlatCommand(UniformVelocityCommand):
        def _resample_command(self, ids):
            self.vel_command_b[ids] = 0
            self.vel_command_b[ids, 0] = torch.tensor(speeds, device=self.device)[ids // per_speed]

    class StairCommand(GaitVelocityCommand):
        def _resample(self, ids):
            self.commands[ids] = 0
            self.commands[ids, 0] = torch.tensor(speeds, device=self.device)[ids // per_speed]
            self.time_left[ids] = 1000

    @configclass
    class FlatCommandCfg(UniformVelocityCommandCfg):
        class_type: type = FlatCommand

    @configclass
    class StairCommandCfg(GaitVelocityCommandCfg):
        class_type: type = StairCommand

    stairs = request["terrain"] == "stairs"
    cfg = A2StairsEnvCfg() if stairs else A2EnvCfg()
    cfg.sim.device = f"cuda:{request['gpu']}"
    cfg.scene.num_envs = count
    cfg.scene.env_spacing = 5.
    cfg.seed = request.get("eval_seed", 42)
    cfg.log_dir = request["output"]
    cfg.scene.robot.spawn.usd_dir = str(Path(request["output"]) / "assets/a2")
    cfg.scene.terrain.visual_material = None
    cfg.episode_length_s = request.get("seconds", 30.) + 10
    cfg.scene.contact_forces.history_length = cfg.decimation
    cfg.observations.critic.contact_force.func = legacy_critic_contact_forces
    for group in (cfg.observations.policy, cfg.observations.single_obs, cfg.observations.critic):
        group.enable_corruption = False
    for group in (cfg.events, cfg.curriculum, cfg.rewards):
        for name in vars(group):
            if not name.startswith("_"):
                setattr(group, name, None)
    cfg.events.reset_base = EventTermCfg(func=reset_evaluation, mode="reset")
    if stairs:
        terrain = cfg.scene.terrain.terrain_generator
        terrain.curriculum = False
        for name, sub in terrain.sub_terrains.items():
            sub.proportion = float(name == "target_12")
        cfg.commands.base_velocity = StairCommandCfg(
            expected_columns={name: 20 if name == "target_12" else 0 for name in terrain.sub_terrains},
            require_settled_completion=True)
    else:
        cfg.scene.terrain.terrain_type = "plane"
        cfg.scene.terrain.terrain_generator = None
        cfg.scene.terrain.max_init_terrain_level = None
        cfg.commands.base_velocity = FlatCommandCfg(
            asset_name="robot", resampling_time_range=(1000., 1000.), heading_command=False,
            rel_standing_envs=0., debug_vis=False,
            ranges=UniformVelocityCommandCfg.Ranges(lin_vel_x=(.5, 1.2), lin_vel_y=(0., 0.), ang_vel_z=(0., 0.)))
    cfg.scene.sky_light.spawn.texture_file = None
    cfg.scene.robot.actuators["hip_thigh"].min_delay = 3
    cfg.scene.robot.actuators["hip_thigh"].max_delay = 3
    cfg.scene.robot.actuators["calf"].min_delay = 2
    cfg.scene.robot.actuators["calf"].max_delay = 2
    env = A2Env(cfg)
    try:
        obs, _ = env.reset(seed=cfg.seed)
        obs = TensorDict(obs, [count])
        checkpoint = Path(request["checkpoint"])
        with (checkpoint.parent / "params/agent.yaml").open() as stream:
            policy_cfg = yaml.load(stream, Loader=Loader)["policy"]
        if policy_cfg.pop("class_name") != "ActorCriticMoECTS":
            raise ValueError("Expected a CTS checkpoint")
        policy = ActorCriticMoECTS(obs, {"policy": ["policy"], "critic": ["critic"]}, 12, **policy_cfg).to(env.device)
        saved = torch.load(checkpoint, map_location=env.device, weights_only=False)
        policy.load_state_dict(saved["model_state_dict"], strict=True)
        del saved
        policy.eval()
        robot = env.scene["robot"]
        sensor = env.scene["contact_forces"]
        names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        feet, actual = robot.find_bodies(names, preserve_order=True)
        contacts, _ = sensor.find_bodies(names, preserve_order=True)
        if actual != names or obs["policy"].shape[1] != 450 or obs["critic"].shape[1] != 275:
            raise ValueError("Unexpected observation or foot layout")
        rows = []
        for branch in request.get("policies", ["student", "teacher"]):
            obs, _ = env.reset(seed=cfg.seed)
            obs = TensorDict(obs, [count])
            alive = torch.ones(count, dtype=torch.bool, device=env.device)
            fallen = ~alive
            ascended, settled = ~alive, ~alive
            records = {k: [] for k in ("time", "fraction", "height", "velocity", "command", "yaw", "valid")}
            for step in range(round(request.get("seconds", 30.) / env.step_dt)):
                if not app.is_running():
                    raise RuntimeError("Isaac app closed during evaluation")
                import time
                if time.time() >= request["deadline"]:
                    raise TimeoutError("Evaluation budget expired")
                with torch.inference_mode():
                    command = env.command_manager.get_command("base_velocity").clone()
                    force = sensor.data.net_forces_w_history[:, :cfg.decimation, contacts, 2]
                    foot = robot.data.body_pos_w[:, feet] - env.scene.env_origins[:, None]
                    ground = torch.clamp(torch.floor((foot[:, :, 0] - 1.5)/.2)+1, 0, 12)*.15 if stairs else 0
                    yaw = euler_xyz_from_quat(robot.data.root_quat_w)[2]
                    values = (step*env.step_dt, (force > 10).float().mean(1), foot[:, :, 2]-.032-ground,
                              torch.cat([robot.data.root_lin_vel_b[:, :2], robot.data.root_ang_vel_b[:, 2:3]], -1),
                              command, torch.atan2(torch.sin(yaw), torch.cos(yaw))*180/np.pi, alive.clone())
                    for key, value in zip(records, values):
                        records[key].append(value.detach().cpu().numpy().copy() if isinstance(value, torch.Tensor) else value)
                    if stairs:
                        ascended |= env.command_manager.get_term("base_velocity").course.ascent_completed & alive
                    if branch == "teacher":
                        latent = policy.teacher_encoder(policy.critic_obs_normalizer(obs["critic"]))
                        action = policy.actor(torch.cat([latent, policy.single_obs_normalizer(obs["single_obs"])], -1))
                    else:
                        action = policy.act_inference(obs)
                    if not torch.isfinite(action).all():
                        raise FloatingPointError("Nonfinite policy output")
                    obs, _, terminated, truncated, _ = env.step(action)
                    obs = TensorDict(obs, [count])
                    if stairs:
                        complete = env.termination_manager.get_term("course_completed") & alive
                        settled |= complete
                        ascended |= complete
                    fallen |= terminated & alive
                    alive &= ~(terminated | truncated)
            arrays = {key: np.asarray(value) for key, value in records.items()}
            Path(request["output"]).mkdir(parents=True, exist_ok=True)
            np.savez_compressed(Path(request["output"]) / f"{branch}_trajectories.npz", **arrays)
            for index, speed in enumerate(speeds):
                ids = slice(index*per_speed, (index+1)*per_speed)
                row = summarize(arrays["time"], *(arrays[k][:, ids] for k in
                    ("fraction", "height", "velocity", "command", "yaw", "valid")),
                    (~fallen[ids]).cpu().numpy(), ascended[ids].cpu().numpy(), settled[ids].cpu().numpy())
                rows.append(dict(row, simulator="isaac", terrain=request["terrain"], policy=branch, speed=speed))
        return dict(rows=rows, protocol="nominal paired joint perturbations; no observation noise/pushes; fixed 15/10ms lags",
                    checkpoint=str(checkpoint), num_envs=count)
    finally:
        env.close()
