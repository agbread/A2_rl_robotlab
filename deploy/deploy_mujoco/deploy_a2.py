"""Run CTS A2 policy deployment in MuJoCo with joystick command input.

Overview:
This script loads a TorchScript policy, steps MuJoCo simulation with PD control, and
queries the policy at control decimation using only current-frame observations. The
exported policy manages internal observation history for CTS inference.

Differences from deploy_go2.py:
    - a2.xml orders qpos (FL, RL, FR, RR) and actuators (FR, FL, RR, RL) differently, so
      joints are mapped to the policy order by name for state reads and ctrl writes separately.
    - Applies the UnitreeActuator torque-speed curve and a per-group actuator delay at
      simulation-step resolution, as in IsaacLab training (A2_CFG).

Quick Start:
    python deploy/deploy_mujoco/deploy_a2.py
    python deploy/deploy_mujoco/deploy_a2.py --xml resources/a2_description/flat.xml --cmd 1.0 0 0
    python deploy/deploy_mujoco/deploy_a2.py --headless --duration 20  # no viewer, prints a tracking summary
"""

import argparse
import os
import sys
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace

import mujoco
import mujoco.viewer
import numpy as np
import torch
import yaml

from utils import (
    CONFIG_DIR,
    MujocoRenderUtils,
    display_current_command,
    gravity_from_quat,
    init_joystick,
    load_config,
    open_video_writer,
    read_joystick_command,
    setup_tracking_camera,
)


CONFIG_NAME = "a2.yaml"
VIDEO_DIR = Path(__file__).with_name("videos")


def parse_args():
    parser = argparse.ArgumentParser(description="Deploy an A2 CTS policy in MuJoCo.")
    parser.add_argument("--config", default=CONFIG_NAME, help="Config file name in deploy/deploy_mujoco/configs.")
    parser.add_argument("--policy", default=None, help="Override policy_path (TorchScript policy.pt).")
    parser.add_argument("--xml", default=None, help="Override xml_path (MuJoCo scene).")
    parser.add_argument("--cmd", type=float, nargs=3, default=None, metavar=("VX", "VY", "WZ"),
                        help="Fixed command [m/s, m/s, rad/s]; disables joystick input.")
    parser.add_argument("--duration", type=float, default=None,
                        help="Override simulation_duration [s] (headless default: 20).")
    parser.add_argument("--headless", action="store_true", help="Run without viewer and print a tracking summary.")
    parser.add_argument("--save_video", action="store_true", help="Record video to deploy/deploy_mujoco/videos.")
    parser.add_argument("--zero_joint_friction", action="store_true",
                        help="Remove MJCF joint damping/frictionloss (not modeled in IsaacLab training).")
    return parser.parse_args()


def load_a2_config(config_name: str, args):
    """Load the shared deployment config plus A2-specific actuator and joint-order entries."""
    cfg = load_config(config_name)
    with (CONFIG_DIR / config_name).open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file)
    cfg.model_joint_names = list(raw["model_joint_names"])
    cfg.knee_velocity = np.asarray(raw["knee_velocity"], dtype=np.float64)
    cfg.no_load_velocity = np.asarray(raw["no_load_velocity"], dtype=np.float64)
    cfg.zero_joint_friction = bool(raw.get("zero_joint_friction", False)) or args.zero_joint_friction
    cfg.delay_ms_min = float(raw.get("actuator_delay_ms_min", 0.0))
    cfg.delay_ms_max = float(raw.get("actuator_delay_ms_max", 0.0))
    if args.policy is not None:
        cfg.policy_path = Path(args.policy).resolve()
    if args.xml is not None:
        cfg.xml_path = Path(args.xml).resolve()
    if args.duration is not None:
        cfg.duration = args.duration
    elif args.headless:
        cfg.duration = 20.0
    if args.save_video:
        cfg.save_video = True
    return cfg


def build_joint_maps(model, joint_names: list[str]):
    """Return qpos addresses, qvel addresses and actuator indices in policy joint order."""
    qpos_adr = np.array([model.joint(name).qposadr[0] for name in joint_names], dtype=np.int64)
    qvel_adr = np.array([model.joint(name).dofadr[0] for name in joint_names], dtype=np.int64)
    actuator_of_joint = {model.joint(model.actuator_trnid[i, 0]).name: i for i in range(model.nu)}
    ctrl_idx = np.array([actuator_of_joint[name] for name in joint_names], dtype=np.int64)
    return qpos_adr, qvel_adr, ctrl_idx


def sample_delay_steps(cfg, rng: np.random.Generator) -> np.ndarray:
    """Per-joint actuation delay in simulation steps.

    IsaacLab's DelayedPDActuator samples one lag per actuator group (A2: hip+thigh, calf),
    so the same is done here and held for the whole run.
    """
    is_calf = np.array(["calf" in name for name in cfg.model_joint_names])
    lag_ms = rng.uniform(cfg.delay_ms_min, cfg.delay_ms_max, size=2) if cfg.delay_ms_max > 0 else np.zeros(2)
    steps = np.rint(np.where(is_calf, lag_ms[1], lag_ms[0]) * 1e-3 / cfg.dt).astype(np.int64)
    print(f"Actuator delay | hip+thigh: {steps[~is_calf][0] * cfg.dt * 1e3:.0f} ms  calf: {steps[is_calf][0] * cfg.dt * 1e3:.0f} ms")
    return steps


def clip_torque(tau: np.ndarray, dq: np.ndarray, cfg) -> np.ndarray:
    """UnitreeActuator._clip_effort; A2 uses the same peak torque in both directions (Y1 == Y2)."""
    speed = np.abs(dq)
    falloff = cfg.torque_limit * (cfg.no_load_velocity - speed) / (cfg.no_load_velocity - cfg.knee_velocity)
    limit = np.where(speed < cfg.knee_velocity, cfg.torque_limit, np.clip(falloff, 0.0, None))
    return np.clip(tau, -limit, limit)


def build_single_obs(data, action: np.ndarray, cmd: np.ndarray, cfg, qpos_adr, qvel_adr) -> np.ndarray:
    """Current-frame observation in IsaacLab PolicyCfg term order (all joint terms in policy order)."""
    return np.concatenate([
        data.qvel[3:6] * cfg.ang_vel_scale,
        gravity_from_quat(data.qpos[3:7]),
        cmd * cfg.cmd_scale,
        (data.qpos[qpos_adr] - cfg.default_angles) * cfg.dof_pos_scale,
        data.qvel[qvel_adr] * cfg.dof_vel_scale,
        action,
    ]).astype(np.float32)


def base_velocity_body(data) -> np.ndarray:
    """Base (vx, vy, wz) in the body frame."""
    lin_vel = np.zeros(3)
    inv_quat = np.zeros(4)
    mujoco.mju_negQuat(inv_quat, data.qpos[3:7])
    mujoco.mju_rotVecQuat(lin_vel, data.qvel[0:3], inv_quat)
    return np.array([lin_vel[0], lin_vel[1], data.qvel[5]])


def main() -> None:
    """Run MuJoCo simulation and deploy the CTS policy in closed-loop control."""
    args = parse_args()
    cfg = load_a2_config(args.config, args)
    joystick = None if (args.cmd is not None or args.headless) else init_joystick()
    cmd = np.asarray(args.cmd, dtype=np.float32) if args.cmd is not None else cfg.cmd_init.copy()

    model = mujoco.MjModel.from_xml_path(str(cfg.xml_path))
    model.opt.timestep = cfg.dt
    if cfg.zero_joint_friction:
        model.dof_damping[6:] = 0.0
        model.dof_frictionloss[6:] = 0.0
    data = mujoco.MjData(model)
    qpos_adr, qvel_adr, ctrl_idx = build_joint_maps(model, cfg.model_joint_names)

    data.qpos[:3] = cfg.base_init_pos
    data.qpos[3:7] = cfg.base_init_quat
    data.qpos[qpos_adr] = cfg.default_angles
    mujoco.mj_forward(model, data)

    policy = torch.jit.load(str(cfg.policy_path))
    policy.reset()
    print(f"Policy: {cfg.policy_path}\nScene:  {cfg.xml_path}")
    writer, frame_skip, video_path = open_video_writer(
        cfg.save_video, policy_path=cfg.policy_path, cmd=cmd, dt=cfg.dt, video_dir=VIDEO_DIR, video_fps=cfg.video_fps
    )
    renderer = mujoco.Renderer(model, height=360, width=640) if writer else None

    action = np.zeros(cfg.num_actions, dtype=np.float32)
    target_pos = cfg.default_angles.astype(np.float64)
    delay_steps = sample_delay_steps(cfg, np.random.default_rng(cfg.delay_seed))
    target_history = deque([target_pos.copy()] * (int(delay_steps.max()) + 1), maxlen=int(delay_steps.max()) + 1)
    joint_ids = np.arange(cfg.num_actions)
    render_substeps = max(1, int((1.0 / cfg.render_fps) / cfg.dt))
    mujoco_render_utils = MujocoRenderUtils()

    stats = SimpleNamespace(vel=deque(maxlen=int(5.0 / cfg.dt)), max_height=-np.inf, fell=False)

    def step(counter: int):
        nonlocal action, target_pos, cmd
        if counter % cfg.decimation == 0:
            if joystick:
                cmd = read_joystick_command(joystick, cfg.max_cmd)
            single_obs = build_single_obs(data, action, cmd, cfg, qpos_adr, qvel_adr)
            with torch.no_grad():
                action = policy(torch.from_numpy(single_obs).unsqueeze(0)).numpy().squeeze()
            target_pos = cfg.default_angles + action * cfg.action_pos_scale
            if not args.headless:
                display_current_command(cmd)
        target_history.append(target_pos)
        delayed_target = np.stack(target_history)[-1 - delay_steps, joint_ids]
        q, dq = data.qpos[qpos_adr], data.qvel[qvel_adr]
        data.ctrl[ctrl_idx] = clip_torque(cfg.kps * (delayed_target - q) - cfg.kds * dq, dq, cfg)
        mujoco.mj_step(model, data)
        mujoco_render_utils.update(cmd, data)

        stats.max_height = max(stats.max_height, data.qpos[2])
        stats.vel.append(base_velocity_body(data))
        if gravity_from_quat(data.qpos[3:7])[2] > -0.5:
            stats.fell = True

        if writer and counter % frame_skip == 0:
            try:
                renderer.update_scene(data, camera=render_cam)
                mujoco_render_utils.update_external_rendering(renderer, ctype='renderer')
                writer.append_data(renderer.render())
            except Exception as exc:
                print(f"Render error: {exc}")

    counter = 0
    num_steps = int(cfg.duration / cfg.dt)
    if args.headless:
        render_cam = mujoco.MjvCamera()
        setup_tracking_camera(SimpleNamespace(cam=render_cam))
        while counter < num_steps and not stats.fell:
            step(counter)
            counter += 1
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            setup_tracking_camera(viewer)
            render_cam = viewer.cam
            start_time = time.time()
            while viewer.is_running() and counter < num_steps:
                step(counter)
                counter += 1
                if counter % render_substeps == 0:
                    mujoco_render_utils.update_external_rendering(viewer, ctype='viewer')
                    viewer.sync()
                # keep simulation time in step with wall-clock time
                sleep_time = start_time + counter * cfg.dt - time.time()
                if sleep_time > 0:
                    time.sleep(sleep_time)

    print()
    mean_vel = np.mean(stats.vel, axis=0) if stats.vel else np.full(3, np.nan)
    print(f"Summary | sim {counter * cfg.dt:.1f}s | cmd ({cmd[0]:.2f}, {cmd[1]:.2f}, {cmd[2]:.2f}) | "
          f"mean body vel, last 5s ({mean_vel[0]:+.2f}, {mean_vel[1]:+.2f}, {mean_vel[2]:+.2f}) | "
          f"base x={data.qpos[0]:.2f} z={data.qpos[2]:.2f} max_z={stats.max_height:.2f} | "
          f"{'FELL' if stats.fell else 'ok'}")
    if writer:
        writer.close()
        print(f"Video saved: {video_path}")
    if not args.headless:
        # The passive viewer segfaults during interpreter shutdown (seen with mujoco 3.15 here);
        # everything is written by now, so skip that teardown.
        sys.stdout.flush()
        os._exit(0)


if __name__ == "__main__":
    main()
