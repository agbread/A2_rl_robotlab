"""Remote GPU worker. Never imported by preflight --static or dry-run.

Only the explicit worker entry point starts Isaac Sim. The parent owns its GPU,
deadline and process group. Checkpoints from trusted project sources only.
"""
from __future__ import annotations
import argparse
import math
import os
from pathlib import Path
import signal
import sys
import time

from common import recipe, read_json, write_json
from initialization import initialize, config_hash


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--request", type=Path, required=True)
    args = parser.parse_args()
    request = read_json(args.request)
    sys.argv = [sys.argv[0]]  # Do not pass our request-file flag into Kit.
    # Preload native extensions before Kit (same order as the project train script).
    import h5py  # noqa: F401
    import tensordict  # noqa: F401
    import torch
    physical_gpu = int(request["gpu"])
    torch.cuda.set_device(physical_gpu)
    import subprocess
    expected_uuid = subprocess.check_output([
        "nvidia-smi", "-i", str(physical_gpu), "--query-gpu=uuid", "--format=csv,noheader"], text=True).strip()
    actual_uuid = str(torch.cuda.get_device_properties(physical_gpu).uuid)
    if actual_uuid.lower().removeprefix("gpu-") != expected_uuid.lower().removeprefix("gpu-"):
        raise RuntimeError("CUDA/graphics physical GPU mapping differs; refusing to train on the wrong GPU")
    from isaaclab.app import AppLauncher
    launcher = AppLauncher(dict(headless=True, device=f"cuda:{physical_gpu}", enable_cameras=False,
                               multi_gpu=False, kit_args="--/renderer/multiGpu/enabled=false"))
    app = launcher.app
    try:
        if request["mode"] == "evaluate":
            from isaac_evaluate import evaluate
            result = evaluate(request, app)
        else:
            result = train(request, app)
        write_json(request["result"], result)
    finally:
        app.close()


def train(request, app):
    import torch
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from isaaclab.utils.io import dump_yaml
    from robot_lab.tasks.a2.gait_env_cfg import A2GaitEnv, A2GaitEnvCfg, configure_gait
    from robot_lab.tasks.a2.rsl_rl_cfg import GaitWarmRunnerCfg, GaitScratchRunnerCfg
    from rsl_rl.runners import OnPolicyRunnerCTS

    torch.set_num_threads(4)
    state = request["state"]
    spec = recipe(state["branch"], state["phase"])
    phase_entry = state.get("phase_entry", True)
    # Phase offsets preserve completed reward ramps without restoring old terrain.
    offset = state["updates"] if phase_entry else state["phase_offset"]
    cfg = A2GaitEnvCfg()
    configure_gait(cfg, state["branch"], state["phase"], offset)
    cfg.scene.num_envs = request["num_envs"]
    cfg.sim.device = f"cuda:{request['gpu']}"
    cfg.seed = state["seed"]
    out = Path(request["output"])
    out.mkdir(parents=True, exist_ok=True)
    cfg.log_dir = str(out)
    # Identically seeded simultaneous processes can otherwise pick the same
    # timestamp/random /tmp USD conversion directory.
    cfg.scene.robot.spawn.usd_dir = str(out / "assets/a2")
    cfg.scene.terrain.visual_material = None
    cfg.scene.sky_light.spawn.texture_file = None
    agent = GaitWarmRunnerCfg() if state["branch"] == "warm" else GaitScratchRunnerCfg()
    agent.seed = state["seed"]
    agent.device = cfg.sim.device
    agent.save_interval = 100 if state["branch"] == "warm" else 500
    agent.algorithm.learning_rate = spec["learning_rate"]
    agent.algorithm.student_encoder_learning_rate = spec["student_learning_rate"]
    agent.algorithm.schedule = spec["schedule"]
    agent.algorithm.symmetry_cfg["ramp_updates"] = spec["ramp_updates"]
    agent_cfg = agent.to_dict()
    agent_cfg["robogauge"] = {"enabled": False}
    env = A2GaitEnv(cfg)
    try:
        if request["mode"] == "probe":
            obs, _ = env.reset(seed=state["seed"])
            for _ in range(20):
                if not app.is_running():
                    raise RuntimeError("Simulator closed during probe")
                obs, _, _, _, _ = env.step(torch.zeros((env.num_envs, 12), device=env.device))
                if not all(torch.isfinite(value).all() for value in obs.values()):
                    raise FloatingPointError("Probe produced nonfinite observations")
            return dict(ok=True, trained=False, shapes={key: list(value.shape) for key, value in obs.items()})
        wrapped = RslRlVecEnvWrapper(env, clip_actions=agent.clip_actions)
        torch.manual_seed(state["seed"])
        runner = OnPolicyRunnerCTS(wrapped, agent_cfg, log_dir=str(out), device=cfg.sim.device)
        initialization = initialize(runner, state, request["num_envs"])
        dump_yaml(str(out / "params/env.yaml"), cfg)
        dump_yaml(str(out / "params/agent.yaml"), agent)
        start_it = runner.current_learning_iteration
        stop = [False]
        for sig in (signal.SIGTERM, signal.SIGINT):
            signal.signal(sig, lambda *_: stop.__setitem__(0, True))
        timings = []
        mode = request["mode"]
        last_update_wall = [time.monotonic()]

        def hook(r, it, collect_s, learn_s, losses):
            if not all(math.isfinite(float(v)) for v in losses.values()):
                raise FloatingPointError("Nonfinite training loss")
            if not all(torch.isfinite(p).all() for p in r.alg.policy.parameters()):
                raise FloatingPointError("Nonfinite policy weights")
            done = it - start_it + 1
            cumulative = state["updates"] + done
            # These are the original reward schedules, referenced to global progress
            # so phase changes do not restart the initial standing curriculum.
            if state["branch"] == "scratch":
                for name, weight in (("lin_vel_z_l2", -2*max(0., 1-cumulative/1500)),
                                     ("base_height_l2", -1-9*min(1., cumulative/5000))):
                    term = env.reward_manager.get_term_cfg(name)
                    term.weight = weight
                    env.reward_manager.set_term_cfg(name, term)
            term = env.command_manager.get_term("base_velocity")
            r.experiment_metadata = dict(
                branch=state["branch"], phase=state["phase"], seed=state["seed"], updates=cumulative,
                transitions=state["transitions"] + done*request["num_envs"]*24,
                phase_offset=offset, phase_entry=False, deadline=request["deadline"],
                config_hash=config_hash(state, request["num_envs"]), initialization=initialization,
                recent_student_stairs=list(term.recent_student_stairs))
            now_wall = time.monotonic()
            timings.append(now_wall - last_update_wall[0])
            last_update_wall[0] = now_wall
            write_json(out / "progress.json", dict(r.experiment_metadata, phase_iteration=it,
                                                   time=time.time(), loss=losses))
            return app.is_running() and not stop[0] and time.time() < request["deadline"]

        runner.training_update_hook = hook
        # Restore the reward ramp *before* the first rollout as well as on updates.
        if state["branch"] == "scratch":
            for name, weight in (("lin_vel_z_l2", -2*max(0., 1-state["updates"]/1500)),
                                 ("base_height_l2", -1-9*min(1., state["updates"]/5000))):
                term = env.reward_manager.get_term_cfg(name)
                term.weight = weight
                env.reward_manager.set_term_cfg(name, term)
        runner.learn(request["updates"], init_at_random_ep_len=False)
        result = dict(runner.experiment_metadata)
        result["checkpoint"] = str(out / f"model_{runner.current_learning_iteration}.pt")
        result["timing_seconds"] = timings
        result["mode"] = mode
        result["num_envs"] = request["num_envs"]
        return result
    finally:
        env.close()


if __name__ == "__main__":
    # Kit shutdown can mask exception exit status: explicitly propagate errors.
    try:
        main()
    except BaseException:
        import traceback
        traceback.print_exc()
        sys.exit(1)
