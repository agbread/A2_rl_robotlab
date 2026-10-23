"""Measure A2 policy mirror loss on a fixed bank of recorded observations.

This is an open-loop diagnostic of the training objective, not a gait or stair
success evaluation. Reuse the SAME NPZ inputs when comparing candidate policies.
Each NPZ must contain one uninterrupted deployment episode beginning with a
reset history, or provide its exact 450-D history through --history-key.

Example for the previous eight-riser recordings:
  python scripts/a2_stairs/measure_mirror_residual.py --policy exported.pt \
    --inputs outputs/a2_gait_analysis_2026-10-23 --recursive --layout legacy \
    --output outputs/mirror.json
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np
import torch
from tensordict import TensorDict


ROOT = Path(__file__).resolve().parents[2]
TERM_WIDTHS = (3, 3, 3, 12, 12, 12)
JOINT_NAMES = [f"{leg}_{joint}" for leg in ("FL", "FR", "RL", "RR") for joint in ("hip", "thigh", "calf")]


def load_pure_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


symmetry = load_pure_module(
    "a2_residual_symmetry", ROOT / "source/robot_lab/robot_lab/tasks/a2/mdp/symmetry.py"
)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def make_history(single):
    """Zero-reset history, oldest-to-newest inside each observation term."""
    history = np.zeros((len(single), 450), dtype=np.float32)
    frames = np.zeros((10, 45), dtype=np.float32)
    for index, current in enumerate(single):
        frames[:-1] = frames[1:]
        frames[-1] = current
        blocks, offset = [], 0
        for width in TERM_WIDTHS:
            blocks.append(frames[:, offset:offset + width].reshape(-1))
            offset += width
        history[index] = np.concatenate(blocks)
    return history


def predict_stateless(model, obs, batch_size=512):
    """Call encoder and actor directly; never advance the JIT history buffer."""
    outputs = []
    with torch.inference_mode():
        for start in range(0, obs.batch_size[0], batch_size):
            batch = obs[start:start + batch_size]
            latent, _ = model.student_moe_encoder(model.actor_obs_normalizer(batch["policy"]))
            mean = model.actor(torch.cat([latent, model.single_obs_normalizer(batch["single_obs"])], -1))
            if getattr(model, "state_dependent_std", False):
                mean = mean[..., 0, :]
            outputs.append(mean.cpu())
    return torch.cat(outputs)


def summarize(residual, action_scale):
    error = np.asarray(residual, dtype=np.float64)
    if not len(error):
        return None
    mse = float(np.mean(error ** 2))
    rmse = float(np.sqrt(mse))
    return {
        "samples": len(error),
        "action_dimension_mean_mse": mse,
        "action_rmse": rmse,
        "action_mae": float(np.mean(np.abs(error))),
        "action_abs_p95": float(np.quantile(np.abs(error), 0.95)),
        "action_abs_max": float(np.max(np.abs(error))),
        "scaled_joint_target_rmse_rad": rmse * action_scale,
        "scaled_joint_target_rmse_deg": float(np.rad2deg(rmse * action_scale)),
        "per_joint_action_rmse": np.sqrt(np.mean(error ** 2, axis=0)).tolist(),
        "per_joint_scaled_target_rmse_rad": (np.sqrt(np.mean(error ** 2, axis=0)) * action_scale).tolist(),
        "per_joint_signed_action_residual_mean": error.mean(axis=0).tolist(),
    }


def segment_masks(data, path, layout):
    count = len(data["obs"])
    t, x = data["time"], data["base_pos"][:, 0]
    masks = {
        "all": np.ones(count, dtype=bool),
        "after_initial_history": np.arange(count) >= 10,
        "after_2s": t >= 2,
    }
    if path.stem.startswith("flat"):
        masks["flat"] = t >= 2
        return masks
    if layout == "legacy":
        masks.update({
            "approach": x < 1.2,
            "ascent": (x >= 1.2) & (x < 2.8),
            "landing": (x >= 2.8) & (x < 3.6),
            "descent": (x >= 3.6) & (x < 5.2),
            "exit": x >= 5.2,
        })
    else:
        geometry = load_pure_module(
            "a2_residual_geometry", ROOT / "source/robot_lab/robot_lab/tasks/a2/stair_geometry.py"
        )
        spec = geometry.get_stair_spec()
        moving = data["command"][:, 0] > 0 if "command" in data else np.ones(count, dtype=bool)
        masks.update({
            "approach": (x < spec.approach) & moving,
            "ascent": (x >= spec.approach) & (x < spec.approach + spec.num_steps * spec.tread) & moving,
            "landing": (x >= spec.approach + spec.num_steps * spec.tread) & moving,
            "settling": ~moving,
        })
    return masks


def verify_checkpoint(model, checkpoint):
    saved = torch.load(checkpoint, weights_only=True, map_location="cpu")
    state = saved["model_state_dict"]
    entries = []
    for name, parameter in model.named_parameters():
        equal = name in state and torch.equal(parameter, state[name])
        entries.append({"name": name, "elements": parameter.numel(), "bitwise_equal": equal})
    if not entries or not all(entry["bitwise_equal"] for entry in entries):
        raise ValueError("Exported actor/student parameters differ from the supplied checkpoint.")
    return {
        "path": str(checkpoint.resolve()), "sha256": digest(checkpoint), "iteration": saved.get("iter"),
        "tensor_count": len(entries), "parameter_count": sum(entry["elements"] for entry in entries),
        "all_exported_parameters_bitwise_equal": True, "parameters": entries,
    }


def measure(args):
    if args.action_scale <= 0:
        raise ValueError("action_scale must be positive.")
    torch.set_num_threads(args.threads)
    model = torch.jit.load(str(args.policy), map_location="cpu").eval()
    buffers_before = {name: tensor.clone() for name, tensor in model.named_buffers()}
    paths = []
    for path in args.inputs:
        paths.extend((path.rglob("*.npz") if args.recursive else path.glob("*.npz")) if path.is_dir() else [path])
    paths = sorted(set(path.resolve() for path in paths))
    if not paths:
        raise ValueError("No input NPZ files found.")
    report = {
        "definition": "residual = mu_student(M(history), M(single_obs)) - M(mu_student(history, single_obs))",
        "interpretation": "Open-loop loss-objective diagnostic on a fixed observation bank. Not a substitute for closed-loop gait, tracking, safety, or stair success.",
        "policy": {"path": str(args.policy.resolve()), "sha256": digest(args.policy)},
        "layout": args.layout, "action_scale_rad_per_unit": args.action_scale,
        "units": "Actions are normalized joint-position offsets. Scaled residuals are target-angle differences in radians, not measured joint tracking errors.",
        "history": "450-D term-major, ten oldest-to-newest frames per term; single_obs stays independent. Zero prehistory per NPZ unless --history-key supplied.",
        "history_key": args.history_key,
        "timing": "Legacy recorded base_pos/time are one 0.02-s control interval after the corresponding observation/action call; segment labels use those recorded positions.",
        "joint_order": JOINT_NAMES,
        "training_loss_relation": "Unweighted action_dimension_mean_mse equals the bidirectional stopped-target mirror loss value for this involutive signed permutation; gradients differ by optimizer pass.",
        "recorded_action_difference_note": "Difference to recorded behavior actions validates baseline replay only for an identical controller. Candidate policies or reflection/ trajectories are expected to differ.",
        "runs": {}, "pooled_nominal_v0.5": {},
    }
    if args.checkpoint:
        report["checkpoint_verification"] = verify_checkpoint(model, args.checkpoint)
    pooled = {"flat": [], "ascent": []}
    for path in paths:
        with np.load(path, allow_pickle=False) as stored:
            data = {key: stored[key] for key in stored.files}
        single = np.asarray(data["obs"], dtype=np.float32)
        if single.ndim != 2 or single.shape[1] != 45 or not len(single):
            raise ValueError(f"Expected a nonempty (T,45) obs array in {path}.")
        history = np.asarray(data[args.history_key], dtype=np.float32) if args.history_key else make_history(single)
        obs = TensorDict({"policy": torch.from_numpy(history), "single_obs": torch.from_numpy(single)}, [len(single)])
        mirrored, _ = symmetry.mirror_a2_observations(obs=obs)
        restored, _ = symmetry.mirror_a2_observations(obs=mirrored)
        if not all(torch.equal(obs[key], restored[key]) for key in obs.keys()):
            raise AssertionError("Observation reflection is not an involution.")
        original_mean = predict_stateless(model, obs)
        mirrored_mean = predict_stateless(model, mirrored)
        _, expected = symmetry.mirror_a2_observations(actions=original_mean)
        residual = (mirrored_mean - expected).numpy()
        if not np.isfinite(residual).all():
            raise ValueError(f"Non-finite action residual in {path}.")
        masks = segment_masks(data, path, args.layout)
        # A full path disambiguates the six reflected-controller runs from originals.
        label = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)
        item = {
            "path": str(path), "sha256": digest(path),
            "segments": {name: summarize(residual[mask], args.action_scale) for name, mask in masks.items() if mask.any()},
        }
        if "action" in data:
            delta = original_mean.numpy().astype(np.float64) - data["action"]
            item["recorded_action_difference"] = {
                "rmse": float(np.sqrt(np.mean(delta ** 2))), "max_abs": float(np.abs(delta).max()),
            }
        report["runs"][label] = item
        if "reflection" not in path.parts and re.fullmatch(r"(flat|stairs)_v0\.5_s\d+", path.stem):
            segment = "flat" if path.stem.startswith("flat") else "ascent"
            pooled[segment].append(residual[masks[segment]])
    for name, chunks in pooled.items():
        if chunks:
            report["pooled_nominal_v0.5"][name] = summarize(np.concatenate(chunks), args.action_scale)
            report["pooled_nominal_v0.5"][name]["run_count"] = len(chunks)
    report["run_count"] = len(paths)
    report["model_buffers_unchanged"] = all(
        torch.equal(buffers_before[name], tensor) for name, tensor in model.named_buffers()
    )
    if not report["model_buffers_unchanged"]:
        raise AssertionError("Stateless diagnostic changed model buffers.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"output": str(args.output), "run_count": len(paths), "pooled_nominal_v0.5": report["pooled_nominal_v0.5"]}, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, help="Optionally verify actor/student weights against this checkpoint.")
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--recursive", action="store_true", help="Include subdirectories, e.g. legacy reflection diagnostics.")
    parser.add_argument("--layout", choices=("legacy", "stairs12"), default="stairs12")
    parser.add_argument("--history-key", help="NPZ key for exact (T,450) histories; otherwise reconstruct zero-reset history.")
    parser.add_argument("--action-scale", type=float, default=0.25)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, required=True)
    measure(parser.parse_args())


if __name__ == "__main__":
    main()
