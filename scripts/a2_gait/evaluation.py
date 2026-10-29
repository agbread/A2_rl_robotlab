"""Evaluation orchestration. Heavy libraries are imported inside CPU functions."""
from pathlib import Path
import sys

from common import ROOT, SPEEDS, PROTOCOL, read_json, write_json, digest
from processes import gpu_environment, run_jobs


def command_job(argv, output, gpu):
    return dict(argv=[sys.executable, *map(str, argv)], cwd=str(ROOT), env=gpu_environment(gpu),
                log=str(output))


def worker_job(request, folder):
    folder = Path(folder)
    folder.mkdir(parents=True, exist_ok=True)
    request = dict(request, output=str(folder), result=str(folder / "result.json"))
    write_json(folder / "request.json", request)
    return command_job([ROOT / "scripts/a2_gait/worker.py", "--request", folder / "request.json"],
                       folder / "worker.log", request["gpu"])


def mujoco_rows(folder):
    import numpy as np
    from metrics import summarize
    cases = read_json(Path(folder) / "summary.json")
    groups = {}
    for case in cases:
        with np.load(Path(folder) / f"{case['name']}.npz", allow_pickle=False) as a:
            n = len(a["time"])
            fraction = a["stance_fraction_vertical"] if "stance_fraction_vertical" in a else a["stance_fraction"]
            row = summarize(a["time"], fraction[:, None], a["foot_clearance"][:, None],
                            a["velocity"][:, None], a["command"][:, None], a["rpy_deg"][:, None, 2],
                            np.ones((n, 1), dtype=bool), [case["status"] not in ("fall", "nonfinite_state")],
                            [case["ascent_completed"]], [case["settled_success"]])
        groups.setdefault((case["terrain"], case["speed"]), []).append(row)
    rows = []
    for (terrain, speed), values in groups.items():
        result = {}
        for field in values[0]:
            if field in ("n", "episodes_with_swing"):
                result[field] = np.sum([v[field] for v in values], axis=0).tolist()
            else:
                array = np.array([v[field] for v in values], dtype=float)
                count = np.isfinite(array).sum(0)
                mean = np.nansum(array, axis=0) / np.maximum(count, 1)
                def clean(x):
                    if isinstance(x, list):
                        return [clean(v) for v in x]
                    return x if np.isfinite(x) else None
                result[field] = clean(np.where(count > 0, mean, np.nan).tolist())
        rows.append(dict(result, simulator="mujoco", terrain=terrain, speed=speed, policy="student"))
    return rows


def suite(checkpoint, output, gpu, deadline, split="validation", quick=False):
    output, checkpoint = Path(output), Path(checkpoint).resolve()
    output.mkdir(parents=True, exist_ok=False)
    rows = []
    for terrain in (["flat"] if quick else ["flat", "stairs"]):
        folder = output / f"isaac_{terrain}"
        job = worker_job(dict(mode="evaluate", gpu=gpu, checkpoint=str(checkpoint), terrain=terrain,
                              deadline=deadline, eval_seed=42 if split == "validation" else 142,
                              seconds=20 if quick else 30), folder)
        run_jobs([job], deadline)
        rows.extend(read_json(folder / "result.json")["rows"])
    if not quick:
        policy = output / "exported/policy.pt"
        run_jobs([command_job([ROOT / "scripts/a2_stairs/export_checkpoint.py", "--checkpoint", checkpoint,
                               "--output", policy], output / "export.log", gpu)], deadline)
        for terrain in ("flat", "stairs"):
            folder = output / f"mujoco_{terrain}"
            argv = [ROOT / "scripts/a2_stairs/evaluate_mujoco.py", "--policy", policy,
                    "--output", folder, "--terrains", terrain, "--speeds", *SPEEDS,
                    "--flat-speeds", *SPEEDS, "--seeds", *PROTOCOL[f"{split}_seeds"],
                    "--duration", 30, "--workers", 2]
            run_jobs([command_job(argv, output / f"mujoco_{terrain}.log", gpu)], deadline)
            rows.extend(mujoco_rows(folder))
    report = dict(directory=str(output), checkpoint=str(checkpoint), checkpoint_sha256=digest(checkpoint), split=split,
                  quick=quick, protocol=PROTOCOL, rows=rows)
    write_json(output / "metrics.json", report)
    return report


def cached_suite(checkpoint, cache, gpu, deadline, split="validation"):
    """Cache only completed reports; interrupted directories remain recoverable."""
    import time
    cache = Path(cache)
    if cache.exists():
        report = read_json(cache)
        if report["checkpoint_sha256"] != digest(checkpoint) or report["protocol"] != PROTOCOL or report["split"] != split:
            raise ValueError("Cached evaluation provenance differs")
        return report
    output = cache.parent / f"{cache.stem}_{time.time_ns()}"
    report = suite(checkpoint, output, gpu, deadline, split=split)
    write_json(cache, report)
    return report


def render(checkpoint, output, gpu, deadline):
    """A fresh single-seed four-speed run per terrain; no state interpolation."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    policy = output / "exported/policy.pt"
    run_jobs([command_job([ROOT / "scripts/a2_stairs/export_checkpoint.py", "--checkpoint", checkpoint,
                          "--output", policy], output / "export.log", gpu)], deadline)
    for terrain in ("flat", "stairs"):
        evaluation = output / terrain / "evaluation"
        run_jobs([command_job([ROOT / "scripts/a2_stairs/evaluate_mujoco.py", "--policy", policy,
                              "--output", evaluation, "--terrains", terrain, "--speeds", *SPEEDS,
                              "--flat-speeds", *SPEEDS, "--seeds", 42, "--duration", 30],
                             output / f"{terrain}_evaluation.log", gpu)], deadline)
        run_jobs([command_job([ROOT / "scripts/a2_stairs/render_mujoco_evaluation.py",
                              "--evaluation", evaluation, "--output", output / terrain / "videos",
                              "--checkpoint", checkpoint], output / f"{terrain}_render.log", gpu)], deadline)
