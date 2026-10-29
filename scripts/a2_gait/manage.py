#!/usr/bin/env python3
"""Portable A2 experiment manager. No GPU activity unless explicitly requested.

Examples:
  python scripts/a2_gait/manage.py preflight --static
  python scripts/a2_gait/manage.py train --branch both --run-dir outputs/gait --dry-run
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
from pathlib import Path
import subprocess
import sys
import time

from common import (ROOT, BASE, PROTOCOL, digest, fingerprint, read_json, write_json,
                    new_state, recipe, advance, select_capacity, full_evaluation_due)
from evaluation import worker_job, suite, cached_suite, render
from processes import exclusive_lock, run_jobs, CANCELLED
from runtimes import probe_runtime, runtime_paths


def source_hash():
    files = sorted((ROOT / "scripts/a2_gait").glob("*.py"))
    files.extend(sorted((ROOT / "scripts/a2_gait").glob("*.txt")))
    for directory in (ROOT / "source/robot_lab/robot_lab/tasks/a2", ROOT / "source/rsl_rl/rsl_rl"):
        files.extend(sorted(directory.rglob("*.py")))
    files.extend(ROOT / "scripts/a2_stairs" / name for name in (
        "evaluate_mujoco.py", "export_checkpoint.py", "render_mujoco_evaluation.py"))
    return fingerprint({str(p.relative_to(ROOT)): digest(p) for p in files})


def static_checks(mujoco_python=None):
    paths = runtime_paths(mujoco_python)
    reports = {role: probe_runtime(paths[f"{role}_python"], role) for role in ("isaac", "mujoco")}
    errors = [f"{role}: {message}" for role, report in reports.items() for message in report["errors"]]
    if all(report.get("prefix") for report in reports.values()):
        if Path(reports["isaac"]["prefix"]).resolve() == Path(reports["mujoco"]["prefix"]).resolve():
            errors.append("Isaac and MuJoCo must use different Python environments")
    metadata_file = BASE.parent / "metadata.json"
    if not metadata_file.exists():
        errors.append("Bundled checkpoint manifest is missing")
    else:
        for name, info in read_json(metadata_file)["files"].items():
            path = BASE.parent / name
            if not path.is_file() or path.stat().st_size != info["bytes"] or digest(path) != info["sha256"]:
                errors.append(f"Checkpoint package integrity failure: {name}")
    for path in ("resources/a2_description/urdf/a2.urdf", "resources/a2_description/a2.xml",
                 "resources/a2_description/flat.xml", "resources/a2_description/stairs_12.xml"):
        if not (ROOT / path).is_file():
            errors.append(f"Missing asset: {path}")
    return dict(ok=not errors, errors=errors, runtimes=paths, environments=reports, source_hash=source_hash(),
                protocol_hash=fingerprint(PROTOCOL), gpu_tested=False,
                note="CPU metadata/hash checks and pip check in both interpreters; no CUDA or simulator initialization")


def hardware():
    import psutil
    output = subprocess.check_output(["nvidia-smi", "--query-gpu=index,uuid,name,memory.free,memory.total,driver_version",
                                      "--format=csv,noheader,nounits"], text=True)
    devices = {}
    for line in output.strip().splitlines():
        index, uuid, name, free, total, driver = [v.strip() for v in line.split(",")]
        devices[int(index)] = dict(uuid=uuid, name=name, free_gib=float(free)/1024,
                                   total_gib=float(total)/1024, driver=driver)
    return dict(gpus=devices, ram_free_gib=psutil.virtual_memory().available/2**30)


def ensure_experiment(folder, gpus, branches, runtimes=None, runtime_fingerprints=None):
    path = folder / "experiment.json"
    if path.exists():
        data = read_json(path)
        if data["source_hash"] != source_hash() or data["protocol_hash"] != fingerprint(PROTOCOL):
            raise ValueError("Experiment source/protocol changed; do not silently resume a different recipe")
        if data.get("baseline_sha256") != digest(BASE):
            raise ValueError("Baseline checkpoint changed")
        if data["gpus"] != gpus or data["branches"] != branches:
            raise ValueError("Experiment GPU assignment/branches differ")
        if data.get("runtimes") != runtimes or data.get("runtime_fingerprints") != runtime_fingerprints:
            raise ValueError("Experiment Python paths/dependencies changed; do not silently resume another environment")
        return data
    now = time.time()
    data = dict(version=1, created_at=now, training_deadline=now+144*3600, deadline=now+168*3600,
                gpus=gpus, branches=branches, source_hash=source_hash(), protocol_hash=fingerprint(PROTOCOL),
                baseline_sha256=digest(BASE), protocol=PROTOCOL, status="prepared", runtimes=runtimes,
                runtime_fingerprints=runtime_fingerprints)
    write_json(path, data)
    return data


def gpu_preflight(folder, experiment, report=None):
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Use physical GPU IDs: unset CUDA_VISIBLE_DEVICES before this manager (do not override scheduler allocations)")
    report = report or static_checks(experiment["runtimes"]["mujoco_python"])
    if not report["ok"]:
        raise RuntimeError(json.dumps(report, indent=2))
    machine = hardware()
    for gpu in experiment["gpus"]:
        if gpu not in machine["gpus"] or machine["gpus"][gpu]["free_gib"] < 8:
            raise RuntimeError(f"GPU {gpu} unavailable or reserve below 8 GiB; no other processes are stopped")
    if machine["ram_free_gib"] < 20:
        raise RuntimeError("At least 20 GiB system RAM reserve is required")
    stamp = str(time.time_ns())
    jobs = []
    for gpu in experiment["gpus"]:
        request = dict(mode="probe", gpu=gpu, state=new_state("scratch"), num_envs=16, updates=0,
                       deadline=min(experiment["training_deadline"], time.time()+900))
        jobs.append(worker_job(request, folder / "preflight" / stamp / f"gpu_{gpu}"))
    run_jobs(jobs, min(experiment["training_deadline"], time.time()+900))
    report.update(gpu_tested=True, hardware=machine, checked_at=time.time())
    write_json(folder / "preflight.json", report)


def benchmark(folder, experiment):
    check = read_json(folder / "preflight.json")
    if not check.get("gpu_tested") or check["source_hash"] != source_hash():
        raise ValueError("Run preflight --gpu on this checkout first")
    results = []
    for count in PROTOCOL["num_env_candidates"]:
        out = folder / "benchmark" / f"{count}_{time.time_ns()}"
        jobs = []
        for branch, gpu in zip(("warm", "scratch"), experiment["gpus"]):
            state = new_state(branch)
            state["phase"] = "mixed"  # benchmark the shared, largest terrain set
            jobs.append(worker_job(dict(mode="benchmark", state=state, gpu=gpu, num_envs=count, updates=70,
                                        deadline=experiment["training_deadline"]), out / branch))
        peak = dict(min_gpu_free_gib=float("inf"), min_ram_free_gib=float("inf"))
        def monitor():
            h = hardware()
            peak["min_gpu_free_gib"] = min(peak["min_gpu_free_gib"], *(h["gpus"][g]["free_gib"] for g in experiment["gpus"]))
            peak["min_ram_free_gib"] = min(peak["min_ram_free_gib"], h["ram_free_gib"])
            return peak["min_gpu_free_gib"] >= 8 and peak["min_ram_free_gib"] >= 20
        try:
            run_jobs(jobs, experiment["training_deadline"], monitor)
            measured = [read_json(out / b / "result.json")["timing_seconds"][20:] for b in ("warm", "scratch")]
            if any(len(t) < 50 for t in measured):
                raise RuntimeError("Benchmark did not complete 20 warmup + 50 measured updates")
            score = sum(count*24*len(t)/sum(t) for t in measured)
            results.append(dict(ok=True, num_envs=count, transitions_per_s=score, **peak))
        except (RuntimeError, TimeoutError) as exc:
            results.append(dict(ok=False, num_envs=count, error=str(exc)))
            break  # A larger population is not a safe fallback after resource failure.
    write_json(folder / f"benchmark_attempt_{time.time_ns()}.json", results)
    chosen = select_capacity(results)
    report = dict(num_envs=chosen, results=results, source_hash=source_hash(), measured_at=time.time(),
                  hardware=hardware())
    write_json(folder / "benchmark.json", report)
    return report


def recover_state(folder, state):
    """Recover only complete, atomically saved checkpoints from owned chunks."""
    import torch
    best = None
    for path in (folder / "chunks").glob("*/model_*.pt"):
        saved = torch.load(path, map_location="cpu", weights_only=False)
        metadata = saved.get("gait_experiment", {})
        if metadata.get("seed") == state["seed"] and metadata.get("branch") == state["branch"]:
            # Never undo a completed phase decision after the checkpoint was evaluated.
            if metadata.get("updates", -1) > state["updates"]:
                if best is None or metadata["updates"] > best[0]["updates"]:
                    best = (metadata, path)
        del saved
    if best:
        state.update(best[0], checkpoint=str(best[1]), phase_entry=False)
    return state


def branch_loop(root, branch, gpu, experiment, num_envs):
    from metrics import acceptance
    folder = root / branch
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "state.json"
    state = read_json(path) if path.exists() else new_state(branch)
    recover_state(folder, state)
    if state["acceptance_streak"] >= 2 and state["phase"] == "mixed":
        return state
    state["status"] = "running"
    write_json(path, state)
    limit = PROTOCOL[f"{branch}_max_updates"]
    try:
        while state["updates"] < limit and time.time() < experiment["training_deadline"]:
            if time.time() >= experiment["training_deadline"] - 180:
                break
            out = folder / "chunks" / f"{state['seed']}_{state['updates']:07d}_{state['phase']}_{time.time_ns()}"
            request = dict(mode="train", state=state, gpu=gpu, num_envs=num_envs,
                           updates=min(500, limit-state["updates"]), deadline=experiment["training_deadline"]-60)
            try:
                run_jobs([worker_job(request, out)], experiment["training_deadline"])
            except TimeoutError:
                recover_state(folder, state)
                break
            result = read_json(out / "result.json")
            state.update({k: result[k] for k in ("updates", "transitions", "phase_offset", "phase_entry", "checkpoint")})
            write_json(path, state)
            if time.time() >= experiment["training_deadline"]-300:
                break
            if branch == "scratch" and state["phase"] != "mixed":
                quick = suite(state["checkpoint"], folder / "evaluations" / f"quick_{time.time_ns()}",
                              gpu, experiment["training_deadline"], quick=True)
                advance(state, quick["rows"], result["recent_student_stairs"])
                write_json(path, state)
            if full_evaluation_due(state, time.time()):
                reference = cached_suite(BASE, folder / "baseline_validation_metrics.json", gpu,
                                         experiment["training_deadline"],
                                         mujoco_python=experiment["runtimes"]["mujoco_python"])
                candidate_dir = folder / "evaluations" / f"full_{time.time_ns()}"
                candidate = suite(state["checkpoint"], candidate_dir, gpu, experiment["training_deadline"],
                                  mujoco_python=experiment["runtimes"]["mujoco_python"])
                decision = acceptance(reference["rows"], candidate["rows"])
                state["candidates"].append(dict(checkpoint=state["checkpoint"], evaluation=str(candidate_dir),
                                                 updates=state["updates"], seed=state["seed"], **decision))
                state["last_full_eval"] = time.time()
                state["acceptance_streak"] = state["acceptance_streak"]+1 if decision["passed"] else 0
            write_json(path, state)
            if state["acceptance_streak"] >= 2 and state["phase"] == "mixed":
                break
        state["status"] = "budget_or_update_limit" if state["acceptance_streak"] < 2 else "accepted_validation"
    except TimeoutError:
        recover_state(folder, state)
        state["status"] = "training_budget_expired"
    except BaseException:
        recover_state(folder, state)
        state["status"] = "interrupted_or_failed"
        write_json(path, state)
        raise
    write_json(path, state)
    return state


def final_evaluation(root, experiment):
    from metrics import acceptance
    decisions = []
    for branch in experiment["branches"]:
        state = read_json(root / branch / "state.json")
        candidates = list(state["candidates"])
        if state.get("checkpoint") and not any(c["checkpoint"] == state["checkpoint"] for c in candidates):
            candidates.append(dict(checkpoint=state["checkpoint"], passed=False, rank=[None]*3))
        candidates.sort(key=lambda c: (not c["passed"], [float("inf") if x is None else x for x in c["rank"]]))
        gpu = experiment["gpus"][0 if branch == "warm" else 1]
        reference_cache = root / "final" / branch / "baseline_metrics.json"
        if not candidates or time.time() >= experiment["deadline"]:
            continue
        try:
            reference = cached_suite(BASE, reference_cache, gpu, experiment["deadline"], split="final",
                                     mujoco_python=experiment["runtimes"]["mujoco_python"])
        except TimeoutError:
            break
        for candidate in candidates[:3]:
            try:
                directory = root / "final" / branch / f"candidate_{time.time_ns()}"
                result = suite(candidate["checkpoint"], directory, gpu, experiment["deadline"], split="final",
                               mujoco_python=experiment["runtimes"]["mujoco_python"])
                decisions.append(dict(checkpoint=candidate["checkpoint"], branch=branch,
                                      evaluation=str(directory), **acceptance(reference["rows"], result["rows"])))
                write_json(root / "final/comparison.json", decisions)
            except TimeoutError:
                break
    passed = sorted((d for d in decisions if d["passed"]), key=lambda d: d["rank"])
    selection = dict(selected=passed[0] if passed else None, retained_baseline=str(BASE),
                     note="No passing candidate means retain baseline, not proof scratch training is impossible.")
    write_json(root / "selection.json", selection)
    if passed and time.time() < experiment["deadline"]:
        try:
            render(passed[0]["checkpoint"], root / f"selected_videos_{time.time_ns()}", experiment["gpus"][0], experiment["deadline"],
                   mujoco_python=experiment["runtimes"]["mujoco_python"])
            selection["render_status"] = "completed"
        except TimeoutError:
            selection["render_status"] = "budget_expired_incomplete_videos"
        write_json(root / "selection.json", selection)


def train_all(folder, experiment, resume):
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Physical GPU assignment requires an unmasked local Ubuntu GPU session")
    report = read_json(folder / "benchmark.json")
    if report["source_hash"] != source_hash():
        raise ValueError("Benchmark source differs")
    # UUIDs detect a different host/device mapping, not just matching numeric IDs.
    machine = hardware()
    for gpu in experiment["gpus"]:
        if machine["gpus"][gpu]["uuid"] != report["hardware"]["gpus"][str(gpu)]["uuid"]:
            raise ValueError("GPU mapping differs: repeat preflight/benchmark in a new experiment directory")
    if not resume and any((folder / b / "state.json").exists() for b in experiment["branches"]):
        raise ValueError("Training state exists; use resume")
    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(branch_loop, folder, b, experiment["gpus"][0 if b == "warm" else 1],
                               experiment, report["num_envs"]) for b in experiment["branches"]]
        try:
            for future in as_completed(futures):
                future.result()
        except BaseException:
            CANCELLED.set()
            raise
    final_evaluation(folder, experiment)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=("preflight", "benchmark", "train", "resume", "evaluate", "render"))
    p.add_argument("--run-dir", type=Path, default=ROOT / "outputs/a2_gait_remote")
    p.add_argument("--branch", choices=("warm", "scratch", "both"), default="both")
    p.add_argument("--gpus", nargs=2, type=int, default=[0, 1])
    p.add_argument("--mujoco-python", type=Path,
                   help="Separate MuJoCo environment's bin/python (or A2_MUJOCO_PYTHON); saved for resume")
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--static", action="store_true")
    mode.add_argument("--gpu", action="store_true")
    p.add_argument("--checkpoint", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--split", choices=("validation", "final"), default="final")
    p.add_argument("--dry-run", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if len(set(args.gpus)) != 2 or min(args.gpus) < 0:
        raise ValueError("Specify two distinct nonnegative physical GPU indices")
    branches = ["warm", "scratch"] if args.branch == "both" else [args.branch]
    folder = args.run_dir.resolve()
    saved = read_json(folder / "experiment.json") if (folder / "experiment.json").exists() else {}
    mujoco_python = (args.mujoco_python or os.environ.get("A2_MUJOCO_PYTHON")
                    or (saved.get("runtimes") or {}).get("mujoco_python"))
    paths = runtime_paths(mujoco_python)
    if args.command in ("evaluate", "render") and (args.checkpoint is None or args.output is None):
        raise ValueError("evaluate/render require --checkpoint and a fresh --output directory")
    if args.dry_run:
        jobs = [dict(branch=b, physical_gpu=args.gpus[0 if b == "warm" else 1],
                                         recipe=recipe(b, "mixed" if b == "warm" else "flat_slow"),
                                         checkpoint=str(BASE) if b == "warm" else None,
                                         argv=[sys.executable, str(ROOT / "scripts/a2_gait/worker.py"),
                                               "--request", str(folder / b / "<chunk>/request.json")]) for b in branches]
        if args.command == "preflight":
            jobs = [dict(mode="probe", physical_gpu=g, num_envs=16, optimizer_updates=0)
                    for g in args.gpus] if args.gpu else []
        elif args.command == "benchmark":
            jobs = [dict(mode="benchmark", physical_gpu=g, num_envs=n, warmup_updates=20, measured_updates=50)
                    for n in PROTOCOL["num_env_candidates"] for g in args.gpus]
        elif args.command in ("evaluate", "render"):
            jobs = [dict(mode=args.command, physical_gpu=args.gpus[0], checkpoint=str(args.checkpoint.resolve()),
                         output=str(args.output.resolve()), terrains=["flat", "stairs"], speeds=PROTOCOL["speeds"],
                         seeds=PROTOCOL[f"{args.split}_seeds"] if args.command == "evaluate" else [42])]
        print(json.dumps(dict(command=args.command, run_dir=str(folder), gpu_activity=False, jobs=jobs,
                              runtimes=paths,
                              execution_python=dict(isaac=paths["isaac_python"], export=paths["isaac_python"],
                                                    mujoco=paths["mujoco_python"], render=paths["mujoco_python"]),
                              note="Preview only; creates no files or child processes"), indent=2))
        return 0
    if args.command == "preflight" and not args.gpu:
        report = static_checks(paths["mujoco_python"])
        print(json.dumps(report, indent=2))
        return 0 if report["ok"] else 1
    if os.environ.get("CUDA_VISIBLE_DEVICES"):
        raise ValueError("Use physical GPU IDs in an unmasked, allocated local session; do not bypass scheduler masks")
    report = static_checks(paths["mujoco_python"])
    if not report["ok"]:
        raise RuntimeError(json.dumps(report, indent=2))
    runtime_fingerprints = {role: info["package_fingerprint"] for role, info in report["environments"].items()}
    with exclusive_lock(folder / ".manager.lock"):
        experiment = ensure_experiment(folder, args.gpus, branches, paths, runtime_fingerprints)
        if args.command == "preflight":
            gpu_preflight(folder, experiment, report)
        elif args.command == "benchmark":
            benchmark(folder, experiment)
        elif args.command in ("train", "resume"):
            train_all(folder, experiment, args.command == "resume")
        else:
            if args.command == "evaluate":
                suite(args.checkpoint, args.output, args.gpus[0], experiment["deadline"], split=args.split,
                      mujoco_python=paths["mujoco_python"])
            else:
                render(args.checkpoint, args.output, args.gpus[0], experiment["deadline"],
                       mujoco_python=paths["mujoco_python"])
    return 0


if __name__ == "__main__":
    import signal
    def interrupted(*_):
        raise KeyboardInterrupt("Stop requested; checkpointing owned workers")
    signal.signal(signal.SIGTERM, interrupted)
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print("Stopped owned jobs; use resume with the same run directory.", file=sys.stderr)
        sys.exit(130)
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
