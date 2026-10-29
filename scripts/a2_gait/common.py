"""Portable experiment protocol; deliberately imports neither CUDA nor Isaac."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import time

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "deploy/pre_train/a2/gait_base_model1999/model_1999.pt"
SPEEDS = [.5, .7, .9, 1.2]
PHASES = ["flat_slow", "flat_fast", "stairs_2_5", "stairs_4_10", "stairs_6_15",
          "stairs_8_15", "stairs_10_15", "stairs_12_15", "mixed"]
STAIRS = dict(zip(PHASES[2:-1], [(2, .05), (4, .10), (6, .15), (8, .15), (10, .15), (12, .15)]))
MIX = dict(target_12=.20, stair_curriculum=.10, near_low=.05, near_high=.05,
           near_narrow=.05, near_wide=.05, flat=.50)
PROTOCOL = {
    "version": 1, "speeds": SPEEDS, "seed": 42, "training_hours": 144,
    "total_hours": 168, "num_env_candidates": [1024, 2048, 4096],
    "gpu_free_gib": 8, "ram_free_gib": 20, "teacher_ratio": .75,
    "steps_per_update": 24, "eval_every_updates": 500, "full_eval_hours": 6,
    "warm_max_updates": 10000, "scratch_max_updates": 300000,
    "warm_first_evaluation_updates": 2000,
    "mirror_weight": .05, "swing_weight": -.2, "duty_weight": -1.,
    "contact_threshold_n": 10., "window_s": 2., "mixture": MIX,
    "validation_seeds": list(range(20)), "final_seeds": list(range(100, 120)),
}


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def read_json(path):
    return json.loads(Path(path).read_text())


def recipe(branch, phase):
    if branch not in ("warm", "scratch") or phase not in PHASES:
        raise ValueError("Invalid branch/phase")
    if branch == "warm" and phase != "mixed":
        raise ValueError("Warm branch starts and stays in mixed terrain")
    flat = phase.startswith("flat")
    return dict(branch=branch, phase=phase,
                task="RobotLab-A2-Stairs-Gait-v0" if branch == "warm" else "RobotLab-A2-Gait-Scratch-v0",
                learning_rate=1e-3 if flat else (1e-5 if phase == "mixed" else 1e-4),
                student_learning_rate=1e-3 if flat else 1e-4,
                schedule="adaptive" if flat else "fixed", ramp_updates=200 if branch == "warm" else 1000)


def new_state(branch, now=None):
    now = time.time() if now is None else now
    return dict(version=1, branch=branch, phase="mixed" if branch == "warm" else "flat_slow",
                seed=42, updates=0, transitions=0, checkpoint=None, phase_entry=True,
                flat_streak=0, acceptance_streak=0, last_full_eval=0., candidates=[],
                created_at=now, status="ready")


def flat_gate(rows, slow=False):
    required = [.5] if slow else SPEEDS
    for policy in ("teacher", "student"):
        for speed in required:
            row = next((r for r in rows if r["terrain"] == "flat" and r["policy"] == policy
                        and abs(r["speed"] - speed) < 1e-6), None)
            metrics = ("survival", "vx_rmse") if slow else (
                "survival", "vx_rmse", "front_swing_ratio", "front_duty_gap")
            if not row or any(not isinstance(row.get(k), (float, int)) or not math.isfinite(row[k])
                              for k in metrics):
                return False
            if row["survival"] < .95 or row["vx_rmse"] > .15:
                return False
            if not slow and (row["front_swing_ratio"] > 1.5
                             or row["front_duty_gap"] > .15):
                return False
    return True


def full_evaluation_due(state, now):
    if state["branch"] == "warm" and state["updates"] < PROTOCOL["warm_first_evaluation_updates"]:
        return False
    return not state["last_full_eval"] or now - state["last_full_eval"] >= PROTOCOL["full_eval_hours"]*3600


def advance(state, rows, recent_stairs):
    """State machine uses *student* completed episodes, never teacher-only wins."""
    phase = state["phase"]
    if state["branch"] != "scratch" or phase == "mixed":
        return False
    passed = flat_gate(rows, slow=phase == "flat_slow")
    state["flat_streak"] = state["flat_streak"] + 1 if passed else 0
    promote = state["flat_streak"] >= 2 if phase.startswith("flat") else (
        passed and len(recent_stairs) >= 200 and sum(recent_stairs[-200:]) / 200 >= .90)
    if promote:
        state.update(phase=PHASES[PHASES.index(phase) + 1], phase_entry=True, flat_streak=0)
    return promote


def select_capacity(results):
    valid = [r for r in results if r.get("ok") and r["min_gpu_free_gib"] >= 8
             and r["min_ram_free_gib"] >= 20 and r["transitions_per_s"] > 0]
    if not valid:
        raise RuntimeError("No safe simultaneous two-GPU capacity; do not start training")
    return max(valid, key=lambda r: (r["transitions_per_s"], -r["num_envs"]))["num_envs"]
