"""Aggregate an evaluate_mujoco.py output without discarding failed attempts."""
import argparse
from collections import Counter
import csv
import json
from pathlib import Path

import numpy as np


FIELDS = ("duration_s", "mean_velocity_body", "velocity_rmse", "roll_mean_deg", "roll_rms_deg",
          "roll_abs_p95_deg", "max_abs_lateral_m", "duty_factor", "front_duty_gap", "rear_duty_gap",
          "mean_foot_vertical_force_N", "front_left_load_share", "rear_left_load_share",
          "front_lr_mean_joint_diff_deg", "rear_lr_mean_joint_diff_deg", "joint_rom_p90_deg",
          "mean_leg_abs_power_W", "saturation_fraction", "nonfoot_contact_fraction")


def stats(values):
    values = np.asarray(values, dtype=float)
    return {"n": len(values), "mean": values.mean(0).tolist(), "sd": values.std(0).tolist(),
            "min": values.min(0).tolist(), "max": values.max(0).tolist()}


def summarize(output):
    results = json.loads((output / "summary.json").read_text())
    groups = []
    for terrain, speed in sorted({(r["terrain"], r["speed"]) for r in results}):
        selected = [r for r in results if r["terrain"] == terrain and r["speed"] == speed]
        group = {"terrain": terrain, "speed": speed, "n": len(selected),
                 "status_counts": dict(Counter(r["status"] for r in selected)),
                 "success_count": sum(r["success"] for r in selected),
                 "ascent_completed_count": sum(r["ascent_completed"] for r in selected),
                 "settled_success_count": sum(r["settled_success"] for r in selected),
                 "stages": {}}
        completed = [r["duration_s"] for r in selected if r["success"]]
        group["successful_duration_s"] = stats(completed) if completed else None
        for stage in sorted({s for r in selected for s in r["stages"]}):
            metrics = [r["stages"][stage] for r in selected if r["stages"].get(stage)]
            if not metrics:
                continue
            stage_metrics = {"n_runs_with_stage_samples": len(metrics)}
            for field in FIELDS:
                values = [m[field] for m in metrics if m.get(field) is not None]
                if values:
                    stage_metrics[field] = stats(values)
            for pair in ("front", "rear"):
                for field in ("mean_bias_deg", "waveform_rmse_deg", "shape_rmse_deg"):
                    values = [m["phase_aligned"][pair][field] for m in metrics if pair in m["phase_aligned"]]
                    if values:
                        stage_metrics[f"phase_{pair}_{field}"] = stats(values)
            group["stages"][stage] = stage_metrics
        groups.append(group)
    aggregate = {"case_count": len(results), "status_counts": dict(Counter(r["status"] for r in results)),
                 "aggregation": "Equal weight per run, including failures where stage data exists. SD is population spread across delay seeds, not a confidence interval. Partial failed stages are retained and may contain fewer strides.",
                 "groups": groups}
    (output / "aggregate_metrics.json").write_text(json.dumps(aggregate, indent=2, allow_nan=False) + "\n")
    with (output / "case_metrics.csv").open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["name", "terrain", "command_vx", "delay_seed", "status", "duration_s",
                         "ascent_completed_s", "final_x", "final_y", "ascent_or_flat_vx_rmse",
                         "front_duty_gap", "front_left_load_share", "rear_left_load_share"])
        for r in results:
            metric = r["stages"].get("ascent" if r["terrain"] == "stairs" else "flat") or {}
            writer.writerow([r["name"], r["terrain"], r["speed"], r["delay_seed"], r["status"], r["duration_s"],
                             r["stop_command_time_s"], *r["final_position"][:2],
                             metric.get("velocity_rmse", [None])[0], metric.get("front_duty_gap"),
                             metric.get("front_left_load_share"), metric.get("rear_left_load_share")])
    print(json.dumps({"output": str(output), "status_counts": aggregate["status_counts"]}))
    return aggregate


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    summarize(parser.parse_args().output)
