"""Compare baseline, control fine-tune, and symmetry fine-tune evaluation folders.

Run-level means give every evaluated episode equal weight, including failed or
partial ascents when their stage measurements exist. Paired differences use the
same terrain, command speed, and actuator-delay seed. Evaluation seeds from a
single training seed are descriptive repeats, not independent training trials.

Example:
  python scripts/a2_stairs/compare_runs.py --baseline outputs/baseline \
    --control outputs/control --symmetry outputs/symmetry --output outputs/comparison.json
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics


METRICS = {
    "vx_rmse_mps": ("velocity_rmse", 0),
    "front_duty_gap": ("front_duty_gap", None),
    "rear_duty_gap": ("rear_duty_gap", None),
    "roll_signed_mean_deg": ("roll_mean_deg", None),
    "roll_rms_deg": ("roll_rms_deg", None),
    "roll_abs_p95_deg": ("roll_abs_p95_deg", None),
    "front_left_load_share": ("front_left_load_share", None),
    "rear_left_load_share": ("rear_left_load_share", None),
}
for _pair in ("front", "rear"):
    for _field, _label in (("mean_bias_deg", "bias"), ("shape_rmse_deg", "shape_rmse"),
                           ("waveform_rmse_deg", "waveform_rmse")):
        METRICS[f"{_pair}_phase_hip_{_label}_deg"] = (f"phase_{_pair}_{_field}", 0)

OUTCOMES = ("ascent_completed", "settled_success", "fall", "edge_exit", "width_exit",
            "landing_end_exit", "timeout", "nonfinite_state")
PROTOCOL_FIELDS = ("geometry", "physics_dt", "policy_dt", "criteria_version", "completion",
                   "failures", "stage_definition", "contact")


def read_json(path):
    return json.loads(path.read_text())


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    values = [float(value) for value in values if value is not None and math.isfinite(float(value))]
    if not values:
        return {"n": 0, "mean": None, "sd": None, "min": None, "max": None}
    return {"n": len(values), "mean": statistics.fmean(values),
            "sd": statistics.pstdev(values), "min": min(values), "max": max(values)}


def scalar(value, index=None):
    if value is None:
        return None
    value = value[index] if index is not None else value
    value = float(value)
    return value if math.isfinite(value) else None


def stage_metrics(stage):
    if not stage:
        return {name: None for name in METRICS}
    fields = dict(stage)
    for pair in ("front", "rear"):
        for field, value in stage.get("phase_aligned", {}).get(pair, {}).items():
            fields[f"phase_{pair}_{field}"] = value
    return {name: scalar(fields.get(field), index) for name, (field, index) in METRICS.items()}


def outcome(run, name):
    status = run["status"]
    if name in ("ascent_completed", "settled_success"):
        value = run.get(name)
        return int(value) if value is not None else None
    if name == "edge_exit":
        return int(status in ("foot_width_exit", "landing_end_exit"))
    status_name = "foot_width_exit" if name == "width_exit" else name
    return int(status == status_name)


def case_key(run):
    return str(run["terrain"]), float(run["speed"]), int(run["delay_seed"])


def group_key(terrain, speed):
    return f"{terrain}@{speed:g}"


def aggregate_summary(runs):
    groups = {}
    for terrain, speed in sorted({case_key(run)[:2] for run in runs}):
        selected = [run for run in runs if case_key(run)[:2] == (terrain, speed)]
        stages = {}
        for stage in sorted({name for run in selected for name in run.get("stages", {})}):
            extracted = [stage_metrics(run.get("stages", {}).get(stage)) for run in selected]
            stages[stage] = {name: stats([values[name] for values in extracted]) for name in METRICS}
        counts, denominators = {}, {}
        for name in OUTCOMES:
            values = [outcome(run, name) for run in selected]
            known = [value for value in values if value is not None]
            counts[name] = sum(known) if known else None
            denominators[name] = len(known)
        crossings = [event for run in selected for event in run.get("edge_crossings", [])]
        groups[group_key(terrain, speed)] = {
            "terrain": terrain, "speed": speed, "n_runs": len(selected),
            "status_counts": dict(Counter(run["status"] for run in selected)),
            "outcome_counts": counts, "outcome_known_run_counts": denominators,
            "riser_crossing_events": len(crossings),
            "negative_clearance_crossing_events": sum(
                event.get("sole_clearance_above_edge_m", 0.0) < 0 for event in crossings
            ),
            "primary_stage": "ascent" if terrain == "stairs" else "flat", "stages": stages,
        }
    return groups


def aggregate_only(data):
    groups = {}
    for group in data["groups"]:
        status = group["status_counts"]
        stages = {}
        for stage, fields in group.get("stages", {}).items():
            values = {}
            for name, (field, index) in METRICS.items():
                source = fields.get(field)
                values[name] = (
                    {"n": int(source["n"]), **{key: scalar(source[key], index) for key in ("mean", "sd", "min", "max")}}
                    if source else stats([])
                )
            stages[stage] = values
        counts = {
            "ascent_completed": group.get("ascent_completed_count"),
            "settled_success": group.get("settled_success_count"),
            "fall": status.get("fall", 0),
            "edge_exit": status.get("foot_width_exit", 0) + status.get("landing_end_exit", 0),
            "width_exit": status.get("foot_width_exit", 0),
            "landing_end_exit": status.get("landing_end_exit", 0),
            "timeout": status.get("timeout", 0), "nonfinite_state": status.get("nonfinite_state", 0),
        }
        terrain, speed = group["terrain"], float(group["speed"])
        groups[group_key(terrain, speed)] = {
            "terrain": terrain, "speed": speed, "n_runs": group["n"], "status_counts": status,
            "outcome_counts": counts,
            "outcome_known_run_counts": {name: group["n"] if count is not None else 0 for name, count in counts.items()},
            "riser_crossing_events": None, "negative_clearance_crossing_events": None,
            "primary_stage": "ascent" if terrain == "stairs" else "flat", "stages": stages,
        }
    return groups


def load_evaluation(directory):
    directory = directory.resolve()
    summary = directory / "summary.json"
    aggregate = directory / "aggregate_metrics.json"
    manifest = directory / "manifest.json"
    runs = None
    if summary.exists():
        runs = read_json(summary)
        if not isinstance(runs, list) or not runs:
            raise ValueError(f"Expected a nonempty run list in {summary}")
        keys = [case_key(run) for run in runs]
        if len(keys) != len(set(keys)):
            raise ValueError(f"Duplicate terrain/speed/delay_seed cases in {summary}")
        groups = aggregate_summary(runs)
        source = summary
    elif aggregate.exists():
        groups = aggregate_only(read_json(aggregate))
        source = aggregate
    else:
        raise FileNotFoundError(f"No summary.json or aggregate_metrics.json in {directory}")
    consistency = None
    if runs is not None and aggregate.exists():
        saved = read_json(aggregate)
        consistency = {
            "case_count_matches": saved.get("case_count") == len(runs),
            "status_counts_match": saved.get("status_counts") == dict(Counter(run["status"] for run in runs)),
        }
    return {
        "directory": str(directory), "source": str(source), "sha256": sha256(source),
        "aggregation_source": "recomputed_from_run_summary" if runs is not None else "aggregate_only_no_pairing",
        "aggregate_crosscheck": consistency, "groups": groups,
        "manifest": read_json(manifest) if manifest.exists() else None, "_runs": runs,
    }


def differences(left, right):
    # Every numeric difference is LEFT minus RIGHT.
    group_differences = {}
    for key in sorted(set(left["groups"]) | set(right["groups"])):
        a, b = left["groups"].get(key), right["groups"].get(key)
        if a is None or b is None:
            group_differences[key] = {"missing_in": "left" if a is None else "right"}
            continue
        stages = {}
        for stage in sorted(set(a["stages"]) | set(b["stages"])):
            values = {}
            for metric in METRICS:
                av = a["stages"].get(stage, {}).get(metric, {}).get("mean")
                bv = b["stages"].get(stage, {}).get(metric, {}).get("mean")
                values[metric] = av - bv if av is not None and bv is not None else None
            stages[stage] = values
        group_differences[key] = {
            "run_count_difference": a["n_runs"] - b["n_runs"],
            "outcome_count_differences": {
                name: a["outcome_counts"][name] - b["outcome_counts"][name]
                if a["outcome_counts"][name] is not None and b["outcome_counts"][name] is not None else None
                for name in OUTCOMES
            },
            "stage_run_mean_differences": stages,
        }

    paired = {"available": left["_runs"] is not None and right["_runs"] is not None}
    if paired["available"]:
        left_runs = {case_key(run): run for run in left["_runs"]}
        right_runs = {case_key(run): run for run in right["_runs"]}
        keys = sorted(set(left_runs) & set(right_runs))
        paired.update({
            "matched_cases": len(keys),
            "unmatched_left": [list(key) for key in sorted(set(left_runs) - set(right_runs))],
            "unmatched_right": [list(key) for key in sorted(set(right_runs) - set(left_runs))],
            "cases": [], "groups": {},
        })
        pools = {}
        for key in keys:
            a, b = left_runs[key], right_runs[key]
            primary = "ascent" if key[0] == "stairs" else "flat"
            am, bm = stage_metrics(a.get("stages", {}).get(primary)), stage_metrics(b.get("stages", {}).get(primary))
            delta = {name: am[name] - bm[name] if am[name] is not None and bm[name] is not None else None for name in METRICS}
            outcome_delta = {
                name: outcome(a, name) - outcome(b, name)
                if outcome(a, name) is not None and outcome(b, name) is not None else None for name in OUTCOMES
            }
            item = {"terrain": key[0], "speed": key[1], "delay_seed": key[2], "stage": primary,
                    "left_status": a["status"], "right_status": b["status"],
                    "metric_differences": delta, "outcome_differences": outcome_delta}
            paired["cases"].append(item)
            pools.setdefault(group_key(*key[:2]), []).append(item)
        for key, cases in pools.items():
            paired["groups"][key] = {
                "matched_cases": len(cases),
                "metrics": {name: stats([case["metric_differences"][name] for case in cases]) for name in METRICS},
                "outcomes": {name: stats([case["outcome_differences"][name] for case in cases]) for name in OUTCOMES},
            }
    lm, rm = left["manifest"], right["manifest"]
    protocol = {"available": lm is not None and rm is not None}
    if protocol["available"]:
        checked = [field for field in PROTOCOL_FIELDS if field in lm and field in rm]
        protocol.update({"checked_fields": checked, "mismatched_fields": [field for field in checked if lm[field] != rm[field]],
                         "unavailable_fields": [field for field in PROTOCOL_FIELDS if field not in checked]})
    return {"direction": "left_minus_right", "protocol": protocol,
            "group_differences": group_differences, "paired_differences": paired}


def number(value, places=3):
    return "—" if value is None else f"{value:.{places}f}"


def render_table(report):
    lines = [
        "Equal weight per evaluated run; failed and partial stages remain included.",
        "One training seed / short pilot: no statistical significance or generalization claim.",
        "Signed roll is its mean; absolute roll is its 95th percentile, not abs(mean roll).",
        "Duty gaps are fractions. Edge exits count lateral/landing exits, not riser crossings.",
        "",
        "| Condition | Model | n | Ascent | Settled | Falls | Edge exits | vx RMSE m/s | Duty gap F/R | Roll mean / abs p95 deg |",
        "|---|---|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for label, source in report["evaluations"].items():
        for key, group in source["groups"].items():
            metrics = group["stages"].get(group["primary_stage"], {})
            mean = lambda name: metrics.get(name, {}).get("mean")
            counts = group["outcome_counts"]
            lines.append(
                f"| {key} | {label} | {group['n_runs']} | {counts['ascent_completed']} | {counts['settled_success']} | "
                f"{counts['fall']} | {counts['edge_exit']} | {number(mean('vx_rmse_mps'))} | "
                f"{number(mean('front_duty_gap'))} / {number(mean('rear_duty_gap'))} | "
                f"{number(mean('roll_signed_mean_deg'))} / {number(mean('roll_abs_p95_deg'))} |"
            )
    lines += ["", "Phase-aligned hip metrics, primary ascent/flat stage; n is runs with phase estimates, not stride count.", "",
              "| Condition | Model | Front bias / shape RMSE deg (n) | Rear bias / shape RMSE deg (n) |",
              "|---|---|---|---|"]
    for label, source in report["evaluations"].items():
        for key, group in source["groups"].items():
            metrics = group["stages"].get(group["primary_stage"], {})
            cells = []
            for pair in ("front", "rear"):
                bias = metrics.get(f"{pair}_phase_hip_bias_deg", stats([]))
                shape = metrics.get(f"{pair}_phase_hip_shape_rmse_deg", stats([]))
                cells.append(f"{number(bias['mean'])} / {number(shape['mean'])} ({min(bias['n'], shape['n'])})")
            lines.append(f"| {key} | {label} | {cells[0]} | {cells[1]} |")
    lines += ["", "Paired changes are left minus right on matching terrain/speed/delay seeds.", "",
              "| Contrast | Condition | Matched | Δvx RMSE | Δfront duty gap | Δrear duty gap |",
              "|---|---|---:|---:|---:|---:|"]
    for contrast, item in report["comparisons"].items():
        for key, group in item["paired_differences"].get("groups", {}).items():
            metrics = group["metrics"]
            lines.append(f"| {contrast} | {key} | {group['matched_cases']} | "
                         f"{number(metrics['vx_rmse_mps']['mean'])} | {number(metrics['front_duty_gap']['mean'])} | "
                         f"{number(metrics['rear_duty_gap']['mean'])} |")
        mismatches = item["protocol"].get("mismatched_fields", [])
        if mismatches:
            lines.append(f"Protocol mismatch for {contrast}: {', '.join(mismatches)}.")
    return "\n".join(lines) + "\n"


def assert_zero_changes(comparisons):
    checked = 0
    for comparison in comparisons.values():
        for group in comparison["group_differences"].values():
            if "missing_in" in group:
                raise AssertionError("Different condition sets in zero-change check.")
            values = [group["run_count_difference"], *group["outcome_count_differences"].values()]
            values += [value for metrics in group["stage_run_mean_differences"].values() for value in metrics.values()]
            for value in values:
                if value is not None:
                    assert value == 0, value
                    checked += 1
        for case in comparison["paired_differences"].get("cases", []):
            for value in (*case["metric_differences"].values(), *case["outcome_differences"].values()):
                if value is not None:
                    assert value == 0, value
                    checked += 1
    return checked


def compare(args):
    evaluations = {name: load_evaluation(getattr(args, name)) for name in ("baseline", "control", "symmetry")}
    comparisons = {
        "control_minus_baseline": differences(evaluations["control"], evaluations["baseline"]),
        "symmetry_minus_baseline": differences(evaluations["symmetry"], evaluations["baseline"]),
        "symmetry_minus_control": differences(evaluations["symmetry"], evaluations["control"]),
    }
    report = {
        "aggregation": "Equal weight per evaluation run, not per timestep or stride. Missing stage/phase metrics stay missing; n is reported for each metric.",
        "limitations": [
            "Single training seed and short fine-tune pilot: descriptive comparison only, no statistical confidence claim.",
            "Delay seeds are paired evaluation conditions, not independent training seeds.",
            "Failed runs are retained; stage statistics condition on reaching that stage and may contain partial ascents.",
            "Phase estimates use complete observed cycles and can have fewer available runs than other metrics.",
            "Negative clearance crossings are approximate diagnostics, not proven toe strikes.",
        ],
        "evaluations": {name: {key: value for key, value in source.items() if key != "_runs"}
                        for name, source in evaluations.items()},
        "comparisons": comparisons,
    }
    if args.assert_zero_differences:
        report["zero_change_check"] = {"passed": True, "checked_numeric_differences": assert_zero_changes(comparisons)}
    table = render_table(report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    table_path = args.output.with_suffix(".txt")
    table_path.write_text(table)
    print(table, end="")
    print(f"JSON: {args.output}\nTable: {table_path}")
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--symmetry", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--assert-zero-differences", action="store_true", help="Identity dry-run: fail if any comparable difference is nonzero.")
    compare(parser.parse_args())
