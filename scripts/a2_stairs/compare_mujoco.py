"""Compare paired actuator-delay evaluations and audit their saved artifacts.

Usage: python scripts/a2_stairs/compare_mujoco.py BASELINE CANDIDATE [--control CONTROL]
Writes comparison.json and validation.json inside CANDIDATE. Source manifests
must be inspected before treating comparisons as an unchanged-physics experiment.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
FIELDS = ("velocity_rmse", "roll_mean_deg", "roll_rms_deg", "roll_abs_p95_deg", "max_abs_lateral_m",
          "front_duty_gap", "rear_duty_gap", "front_left_load_share", "rear_left_load_share",
          "front_lr_mean_joint_diff_deg", "rear_lr_mean_joint_diff_deg")


def read_results(path):
    return {r["name"]: r for r in json.loads((path / "summary.json").read_text())}


def compare(reference, candidate):
    ref, cand = read_results(reference), read_results(candidate)
    groups = []
    for terrain, speed in sorted({(r["terrain"], r["speed"]) for r in cand.values()}):
        names = sorted(n for n, r in cand.items() if r["terrain"] == terrain and r["speed"] == speed and n in ref)
        stage = "flat" if terrain == "flat" else "ascent"
        group = {"terrain": terrain, "speed": speed, "paired_n": len(names), "metrics": {}}
        for outcome in ("ascent_completed", "settled_success", "success"):
            group[outcome] = {"reference": sum(ref[n][outcome] for n in names),
                              "candidate": sum(cand[n][outcome] for n in names)}
        getters = {field: (lambda m, key=field: m.get(key)) for field in FIELDS}
        for pair in ("front", "rear"):
            for key in ("mean_bias_deg", "waveform_rmse_deg", "shape_rmse_deg"):
                getters[f"phase_{pair}_{key}"] = lambda m, p=pair, k=key: m.get("phase_aligned", {}).get(p, {}).get(k)
        for field, getter in getters.items():
            available = [n for n in names if ref[n]["stages"].get(stage) and cand[n]["stages"].get(stage)
                         and getter(ref[n]["stages"][stage]) is not None and getter(cand[n]["stages"][stage]) is not None]
            if not available:
                continue
            rv = np.asarray([getter(ref[n]["stages"][stage]) for n in available], dtype=float)
            cv = np.asarray([getter(cand[n]["stages"][stage]) for n in available], dtype=float)
            group["metrics"][field] = {"paired_n": len(available), "reference_mean": rv.mean(0).tolist(),
                                       "candidate_mean": cv.mean(0).tolist(), "paired_delta_mean": (cv - rv).mean(0).tolist(),
                                       "paired_delta_by_case": dict(zip(available, (cv - rv).tolist())),
                                       "candidate_numerically_lower_count": (cv < rv).sum(0).tolist()}
        groups.append(group)
    return {"reference": str(reference), "candidate": str(candidate), "groups": groups}


def audit(reference, candidate):
    rm = json.loads((reference / "manifest.json").read_text())
    cm = json.loads((candidate / "manifest.json").read_text())
    results = read_results(candidate)
    report = {"case_count": len(results), "same_cases_as_reference": rm["cases"] == cm["cases"],
              "same_criteria_version": rm["criteria_version"] == cm["criteria_version"],
              "same_source_hashes_as_reference": rm["sources_sha256"] == cm["sources_sha256"],
              "current_source_hashes_match": {p: hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h
                                               for p, h in cm["sources_sha256"].items()},
              "policy_hash_differs_from_reference": rm["policy_sha256"] != cm["policy_sha256"],
              "raw_complete": True, "raw_finite": True}
    for name in results:
        with np.load(candidate / f"{name}.npz") as raw:
            n = len(raw["time"])
            report["raw_complete"] &= n > 0 and all(len(raw[key]) == n for key in raw.files)
            report["raw_finite"] &= all(np.isfinite(raw[key]).all() for key in raw.files)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--control", type=Path)
    args = parser.parse_args()
    report = {"interpretation": "Equal run weighting, paired actuator-delay seeds, not multiple independent training seeds. Numerically lower is descriptive: signed joint biases and load shares are not lower-is-better scores. Flat completion means survival for the prescribed duration, not passing a straight-path accuracy criterion.",
              "baseline": compare(args.baseline, args.candidate)}
    validation = {"baseline": audit(args.baseline, args.candidate)}
    if args.control:
        report["control"] = compare(args.control, args.candidate)
        validation["control"] = audit(args.control, args.candidate)
    (args.candidate / "comparison.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    (args.candidate / "validation.json").write_text(json.dumps(validation, indent=2, allow_nan=False) + "\n")
    print(json.dumps(validation))


if __name__ == "__main__":
    main()
