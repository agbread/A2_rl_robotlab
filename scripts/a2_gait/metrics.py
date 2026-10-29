"""Common offline metrics for Isaac and MuJoCo; no renderer or simulator imports."""
from __future__ import annotations
import numpy as np


def summarize(time, fraction, height, velocity, command, yaw, valid, survival, ascent, settled):
    """T x N records, equal episode weighting; incomplete swings are censored."""
    means, peaks, air_fractions, errors, yaws = [], [], [], [], []
    dt = float(np.median(np.diff(time))) if len(time) > 1 else .02
    for i in range(fraction.shape[1]):
        mask = (time >= 2.) & valid[:, i] & (command[:, i, 0] > .1)
        f, h = fraction[mask, i], height[mask, i]
        leg_means, leg_peaks = [], []
        for leg in range(4):
            airborne = f[:, leg] < .5
            edges = np.r_[0, np.flatnonzero(airborne[1:] != airborne[:-1]) + 1, len(airborne)]
            spans = [(a, b) for a, b in zip(edges[:-1], edges[1:])
                     if a > 0 and b < len(airborne) and airborne[a]]
            leg_means.append(np.mean([(b-a)*dt for a, b in spans]) if spans else np.nan)
            leg_peaks.append(np.mean([h[a:b, leg].max() for a, b in spans]) if spans else np.nan)
        means.append(leg_means)
        peaks.append(leg_peaks)
        air_fractions.append(1 - f.mean(0) if len(f) else np.full(4, np.nan))
        errors.append(np.sqrt(np.mean((velocity[mask, i, 0] - command[mask, i, 0])**2)) if mask.any() else np.nan)
        visible = np.flatnonzero(valid[:, i])
        yaws.append(abs(yaw[visible[-1], i]) if len(visible) else np.nan)

    def average(values):
        values = np.asarray(values, dtype=float)
        count = np.isfinite(values).sum(0)
        result = np.nansum(values, axis=0) / np.maximum(count, 1)
        return np.where(count > 0, result, np.nan)

    swing = average(means)
    air = average(air_fractions)
    ratio = max(swing[:2]) / min(swing[:2]) if np.isfinite(swing[:2]).all() and min(swing[:2]) > 0 else np.nan
    result = dict(n=len(survival), survival=float(np.mean(survival)), fall_rate=1-float(np.mean(survival)),
                  ascent_success=float(np.mean(ascent)), settled_success=float(np.mean(settled)),
                  swing_s=swing.tolist(), swing_peak_m=average(peaks).tolist(), air_fraction=air.tolist(),
                  front_swing_ratio=float(ratio), front_duty_gap=float(abs(air[0]-air[1])),
                  rear_duty_gap=float(abs(air[2]-air[3])), vx_rmse=float(average(errors)),
                  abs_final_yaw_deg=float(average(yaws)),
                  episodes_with_swing=np.isfinite(means).sum(0).tolist())

    def finite(value):
        if isinstance(value, dict):
            return {k: finite(v) for k, v in value.items()}
        if isinstance(value, list):
            return [finite(v) for v in value]
        return None if isinstance(value, float) and not np.isfinite(value) else value
    return finite(result)


def acceptance(reference, candidate):
    """All speeds and both simulators must pass; absent data never pass."""
    def key(r):
        return r["simulator"], r["terrain"], r["speed"], r["policy"]
    refs = {key(r): r for r in reference if r["policy"] == "student"}
    candidates = {key(r): r for r in candidate if r["policy"] == "student"}
    failures, gaps, errors, yaws = [], [], [], []
    expected = {(s, t, v, "student") for s in ("isaac", "mujoco")
                for t in ("flat", "stairs") for v in (.5, .7, .9, 1.2)}
    for k in sorted(expected):
        r, c = refs.get(k), candidates.get(k)
        if r is None or c is None:
            failures.append(f"{k}: missing case")
            continue
        try:
            if k[1] == "flat":
                checks = dict(fl_shorter=c["swing_s"][0] <= .8*r["swing_s"][0],
                              swing_ratio=c["front_swing_ratio"] <= 1.25,
                              front_duty=c["front_duty_gap"] <= .10,
                              rear_duty=c["rear_duty_gap"] <= r["rear_duty_gap"] + .05,
                              falls=c["fall_rate"] <= r["fall_rate"] + 1e-9,
                              yaw=c["abs_final_yaw_deg"] <= r["abs_final_yaw_deg"] + 1e-9,
                              velocity=c["vx_rmse"] <= r["vx_rmse"] + .05)
                # Do not let censoring hide missing strides in the majority of runs.
                checks["swing_coverage"] = min(c["episodes_with_swing"][:2]) >= .9*c["n"]
                gaps.append(c["front_duty_gap"])
                errors.append(c["vx_rmse"])
                yaws.append(c["abs_final_yaw_deg"])
            else:
                checks = {field: c[field] + .05 + 1e-9 >= r[field]
                          for field in ("ascent_success", "settled_success")}
            failures.extend(f"{k}: {name}" for name, ok in checks.items() if not ok)
        except (TypeError, KeyError):
            failures.append(f"{k}: undefined metric")
    return dict(passed=not failures, failures=failures,
                rank=[float(np.mean(x)) if x and all(v is not None for v in x) else None
                      for x in (gaps, errors, yaws)])
