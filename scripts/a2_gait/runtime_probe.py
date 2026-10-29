"""CPU-only metadata and pip check, run by EACH environment's own Python.

No torch/Isaac/MuJoCo imports, CUDA queries, rendering, or model inference.
"""
import argparse
import importlib.metadata as metadata
import importlib.util
import json
from pathlib import Path
import platform
import shutil
import subprocess
import sys

from common import ROOT, fingerprint
from runtimes import EXPECTED
from video_assets import find_video_font


def inspect_runtime(role):
    errors, versions = [], {}
    if sys.version_info[:2] != (3, 11) or platform.system() != "Linux":
        errors.append("Use Python 3.11 on Ubuntu/Linux")
    for name, expected in EXPECTED[role].items():
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            versions[name] = None
        if versions[name] != expected:
            errors.append(f"{name}: expected {expected}, found {versions[name]}")
    forbidden = ["mujoco"] if role == "isaac" else ["isaaclab", "isaacsim", "isaacsim-kernel"]
    for name in forbidden:
        try:
            metadata.version(name)
        except metadata.PackageNotFoundError:
            continue
        errors.append(f"{name} must not be installed in the {role} environment; keep simulators separate")
    modules = ["psutil", "yaml"] if role == "isaac" else ["pygame", "scipy", "PIL", "yaml", "websockets"]
    for name in modules:
        if importlib.util.find_spec(name) is None:
            errors.append(f"Missing dependency: {name}")
    if role == "isaac":
        for name, relative in (("rsl_rl", "source/rsl_rl/rsl_rl"), ("robot_lab", "source/robot_lab/robot_lab")):
            spec = importlib.util.find_spec(name)  # root lookup only; no simulator import
            if spec is None or not spec.origin or Path(spec.origin).resolve().parent != (ROOT / relative).resolve():
                errors.append(f"Install this checkout's {name} in editable mode")
    assets = {}
    if role == "mujoco":
        assets["ffmpeg"] = shutil.which("ffmpeg")
        if not assets["ffmpeg"]:
            errors.append("Install conda-forge::ffmpeg in the MuJoCo environment")
        try:
            assets["font"] = find_video_font()
        except FileNotFoundError as exc:
            errors.append(str(exc))
    try:
        checked = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True, timeout=60)
        pip_check = dict(returncode=checked.returncode, output=(checked.stdout + checked.stderr).strip())
        if checked.returncode:
            errors.append("pip check failed: " + pip_check["output"])
    except (OSError, subprocess.TimeoutExpired) as exc:
        pip_check = dict(returncode=None, output=str(exc))
        errors.append("pip check could not complete: " + str(exc))
    installed = sorted((d.metadata["Name"], d.version) for d in metadata.distributions() if d.metadata["Name"])
    return dict(ok=not errors, role=role, python=sys.executable, prefix=sys.prefix, errors=errors, versions=versions,
                pip_check=pip_check, package_fingerprint=fingerprint(installed), video_assets=assets,
                gpu_tested=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--role", choices=tuple(EXPECTED), required=True)
    report = inspect_runtime(parser.parse_args().role)
    print(json.dumps(report, indent=2))
    sys.exit(0 if report["ok"] else 1)
