"""Interpreter selection and CPU-only dependency probes. Never imports simulators."""
from __future__ import annotations
import json
import os
from pathlib import Path
import subprocess
import sys

from common import ROOT

EXPECTED = {
    "isaac": {"isaaclab": "2.3.2.post1", "isaacsim": "5.1.0.0", "isaacsim-kernel": "5.1.0.0",
              "isaacsim-core": "5.1.0.0", "torch": "2.7.0+cu128", "torchvision": "0.22.0+cu128",
              "rsl-rl-lib": "3.3.0", "tensordict": "0.14.3", "numpy": "1.26.0", "h5py": "3.16.0",
              "click": "8.1.7", "typing-extensions": "4.12.2", "websockets": "12.0",
              "wrapt": "1.16.0", "packaging": "23.0"},
    "mujoco": {"mujoco": "3.17.0", "torch": "2.7.0+cpu", "numpy": "1.26.0",
               "imageio": "2.37.0", "imageio-ffmpeg": "0.6.0"},
}


def python_path(value):
    # Do NOT resolve the executable symlink: venv/bin/python needs its own prefix.
    return str(Path(value).expanduser().absolute())


def runtime_paths(mujoco_python=None):
    return dict(isaac_python=python_path(sys.executable),
                mujoco_python=python_path(mujoco_python) if mujoco_python else None)


def python_environment(python, base=None):
    """Do not leak activated Isaac/Conda Python modules or libraries into MuJoCo."""
    env = dict(os.environ if base is None else base)
    prefix = str(Path(python).parent.parent)
    old_prefix = env.get("CONDA_PREFIX") or env.get("VIRTUAL_ENV") or sys.prefix
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV",
                "CONDA_PROMPT_MODIFIER", "CONDA_SHLVL"):
        env.pop(key, None)
    for key in ("PATH", "LD_LIBRARY_PATH"):
        if key == "LD_LIBRARY_PATH" and Path(prefix).resolve() == Path(old_prefix).resolve():
            continue
        parts = [p for p in env.get(key, "").split(os.pathsep) if p and not (
            p == old_prefix or p.startswith(old_prefix + os.sep))]
        if key == "PATH":
            parts.insert(0, str(Path(python).parent))
        if parts:
            env[key] = os.pathsep.join(parts)
        else:
            env.pop(key, None)
    env.update(CONDA_PREFIX=prefix, PYTHONNOUSERSITE="1")
    return env


def probe_runtime(python, role):
    if not python:
        return dict(ok=False, errors=["Specify --mujoco-python /path/to/a2_mujoco/bin/python "
                                      "or A2_MUJOCO_PYTHON"], role=role)
    python = python_path(python)
    if not Path(python).is_file() or not os.access(python, os.X_OK):
        return dict(ok=False, errors=[f"Python executable unavailable: {python}"], role=role)
    try:
        result = subprocess.run([python, "-s", str(ROOT / "scripts/a2_gait/runtime_probe.py"), "--role", role],
                                env=python_environment(python), cwd=ROOT, text=True, capture_output=True, timeout=90)
        report = json.loads(result.stdout)
        if not isinstance(report, dict) or report.get("role") != role or not isinstance(report.get("errors"), list):
            raise ValueError("Invalid runtime probe report")
        if not isinstance(report.get("ok"), bool) or report["ok"] != (not report["errors"]):
            raise ValueError("Inconsistent runtime probe status")
        if result.returncode != 0 and report.get("ok"):
            raise ValueError("Runtime probe returned success with nonzero exit status")
        return report
    except (OSError, subprocess.TimeoutExpired, ValueError) as exc:
        return dict(ok=False, errors=[f"CPU dependency probe failed for {python}: {exc}"], role=role)
