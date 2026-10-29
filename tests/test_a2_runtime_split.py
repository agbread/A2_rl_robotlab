"""CPU-only runtime routing and dependency regressions; never starts simulators."""
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/a2_gait"))
import common
import evaluation
import manage
import runtime_probe
import runtimes
import video_assets


def mock_environment(monkeypatch, role, pip_status=0, forbidden=()):
    def version(name):
        if name in runtimes.EXPECTED[role]:
            return runtimes.EXPECTED[role][name]
        if name in forbidden:
            return "0.0.1"
        raise runtime_probe.metadata.PackageNotFoundError(name)
    monkeypatch.setattr(runtime_probe.metadata, "version", version)
    monkeypatch.setattr(runtime_probe.metadata, "distributions", lambda: [])
    monkeypatch.setattr(runtime_probe.importlib.util, "find_spec", lambda name: SimpleNamespace(
        origin=str(ROOT / f"source/{name}/{name}/__init__.py")))
    monkeypatch.setattr(runtime_probe.shutil, "which", lambda _: "/mujoco/bin/ffmpeg")
    monkeypatch.setattr(runtime_probe, "find_video_font", lambda: "/mujoco/share/fonts/DejaVuSans.ttf")
    calls = []
    def check(argv, **kw):
        calls.append(argv)
        assert argv[1:] == ["-m", "pip", "check"]
        return subprocess.CompletedProcess(argv, pip_status,
            "dependency conflict" if pip_status else "No broken requirements found.", "")
    monkeypatch.setattr(runtime_probe.subprocess, "run", check)
    return calls


@pytest.mark.parametrize("role", ["isaac", "mujoco"])
def test_each_runtime_runs_its_own_pip_check_without_importing_simulators(monkeypatch, role):
    calls = mock_environment(monkeypatch, role)
    before = set(sys.modules)
    report = runtime_probe.inspect_runtime(role)
    assert report["ok"], report["errors"]
    assert calls == [[sys.executable, "-m", "pip", "check"]]
    assert not ({"isaaclab.app", "mujoco", "torch"} & (set(sys.modules) - before))
    assert not report["gpu_tested"]


@pytest.mark.parametrize("role", ["isaac", "mujoco"])
def test_pip_conflicts_fail_even_when_all_pinned_versions_match(monkeypatch, role):
    mock_environment(monkeypatch, role, pip_status=1)
    report = runtime_probe.inspect_runtime(role)
    assert not report["ok"]
    assert any("dependency conflict" in error for error in report["errors"])


@pytest.mark.parametrize("role,forbidden", [("isaac", "mujoco"), ("mujoco", "isaacsim")])
def test_mixed_simulator_environment_is_rejected(monkeypatch, role, forbidden):
    mock_environment(monkeypatch, role, forbidden=[forbidden])
    assert any("keep simulators separate" in e for e in runtime_probe.inspect_runtime(role)["errors"])


def test_static_check_rejects_same_environment_under_two_executable_names(monkeypatch):
    monkeypatch.setattr(manage, "probe_runtime", lambda python, role: dict(
        ok=True, errors=[], prefix="/same/environment", role=role))
    report = manage.static_checks("/same/environment/bin/python3")
    assert not report["ok"] and "different Python environments" in " ".join(report["errors"])


def test_missing_or_nonexistent_mujoco_python_fails_without_a_process(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: pytest.fail("Must not launch a process"))
    assert not runtimes.probe_runtime(None, "mujoco")["ok"]
    assert not runtimes.probe_runtime("/missing/a2_mujoco/bin/python", "mujoco")["ok"]


def test_dependency_probe_failure_is_not_mistaken_for_success(monkeypatch):
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 1, "not json", "failed"))
    assert not runtimes.probe_runtime(sys.executable, "mujoco")["ok"]


@pytest.mark.parametrize("status,errors", [(False, []), (True, ["conflict"])])
def test_dependency_probe_rejects_inconsistent_status(monkeypatch, status, errors):
    output = json.dumps(dict(role="mujoco", ok=status, errors=errors))
    monkeypatch.setattr(subprocess, "run", lambda *a, **kw: subprocess.CompletedProcess(a, 0, output, ""))
    assert not runtimes.probe_runtime(sys.executable, "mujoco")["ok"]


def test_venv_python_symlink_is_not_resolved_out_of_its_environment(tmp_path):
    python = tmp_path / "venv/bin/python"
    python.parent.mkdir(parents=True)
    python.symlink_to(sys.executable)
    assert runtimes.python_path(python) == str(python)


def test_job_environment_cannot_import_isaac_modules_from_the_activated_shell():
    source = dict(CONDA_PREFIX="/isaac", CONDA_DEFAULT_ENV="isaac", VIRTUAL_ENV="/isaac",
                  PATH="/isaac/bin:/usr/bin", PYTHONPATH="/isaac/source", PYTHONHOME="/isaac",
                  LD_LIBRARY_PATH="/isaac/lib:/usr/local/nvidia/lib64", CUDA_VISIBLE_DEVICES="0,1")
    env = runtimes.python_environment("/mj/bin/python", source)
    assert env["PATH"] == "/mj/bin:/usr/bin"
    assert env["LD_LIBRARY_PATH"] == "/usr/local/nvidia/lib64"
    assert env["CONDA_PREFIX"] == "/mj" and env["CUDA_VISIBLE_DEVICES"] == "0,1"
    assert not {"PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "CONDA_DEFAULT_ENV"} & env.keys()


def test_same_interpreter_keeps_its_required_library_paths():
    source = dict(CONDA_PREFIX="/isaac", PATH="/isaac/bin:/usr/bin",
                  LD_LIBRARY_PATH="/isaac/lib:/usr/local/nvidia/lib64")
    env = runtimes.python_environment("/isaac/bin/python", source)
    assert env["LD_LIBRARY_PATH"] == source["LD_LIBRARY_PATH"]
    assert env["PATH"] == source["PATH"]


def test_full_evaluation_routes_export_and_isaac_to_training_python_and_mujoco_to_its_python(tmp_path, monkeypatch):
    jobs = []
    def run(batch, deadline):
        jobs.extend(batch)
        for job in batch:
            if Path(job["argv"][1]).name == "worker.py":
                request = common.read_json(job["argv"][-1])
                common.write_json(request["result"], {"rows": []})
    monkeypatch.setattr(evaluation, "run_jobs", run)
    monkeypatch.setattr(evaluation, "mujoco_rows", lambda _: [])
    evaluation.suite(common.BASE, tmp_path / "suite", 1, 9999999999, mujoco_python="/mj/bin/python")
    assert len(jobs) == 5
    for job in jobs:
        script = Path(job["argv"][1]).name
        expected = "/mj/bin/python" if script == "evaluate_mujoco.py" else sys.executable
        assert job["argv"][0] == expected
        assert job["env"]["MUJOCO_EGL_DEVICE_ID"] == "1"
    report = common.read_json(tmp_path / "suite/metrics.json")
    assert report["mujoco_python"] == "/mj/bin/python"


def test_render_routes_both_evaluation_and_video_jobs_to_mujoco_python(tmp_path, monkeypatch):
    jobs = []
    monkeypatch.setattr(evaluation, "run_jobs", lambda batch, _: jobs.extend(batch))
    evaluation.render(common.BASE, tmp_path / "videos", 0, 9999999999, mujoco_python="/mj/bin/python")
    assert len(jobs) == 5 and jobs[0]["argv"][0] == sys.executable
    assert all(j["argv"][0] == "/mj/bin/python" for j in jobs[1:])


def test_full_evaluation_and_render_never_fallback_to_isaac_environment(tmp_path):
    for fn in (evaluation.suite, evaluation.render):
        with pytest.raises(ValueError, match="separate MuJoCo"):
            fn(common.BASE, tmp_path / "absent", 0, 9999999999)
        assert not (tmp_path / "absent").exists()


def test_resume_keeps_interpreters_dependencies_and_original_deadline(tmp_path):
    paths = dict(isaac_python="/isaac/bin/python", mujoco_python="/mj/bin/python")
    fingerprints = dict(isaac="train-deps", mujoco="eval-deps")
    first = manage.ensure_experiment(tmp_path, [0, 1], ["warm", "scratch"], paths, fingerprints)
    second = manage.ensure_experiment(tmp_path, [0, 1], ["warm", "scratch"], paths, fingerprints)
    assert first["deadline"] == second["deadline"]
    for new_paths, new_deps in ((dict(paths, mujoco_python="/other/bin/python"), fingerprints),
                                (paths, dict(fingerprints, mujoco="changed"))):
        with pytest.raises(ValueError, match="Python paths/dependencies"):
            manage.ensure_experiment(tmp_path, [0, 1], ["warm", "scratch"], new_paths, new_deps)


def test_resume_dry_run_reads_saved_interpreter_without_subprocesses(tmp_path, monkeypatch, capsys):
    common.write_json(tmp_path / "experiment.json", {"runtimes": {"mujoco_python": "/saved/mj/bin/python"}})
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("dry-run launched a child"))
    monkeypatch.setattr(manage, "static_checks", lambda *a: pytest.fail("dry-run probed a Python environment"))
    assert manage.main(["resume", "--run-dir", str(tmp_path), "--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert preview["execution_python"]["mujoco"] == "/saved/mj/bin/python"
    assert preview["execution_python"]["export"] == sys.executable
    assert sorted(p.name for p in tmp_path.iterdir()) == ["experiment.json"]


def test_failed_static_checks_block_gpu_commands_before_lock_or_budget_creation(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    monkeypatch.setattr(manage, "static_checks", lambda *_: dict(ok=False, errors=["pip conflict"]))
    monkeypatch.setattr(manage, "hardware", lambda: pytest.fail("Queried GPU after dependency failure"))
    with pytest.raises(RuntimeError, match="pip conflict"):
        manage.main(["train", "--run-dir", str(tmp_path / "absent"), "--mujoco-python", "/mj/bin/python"])
    assert not (tmp_path / "absent").exists()


def test_font_discovery_supports_conda_without_system_font_install(tmp_path, monkeypatch):
    monkeypatch.delenv("A2_VIDEO_FONT", raising=False)
    monkeypatch.setattr(video_assets.sys, "prefix", str(tmp_path))
    font = tmp_path / "share/fonts/dejavu/DejaVuSans.ttf"
    font.parent.mkdir(parents=True)
    font.write_bytes(b"test path only")
    assert video_assets.find_video_font() == str(font)


def test_invalid_explicit_font_does_not_silently_fall_back(tmp_path, monkeypatch):
    monkeypatch.setenv("A2_VIDEO_FONT", str(tmp_path / "missing.ttf"))
    with pytest.raises(FileNotFoundError, match="A2_VIDEO_FONT"):
        video_assets.find_video_font()


def test_runtime_pins_match_installation_constraints():
    for role, filename in (("isaac", "constraints.txt"), ("mujoco", "constraints-mujoco.txt")):
        pins = (ROOT / "scripts/a2_gait" / filename).read_text().splitlines()
        for name, version in runtimes.EXPECTED[role].items():
            assert f"{name}=={version}" in pins
    isaac = (ROOT / "scripts/a2_gait/requirements-isaac.txt").read_text()
    assert "isaaclab[isaacsim]" in isaac and "isaacsim,all" not in isaac
