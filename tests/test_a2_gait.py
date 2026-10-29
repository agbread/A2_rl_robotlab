"""CPU-only preparation checks; never starts Isaac Sim, CUDA, or training jobs."""
import ast
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts/a2_gait"))
import common
import manage
import processes
from metrics import acceptance, summarize
from initialization import initialize, config_hash


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


logic = load("a2_gait_logic_cpu", ROOT / "source/robot_lab/robot_lab/tasks/a2/gait_logic.py")


def drive(stats, fractions, steps, command=(.7, 0., 0.), flat=True):
    command = torch.tensor([command], dtype=torch.float).repeat(stats.air.shape[0], 1)
    for _ in range(steps):
        result = stats.update(torch.tensor(fractions, dtype=torch.float).expand_as(stats.air), command,
                              torch.full((stats.air.shape[0],), flat))
    return result


def test_airborne_episode_start_is_censored_and_reset_erases_statistics():
    stats = logic.GaitStatistics(1, .02)
    drive(stats, [0, 1, 1, 1], 30)
    drive(stats, [1, 1, 1, 1], 1)
    assert not stats.has_swing.any()
    drive(stats, [0, 1, 1, 1], 20)
    drive(stats, [1, 1, 1, 1], 1)
    assert stats.last_air[0, 0] == pytest.approx(.4)
    stats.reset([0])
    assert not stats.has_swing.any() and not stats.samples.any()


def test_indefinitely_lifted_leg_is_detected_by_duty_without_fabricated_swing():
    stats = logic.GaitStatistics(1, .02)
    swing, duty = drive(stats, [0, 1, 1, 1], 110)
    assert swing.item() == 0 and duty.item() == pytest.approx(.5)
    assert not stats.has_swing.any()


def test_balanced_alternating_gait_is_not_forced_into_simultaneous_contact():
    stats = logic.GaitStatistics(1, .02)
    for _ in range(10):
        drive(stats, [1, 0, 0, 1], 10)
        swing, duty = drive(stats, [0, 1, 1, 0], 10)
    assert swing.item() < 1e-8 and duty.item() < 1e-8


def test_reflection_invariance_and_direction_gating():
    stats = logic.GaitStatistics(2, .02)
    rng = torch.Generator().manual_seed(4)
    for _ in range(150):
        f = (torch.rand(4, generator=rng) > .5).float()
        values = stats.update(torch.stack([f, f[[1, 0, 3, 2]]]), torch.tensor([[.7, 0, 0]]*2),
                              torch.ones(2, dtype=torch.bool))
        for value in values:
            torch.testing.assert_close(value[0], value[1])
    for cmd, flat in (((.7, .1, 0), True), ((.7, 0, .1), True), ((0, 0, 0), True), ((.7, 0, 0), False)):
        swing, duty = drive(logic.GaitStatistics(1, .02), [0, 1, 1, 1], 110, cmd, flat)
        assert swing.item() == duty.item() == 0


def test_command_change_requires_a_fresh_window():
    stats = logic.GaitStatistics(1, .02)
    drive(stats, [0, 1, 1, 1], 110)
    _, duty = drive(stats, [0, 1, 1, 1], 1, (.9, 0, 0))
    assert duty.item() == 0 and stats.samples.item() == 1


def flat_rows(survival=1., ratio=1.2):
    return [dict(terrain="flat", policy=p, speed=s, survival=survival, vx_rmse=.1,
                 front_swing_ratio=ratio, front_duty_gap=.1) for p in ("teacher", "student") for s in common.SPEEDS]


def test_stage_gates_require_both_policies_and_student_stair_history():
    state = common.new_state("scratch", now=0)
    assert not common.advance(state, flat_rows(), [])
    assert common.advance(state, flat_rows(), []) and state["phase"] == "flat_fast"
    assert not common.advance(state, flat_rows(ratio=1.7), [])
    assert not common.advance(state, flat_rows(), [])
    assert common.advance(state, flat_rows(), []) and state["phase"] == "stairs_2_5"
    assert not common.advance(state, flat_rows(), [True]*199)
    assert not common.advance(state, flat_rows(), [True]*179+[False]*21)
    assert common.advance(state, flat_rows(), [True]*180+[False]*20)
    assert state["phase"] == "stairs_4_10" and state["phase_entry"]
    assert not common.flat_gate([r for r in flat_rows() if r["policy"] == "teacher"])


def test_flat_gate_rejects_missing_or_censored_metrics_without_crashing():
    for metric in ("survival", "vx_rmse", "front_swing_ratio", "front_duty_gap"):
        for value in (None, float("nan"), float("inf")):
            rows = flat_rows()
            rows[0][metric] = value
            assert not common.flat_gate(rows)
            if metric in ("survival", "vx_rmse"):
                assert not common.flat_gate(rows, slow=True)


def test_warm_first_full_evaluation_waits_for_2000_updates():
    state = common.new_state("warm")
    state["updates"] = 1999
    assert not common.full_evaluation_due(state, now=10000)
    state["updates"] = 2000
    assert common.full_evaluation_due(state, now=10000)
    state["last_full_eval"] = 10000
    assert not common.full_evaluation_due(state, now=20000)
    assert common.full_evaluation_due(state, now=10000+6*3600)


def test_configuration_profiles_and_all_stair_geometries():
    path = ROOT / "source/robot_lab/robot_lab/tasks/a2/gait_env_cfg.py"
    tree = ast.parse(path.read_text())
    nodes = [n for n in tree.body if isinstance(n, ast.Assign) and any(
        isinstance(t, ast.Name) and t.id in ("STAGES", "MIX") for t in n.targets)]
    nodes += [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "configure_gait"]
    def terrain():
        return SimpleNamespace(sub_terrains={name: SimpleNamespace(proportion=0., curriculum_mode="steps") for name in common.MIX})
    scope = dict(GaitVelocityCommandCfg=lambda: SimpleNamespace(), make_stair_terrain_cfg=terrain,
                 RewTerm=lambda **kw: kw, DoneTerm=lambda **kw: kw, CurrTerm=lambda **kw: kw,
                 swing_balance=None, duty_balance=None, flat_boundary=None, gait_reward_schedule=None)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    assert scope["MIX"] == common.MIX and scope["STAGES"] == common.STAIRS
    for phase in common.PHASES:
        cfg = SimpleNamespace(commands=SimpleNamespace(), scene=SimpleNamespace(terrain=SimpleNamespace()),
                              rewards=SimpleNamespace(lin_vel_z_l2=SimpleNamespace(), base_height_l2=SimpleNamespace()),
                              terminations=SimpleNamespace(), curriculum=SimpleNamespace())
        scope["configure_gait"](cfg, "scratch", phase, 1234)
        counts = cfg.commands.base_velocity.expected_columns
        assert sum(counts.values()) == 20
        assert cfg.gait_update_offset == 1234
        if phase == "mixed":
            assert counts == dict(target_12=4, stair_curriculum=2, near_low=1, near_high=1, near_narrow=1, near_wide=1, flat=10)
        if phase.startswith("stairs"):
            sub = cfg.scene.terrain.terrain_generator.sub_terrains["stair_curriculum"]
            assert (sub.num_steps, sub.rise) == common.STAIRS[phase]
    assert common.recipe("scratch", "flat_slow")["learning_rate"] == .001
    assert common.recipe("scratch", "stairs_2_5")["learning_rate"] == .0001
    assert common.recipe("warm", "mixed")["learning_rate"] == .00001


def test_capacity_selection_respects_both_reserves():
    def row(n, throughput, gpu=10, ram=25):
        return dict(ok=True, num_envs=n, transitions_per_s=throughput, min_gpu_free_gib=gpu, min_ram_free_gib=ram)
    assert common.select_capacity([row(1024, 100), row(2048, 180), row(4096, 200, ram=19)]) == 2048
    with pytest.raises(RuntimeError):
        common.select_capacity([row(1024, 100, gpu=7)])


def test_dry_run_does_not_start_processes_or_create_run_directory(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(subprocess, "Popen", lambda *a, **kw: pytest.fail("Dry-run started a process"))
    folder = tmp_path / "absent"
    assert manage.main(["train", "--run-dir", str(folder), "--dry-run"]) == 0
    preview = json.loads(capsys.readouterr().out)
    assert [j["physical_gpu"] for j in preview["jobs"]] == [0, 1]
    assert preview["jobs"][1]["checkpoint"] is None
    assert not folder.exists()
    assert "isaaclab.app" not in sys.modules


def test_manifest_budget_and_source_identity_survive_resume(tmp_path):
    a = manage.ensure_experiment(tmp_path, [0, 1], ["warm", "scratch"])
    b = manage.ensure_experiment(tmp_path, [0, 1], ["warm", "scratch"])
    assert a["created_at"] == b["created_at"]
    assert a["deadline"] - a["created_at"] == 168*3600
    assert a["training_deadline"] - a["created_at"] == 144*3600
    with pytest.raises(ValueError):
        manage.ensure_experiment(tmp_path, [1, 0], ["warm", "scratch"])


def test_lock_releases_after_exception_and_prevents_duplicate_owners(tmp_path):
    lock = tmp_path / "lock"
    with processes.exclusive_lock(lock):
        with pytest.raises(RuntimeError):
            with processes.exclusive_lock(lock):
                pass
    with processes.exclusive_lock(lock):
        pass


def test_process_group_stopping_only_uses_owned_children(tmp_path):
    job = dict(argv=[sys.executable, "-c", "print('cpu child')"], env=dict(__import__('os').environ),
               cwd=str(tmp_path), log=str(tmp_path / "child.log"))
    processes.run_jobs([job], time.time()+10)
    assert "cpu child" in (tmp_path / "child.log").read_text()
    with pytest.raises(TimeoutError):
        processes.run_jobs([dict(job, log=str(tmp_path / "never.log"))], time.time()-1)
    assert not (tmp_path / "never.log").exists()


def rows_for_acceptance():
    return [dict(simulator=sim, terrain=t, policy="student", speed=v, n=20,
                 swing_s=[.5, .3, .3, .3], front_swing_ratio=1.67, front_duty_gap=.3,
                 rear_duty_gap=.05, fall_rate=0., abs_final_yaw_deg=20., vx_rmse=.1,
                 ascent_success=.9, settled_success=.8, episodes_with_swing=[20]*4)
            for sim in ("isaac", "mujoco") for t in ("flat", "stairs") for v in common.SPEEDS]


def test_acceptance_checks_every_simulator_speed_and_does_not_reward_fr_only_change():
    baseline = rows_for_acceptance()
    candidate = [dict(r, swing_s=[.35, .3, .3, .3], front_swing_ratio=1.17, front_duty_gap=.08)
                 for r in baseline]
    assert acceptance(baseline, candidate)["passed"]
    assert not acceptance(baseline, candidate[:-1])["passed"]
    candidate[0]["swing_s"] = [.5, .5, .3, .3]
    assert not acceptance(baseline, candidate)["passed"]
    candidate[0]["swing_s"][0] = None
    assert not acceptance(baseline, candidate)["passed"]


def test_offline_metrics_censor_incomplete_swing_and_json_has_no_nan():
    time_values = np.arange(300)*.02
    contact = np.tile(((np.arange(300)//10) % 2)[:, None, None], (1, 2, 4)).astype(float)
    command = np.tile([.7, 0, 0], (300, 2, 1))
    result = summarize(time_values, contact, contact*.1, command, command, np.zeros((300, 2)),
                       np.ones((300, 2), bool), [True, True], [False, False], [False, False])
    assert result["swing_s"] == pytest.approx([.2]*4)
    assert result["front_swing_ratio"] == pytest.approx(1.)
    json.dumps(result, allow_nan=False)


def test_packaged_checkpoint_integrity_and_scratch_source_has_no_base_load():
    metadata = common.read_json(common.BASE.parent / "metadata.json")
    for name, info in metadata["files"].items():
        path = common.BASE.parent / name
        assert path.stat().st_size == info["bytes"] and common.digest(path) == info["sha256"]
    saved = torch.load(common.BASE, map_location="cpu", weights_only=False)
    keys = saved["model_state_dict"]
    assert any(k.startswith("teacher_encoder.") for k in keys)
    assert any(k.startswith("student_moe_encoder.") for k in keys)
    assert keys["teacher_encoder.0.network.0.weight"].shape[1] == 275


def test_student_export_matches_cpu_inference_and_reset(tmp_path):
    module = load("a2_gait_export_cpu", ROOT / "scripts/a2_stairs/export_checkpoint.py")
    output = module.export_checkpoint(common.BASE, output=tmp_path / "policy.pt")
    policy = torch.jit.load(str(output), map_location="cpu")
    assert policy(torch.zeros(1, 45)).shape == (1, 12)
    policy.reset()
    assert not policy.obs_history.any()


def test_scratch_never_loads_a_pretrained_policy(monkeypatch):
    runner = SimpleNamespace(load=Mock(), load_weights=Mock())
    monkeypatch.setattr(torch, "load", lambda *a, **kw: pytest.fail("Scratch attempted checkpoint load"))
    state = common.new_state("scratch")
    assert initialize(runner, state, 1024) == "random"
    runner.load.assert_not_called()
    runner.load_weights.assert_not_called()
    state["checkpoint"] = str(common.BASE)
    with pytest.raises(ValueError, match="must not load"):
        initialize(runner, state, 1024)


def test_phase_transition_and_resume_routing_preserve_counters(tmp_path):
    state = common.new_state("scratch")
    state.update(updates=500, transitions=500*1024*24, phase_entry=False)
    metadata = {key: state[key] for key in ("branch", "seed", "updates", "transitions")}
    metadata["config_hash"] = config_hash(state, 1024)
    path = tmp_path / "checkpoint.pt"
    torch.save({"gait_experiment": metadata}, path)
    state["checkpoint"] = str(path)
    runner = SimpleNamespace(load=Mock(), load_weights=Mock(), alg=SimpleNamespace(finetune_step=0))
    assert initialize(runner, state, 1024) == "resume"
    runner.load.assert_called_once_with(str(path))
    with pytest.raises(ValueError, match="configuration"):
        initialize(runner, state, 2048)
    state.update(phase="flat_fast", phase_entry=True)
    assert initialize(runner, state, 1024) == "phase_transition"
    assert runner.alg.finetune_step == 500
    runner.load_weights.assert_called_once_with(str(path))
    state["updates"] = 501
    with pytest.raises(ValueError, match="counters"):
        initialize(runner, state, 1024)


def test_warm_start_uses_packaged_base_only_at_initial_update():
    runner = SimpleNamespace(load=Mock(), load_weights=Mock())
    state = common.new_state("warm")
    assert initialize(runner, state, 1024) == "warm_start"
    runner.load_weights.assert_called_once_with(str(common.BASE))
    state["updates"] = 5
    with pytest.raises(ValueError):
        initialize(runner, state, 1024)


def test_completed_evaluation_cache_does_not_reuse_partial_outputs(tmp_path, monkeypatch):
    import evaluation
    cache = tmp_path / "baseline_metrics.json"
    report = dict(checkpoint_sha256=common.digest(common.BASE), protocol=common.PROTOCOL, split="validation", rows=[])
    calls = []
    def fake_suite(checkpoint, output, *a, **kw):
        calls.append(output)
        return report
    monkeypatch.setattr(evaluation, "suite", fake_suite)
    (tmp_path / "baseline_metrics_partial").mkdir()
    assert evaluation.cached_suite(common.BASE, cache, 0, time.time()+100) == report
    assert evaluation.cached_suite(common.BASE, cache, 0, time.time()+100) == report
    assert len(calls) == 1 and calls[0].name != "baseline_metrics_partial"


def test_static_checks_do_not_import_isaac_or_query_cuda(monkeypatch):
    monkeypatch.setattr(torch.cuda, "init", lambda: pytest.fail("CUDA initialized"))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: pytest.fail("CUDA queried"))
    monkeypatch.setattr(manage, "probe_runtime", lambda python, role: dict(
        ok=True, errors=[], prefix=f"/separate/{role}", role=role, pip_check={"returncode": 0}))
    report = manage.static_checks("/separate/mujoco/bin/python")
    assert report["ok"], report["errors"]
    assert not report["gpu_tested"]
    assert "isaaclab.app" not in sys.modules


def test_failed_parallel_branch_cancels_only_managed_work(tmp_path, monkeypatch):
    monkeypatch.setattr(manage, "hardware", lambda: {"gpus": {0: {"uuid": "zero"}, 1: {"uuid": "one"}}})
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)
    common.write_json(tmp_path / "benchmark.json", dict(source_hash=manage.source_hash(), num_envs=1024,
                      hardware={"gpus": {0: {"uuid": "zero"}, 1: {"uuid": "one"}}}))
    def branch(root, branch, *args):
        if branch == "scratch":
            raise RuntimeError("synthetic failure, no GPU worker")
        assert processes.CANCELLED.wait(timeout=5)
        return {}
    monkeypatch.setattr(manage, "branch_loop", branch)
    try:
        with pytest.raises(RuntimeError, match="synthetic"):
            manage.train_all(tmp_path, dict(gpus=[0, 1], branches=["warm", "scratch"]), False)
        assert processes.CANCELLED.is_set()
    finally:
        processes.CANCELLED.clear()


def test_scheduler_gpu_masks_are_not_silently_overridden(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    with pytest.raises(ValueError, match="scheduler masks"):
        manage.main(["benchmark", "--run-dir", str(tmp_path / "absent")])
    assert not (tmp_path / "absent").exists()


def test_global_reward_schedule_does_not_restart_on_phase_transition():
    path = ROOT / "source/robot_lab/robot_lab/tasks/a2/gait_env_cfg.py"
    tree = ast.parse(path.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "gait_reward_schedule")
    namespace = {}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    term = SimpleNamespace(weight=0)
    env = SimpleNamespace(common_step_counter=0, cfg=SimpleNamespace(gait_update_offset=6000),
                          reward_manager=SimpleNamespace(get_term_cfg=lambda _: term, set_term_cfg=Mock()))
    actual = namespace["gait_reward_schedule"](env, [], "base_height_l2", -1., -10., 5000)
    assert actual == -10 and term.weight == -10


def test_checkpoint_recovery_preserves_a_completed_phase_decision(tmp_path):
    state = common.new_state("scratch")
    state.update(updates=500, transitions=100, phase="flat_fast", phase_entry=True)
    directory = tmp_path / "chunks/chunk1"
    directory.mkdir(parents=True)
    old = dict(branch="scratch", seed=42, updates=500, transitions=100, phase="flat_slow")
    torch.save({"gait_experiment": old}, directory / "model_499.pt")
    manage.recover_state(tmp_path, state)
    assert state["phase"] == "flat_fast" and state["phase_entry"]
    newer = dict(old, updates=600, phase="flat_fast", phase_offset=500)
    torch.save({"gait_experiment": newer}, directory / "model_99.pt")
    manage.recover_state(tmp_path, state)
    assert state["updates"] == 600 and not state["phase_entry"]
