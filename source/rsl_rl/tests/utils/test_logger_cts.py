"""Conditional stair metrics must survive rollout logging without double counting."""

from collections import deque
from unittest.mock import Mock

import pytest
import torch

from rsl_rl.utils.logger_cts import LoggerCTS


PREFIX = "Metrics/base_velocity/stairs/target/student/"


def logger_with(extras):
    logger = LoggerCTS.__new__(LoggerCTS)
    logger.device = "cpu"
    logger.ep_extras = extras
    return logger


def test_union_keeps_groups_and_stages_first_seen_later():
    logger = logger_with([
        {"old_metric": 2.0},
        {"old_metric": 4.0, PREFIX + "success": 0.75},
        {PREFIX + "late_roll_deg": -2.0},
    ])
    result = logger._aggregate_episode_extras()
    assert result["old_metric"].item() == 3.0
    assert result[PREFIX + "success"].item() == 0.75
    assert result[PREFIX + "late_roll_deg"].item() == -2.0


def test_repeated_stale_snapshots_do_not_average_or_sum_cumulative_totals():
    first = {PREFIX + "episodes_total": 2, PREFIX + "success_total": 2,
             PREFIX + "ascent_total": 2, PREFIX + "failure_total": 0,
             PREFIX + "assigned_seconds_total": 12.5}
    latest = {PREFIX + "episodes_total": 10, PREFIX + "success_total": 3,
              PREFIX + "ascent_total": 5, PREFIX + "failure_total": 7,
              PREFIX + "assigned_seconds_total": 80.25}
    logger = logger_with([first, first, first, latest, latest, {"unrelated": 1.0}])
    result = logger._aggregate_episode_extras()
    assert result[PREFIX + "episodes_total"].item() == 10
    assert result[PREFIX + "assigned_seconds_total"].item() == 80.25
    assert result[PREFIX + "success_rate_cumulative"].item() == pytest.approx(0.3)
    assert result[PREFIX + "ascent_rate_cumulative"].item() == pytest.approx(0.5)
    assert result[PREFIX + "failure_rate_cumulative"].item() == pytest.approx(0.7)
    assert first[PREFIX + "episodes_total"] == 2
    assert latest[PREFIX + "episodes_total"] == 10


def test_other_metrics_keep_legacy_mean_and_input_objects_unchanged():
    original_tensor = torch.tensor([3.0, 5.0])
    infos = [{"ordinary_total": 1.0, "ordinary": 1.0},
             {"ordinary_total": 3.0, "ordinary": original_tensor}]
    result = logger_with(infos)._aggregate_episode_extras()
    assert result["ordinary_total"].item() == 2.0
    assert result["ordinary"].item() == 3.0
    assert isinstance(infos[0]["ordinary"], float)
    assert infos[1]["ordinary"] is original_tensor


def test_unprefixed_stair_totals_and_zero_denominator():
    result = logger_with([
        {"stairs/target/all/episodes_total": 0, "stairs/target/all/success_total": 0},
        {"stairs/near/all/contact_seconds_total": 2**25 + 1},
    ])._aggregate_episode_extras()
    assert "stairs/target/all/success_rate_cumulative" not in result
    assert result["stairs/near/all/contact_seconds_total"].item() == 2**25 + 1


def test_writer_receives_late_metric_and_latest_cumulative_rate(capsys):
    logger = logger_with([
        {"ordinary": 2.0},
        {PREFIX + "episodes_total": 4, PREFIX + "success_total": 3},
        {PREFIX + "episodes_total": 8, PREFIX + "success_total": 5},
    ])
    logger.log_dir = "unused"
    logger.disable_logs = False
    logger.cfg = {"num_steps_per_env": 24, "algorithm": {"rnd_cfg": None}}
    logger.num_envs, logger.gpu_world_size = 8, 1
    logger.tot_timesteps, logger.tot_time = 0, 0
    logger.writer = Mock()
    logger.teacher_rewbuffer = deque()
    logger.student_rewbuffer = deque()
    logger.log(
        it=0, start_it=0, total_it=1, collect_time=0.01, learn_time=0.01,
        loss_dict={}, learning_rate=1e-5, action_std=torch.ones(12), rnd_weight=None,
    )
    written = {call.args[0]: call.args[1] for call in logger.writer.add_scalar.call_args_list}
    assert written[PREFIX + "episodes_total"].item() == 8
    assert written[PREFIX + "success_rate_cumulative"].item() == pytest.approx(0.625)
    assert written["Episode/ordinary"].item() == 2.0
    assert logger.ep_extras == []
