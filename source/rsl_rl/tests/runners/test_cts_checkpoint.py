"""Checkpoint initialization must not silently restore a previous training schedule."""
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from rsl_rl.runners.on_policy_runner_cts import OnPolicyRunnerCTS


class Policy(torch.nn.Linear):
    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True


def runner():
    obj = OnPolicyRunnerCTS.__new__(OnPolicyRunnerCTS)
    obj.device = "cpu"
    obj.alg_cfg = {"rnd_cfg": None}
    policy = Policy(3, 2)
    student = torch.nn.Linear(3, 2)
    obj.alg = SimpleNamespace(
        policy=policy, optimizer=torch.optim.Adam(policy.parameters(), lr=1e-5),
        optimizer_stu_enc=torch.optim.Adam(student.parameters(), lr=1e-4),
        learning_rate=1e-5, finetune_step=0,
    )
    obj.current_learning_iteration = 0
    obj.cfg = {"num_steps_per_env": 24}
    obj.env = SimpleNamespace(common_step_counter=0,
                              get_training_state=lambda: {"curriculum_level": 2},
                              set_training_state=Mock())
    obj.logger = SimpleNamespace(save_model=Mock())
    obj.update_robogauge = Mock()
    return obj


def checkpoint(tmp_path):
    old = runner()
    with torch.no_grad():
        old.alg.policy.weight.fill_(0.42)
    old.alg.optimizer.param_groups[0]["lr"] = 3e-4
    old.alg.optimizer_stu_enc.param_groups[0]["lr"] = 1e-3
    old.alg.finetune_step = 271
    old.current_learning_iteration = 219000
    old.env.common_step_counter = 5256024
    path = tmp_path / "checkpoint.pt"
    old.save(str(path), it=219000, last_model=False)
    return path, old


def test_weights_only_preserves_fresh_optimizers_clocks_and_curriculum(tmp_path):
    path, old = checkpoint(tmp_path)
    new = runner()
    new.load_weights(str(path))
    assert torch.equal(new.alg.policy.weight, old.alg.policy.weight)
    assert new.alg.optimizer.param_groups[0]["lr"] == 1e-5
    assert new.alg.optimizer_stu_enc.param_groups[0]["lr"] == 1e-4
    assert new.alg.optimizer.state == {}
    assert new.current_learning_iteration == 0
    assert new.env.common_step_counter == 0
    assert new.alg.finetune_step == 0
    new.env.set_training_state.assert_not_called()
    assert new.initial_checkpoint["iteration"] == 219000


def test_resume_restores_both_optimizers_and_mirror_curriculum_clocks(tmp_path):
    path, _ = checkpoint(tmp_path)
    new = runner()
    new.load(str(path), map_location="cpu")
    assert new.alg.optimizer.param_groups[0]["lr"] == 3e-4
    assert new.alg.optimizer_stu_enc.param_groups[0]["lr"] == 1e-3
    assert new.alg.learning_rate == 3e-4
    assert new.current_learning_iteration == 219001
    assert new.env.common_step_counter == 5256024
    assert new.alg.finetune_step == 271
    new.env.set_training_state.assert_called_once_with({"curriculum_level": 2})


def test_old_checkpoint_without_new_keys_still_resumes(tmp_path):
    path, _ = checkpoint(tmp_path)
    data = torch.load(path, weights_only=False)
    for key in ("finetune_step", "env_training_state", "env_step_counter", "infos"):
        data.pop(key, None)
    torch.save(data, path)
    new = runner()
    assert new.load(str(path), map_location="cpu") is None
    assert new.alg.finetune_step == 0
    assert new.env.common_step_counter == 219001 * 24


def test_warm_start_provenance_survives_save_and_resume(tmp_path):
    path, _ = checkpoint(tmp_path)
    new = runner()
    new.load_weights(str(path))
    saved = tmp_path / "finetuned.pt"
    new.save(str(saved), it=0, last_model=False)
    resumed = runner()
    resumed.load(str(saved), map_location="cpu")
    assert resumed.initial_checkpoint == new.initial_checkpoint
