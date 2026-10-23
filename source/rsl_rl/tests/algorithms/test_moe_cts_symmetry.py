"""CPU checks for opt-in MoE-CTS mirror loss; no Isaac Sim or physics needed."""

from copy import deepcopy
import importlib.util
from pathlib import Path

import pytest
import torch
from tensordict import TensorDict

from rsl_rl.algorithms.moe_cts import MoECTS
from rsl_rl.modules.actor_critic_moe_cts import ActorCriticMoECTS
from rsl_rl.storage import RolloutStorageCTS


# Loading the pure transform by file avoids robot_lab's simulator registration
# side effects. Production resolves the same file through its normal module path.
_path = Path(__file__).resolve().parents[3] / "robot_lab/robot_lab/tasks/a2/mdp/symmetry.py"
_spec = importlib.util.spec_from_file_location("a2_mirror_transform_test", _path)
_symmetry = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_symmetry)
reflect = _symmetry.mirror_a2_observations
reflect_joints = _symmetry.mirror_a2_joints


@pytest.fixture(autouse=True)
def cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


def observations(count=8):
    return TensorDict(
        {"policy": torch.randn(count, 450), "single_obs": torch.randn(count, 45), "critic": torch.randn(count, 263)},
        batch_size=[count],
    )


def symmetry_config(coefficient=0.1, ramp=0):
    return {
        "use_mirror_loss": True,
        "use_data_augmentation": False,
        "mirror_loss_coeff": coefficient,
        "ramp_updates": ramp,
        "data_augmentation_func": reflect,
    }


def make_algorithm(config=None, *, normalize=False, state_dependent_std=False):
    obs = observations()
    policy = ActorCriticMoECTS(
        obs, {"policy": ["policy"], "critic": ["critic"]}, 12,
        actor_hidden_dims=[16], critic_hidden_dims=[16], teacher_encoder_hidden_dims=[16],
        student_encoder_hidden_dims=[16, 8], latent_dim=8, expert_num=2,
        actor_obs_normalization=normalize, state_dependent_std=state_dependent_std,
    )
    storage = RolloutStorageCTS("rl", 8, 6, 2, obs, [12])
    return MoECTS(
        policy, storage, 8, num_learning_epochs=1, num_mini_batches=1,
        schedule="fixed", symmetry_cfg=config,
    )


def fill_rollout(algorithm):
    with torch.no_grad():
        for _ in range(2):
            obs = observations()
            algorithm.act(obs)
            algorithm.process_env_step(obs, torch.randn(8), torch.zeros(8, dtype=torch.bool), {})
        algorithm.compute_returns(observations())


def test_reflection_is_involution_and_does_not_mutate_inputs():
    torch.manual_seed(17)
    obs, actions = observations(), torch.randn(8, 12)
    before = obs.clone()
    mirrored, mirrored_actions = reflect(obs, actions)
    restored, restored_actions = reflect(mirrored, mirrored_actions)
    for name in obs.keys():
        assert torch.equal(restored[name], obs[name])
        assert torch.equal(obs[name], before[name])
    assert torch.equal(restored_actions, actions)
    assert mirrored.batch_size == obs.batch_size
    assert torch.equal(mirrored["critic"], obs["critic"])


def test_history_sentinels_preserve_time_and_reflect_every_term():
    # Encode time and feature separately, then pack as IsaacLab does.
    frames = 1000.0 * torch.arange(10)[:, None] + torch.arange(45)[None, :] + 1
    slices = (slice(0, 3), slice(3, 6), slice(6, 9), slice(9, 21), slice(21, 33), slice(33, 45))
    packed = torch.cat([frames[:, part].flatten() for part in slices]).unsqueeze(0)
    current = (torch.arange(45) + 20000.0).unsqueeze(0)  # Not the newest history frame.
    obs = TensorDict({"policy": packed, "single_obs": current, "critic": torch.ones(1, 263)}, [1])
    result, _ = reflect(obs)
    joint_perm = torch.tensor([3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8])
    permutation = torch.cat([torch.arange(9), *(joint_perm + start for start in (9, 21, 33))])
    signs = torch.tensor([-1, 1, -1, 1, -1, 1, 1, -1, -1] + [-1, 1, 1] * 12)
    expected_frames = frames[:, permutation] * signs
    expected_history = torch.cat([expected_frames[:, part].flatten() for part in slices]).unsqueeze(0)
    assert torch.equal(result["policy"], expected_history)
    assert torch.equal(result["single_obs"], current[:, permutation] * signs)


def test_joint_signs_and_gaussian_standard_deviation_permutation():
    joints = torch.arange(1.0, 13.0)
    assert torch.equal(reflect_joints(joints), torch.tensor([-4, 5, 6, -1, 2, 3, -10, 11, 12, -7, 8, 9]))
    std = reflect_joints(joints, signed=False)
    assert (std > 0).all()
    assert torch.equal(std, torch.tensor([4, 5, 6, 1, 2, 3, 10, 11, 12, 7, 8, 9]))
    mean, sample = torch.randn(3, 12), torch.randn(3, 12)
    original = torch.distributions.Normal(mean, joints).log_prob(sample).sum(-1)
    mirrored = torch.distributions.Normal(reflect_joints(mean), std).log_prob(reflect_joints(sample)).sum(-1)
    torch.testing.assert_close(mirrored, original)
    defaults = torch.tensor([-0.1, 0.9, -1.8, 0.1, 0.9, -1.8] * 2)
    assert torch.equal(reflect_joints(defaults), defaults)


@pytest.mark.parametrize("train_actor", [True, False])
@pytest.mark.parametrize("state_dependent_std", [False, True])
def test_mirror_loss_updates_only_its_intended_optimizer(train_actor, state_dependent_std):
    torch.manual_seed(21)
    algorithm = make_algorithm(symmetry_config(), state_dependent_std=state_dependent_std)
    policy = algorithm.policy
    obs = observations()
    before = {name: parameter.detach().clone() for name, parameter in policy.named_parameters()}
    policy.zero_grad(set_to_none=True)
    with torch.no_grad():
        policy.act(obs, is_teacher=False)
    distribution = policy.distribution
    loss = algorithm._mirror_consistency_loss(obs, train_actor=train_actor)
    assert loss.isfinite() and loss > 0
    loss.backward()
    intended_prefix = "actor." if train_actor else "student_moe_encoder."
    intended_gradients = []
    for name, parameter in policy.named_parameters():
        if name.startswith(intended_prefix):
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
            intended_gradients.append(parameter.grad.abs().sum())
        else:
            assert parameter.grad is None, name
    assert sum(intended_gradients) > 0
    optimizer = algorithm.optimizer if train_actor else algorithm.optimizer_stu_enc
    optimizer.step()
    assert any(
        not torch.equal(before[name], parameter)
        for name, parameter in policy.named_parameters() if name.startswith(intended_prefix)
    )
    for name, parameter in policy.named_parameters():
        if not name.startswith(intended_prefix):
            assert torch.equal(before[name], parameter), name
    assert policy.distribution is distribution


def test_reflection_precedes_nontrivial_normalization_without_updating_statistics():
    torch.manual_seed(30)
    algorithm = make_algorithm(symmetry_config(), normalize=True)
    policy = algorithm.policy
    obs = observations()
    policy.update_normalization(TensorDict(
        {"policy": 3 * obs["policy"] + 5, "single_obs": 2 * obs["single_obs"] + 7, "critic": obs["critic"]}, [8]
    ))
    before = {name: value.clone() for name, value in policy.named_buffers()}
    encoder_inputs, actor_inputs = [], []
    handles = [
        policy.student_moe_encoder.register_forward_pre_hook(lambda module, args: encoder_inputs.append(args[0].clone())),
        policy.actor.register_forward_pre_hook(lambda module, args: actor_inputs.append(args[0].clone())),
    ]
    try:
        algorithm._mirror_consistency_loss(obs, train_actor=True)
    finally:
        for handle in handles:
            handle.remove()
    mirrored, _ = reflect(obs)
    assert torch.equal(encoder_inputs[0], policy.actor_obs_normalizer(obs["policy"]))
    assert torch.equal(encoder_inputs[1], policy.actor_obs_normalizer(mirrored["policy"]))
    assert torch.equal(actor_inputs[1][:, -45:], policy.single_obs_normalizer(mirrored["single_obs"]))
    for name, value in policy.named_buffers():
        assert torch.equal(before[name], value), name


def test_disabled_mode_preserves_full_update_and_rng_exactly():
    def forbidden_callback(**kwargs):
        raise AssertionError("Disabled mirror loss called the reflector.")
    disabled = {"use_mirror_loss": False, "data_augmentation_func": forbidden_callback}
    torch.manual_seed(43)
    original = make_algorithm()
    torch.manual_seed(43)
    explicit_off = make_algorithm(disabled)
    torch.manual_seed(44)
    fill_rollout(original)
    torch.manual_seed(44)
    fill_rollout(explicit_off)
    torch.manual_seed(45)
    original_losses = original.update()
    original_rng = torch.random.get_rng_state()
    torch.manual_seed(45)
    off_losses = explicit_off.update()
    assert original_losses == off_losses
    assert torch.equal(original_rng, torch.random.get_rng_state())
    for name, value in original.policy.state_dict().items():
        assert torch.equal(value, explicit_off.policy.state_dict()[name]), name


def test_enabled_full_update_has_finite_logs_ramp_and_changed_parameters():
    torch.manual_seed(51)
    algorithm = make_algorithm(symmetry_config(coefficient=0.2, ramp=4))
    fill_rollout(algorithm)
    before = deepcopy(algorithm.policy.state_dict())
    losses = algorithm.update()
    assert losses["mirror_coefficient"] == pytest.approx(0.05)
    assert losses["mirror_actor"] > 0 and losses["mirror_student"] > 0
    assert all(torch.isfinite(torch.tensor(value)) for value in losses.values())
    assert algorithm.finetune_step == 1
    assert algorithm._mirror_loss_coefficient() == pytest.approx(0.1)
    for prefix in ("actor.", "student_moe_encoder."):
        assert any(
            not torch.equal(before[name], value)
            for name, value in algorithm.policy.state_dict().items() if name.startswith(prefix)
        )
    algorithm.finetune_step = 200
    assert algorithm._mirror_loss_coefficient() == pytest.approx(0.2)


def test_unsupported_augmentation_and_invalid_coefficients_fail_explicitly():
    with pytest.raises(NotImplementedError, match="data augmentation"):
        make_algorithm({"use_data_augmentation": True})
    for coefficient in (-1, float("inf"), float("nan")):
        with pytest.raises(ValueError, match="mirror_loss_coeff"):
            make_algorithm(symmetry_config(coefficient))
    for ramp in (-1, 1.5, True):
        with pytest.raises(ValueError, match="ramp_updates"):
            make_algorithm(symmetry_config(ramp=ramp))
    with pytest.raises(ValueError, match="policy observations only"):
        reflect(observations(), obs_type="critic")
    with pytest.raises(ValueError, match="450"):
        reflect(TensorDict({"policy": torch.zeros(1, 45), "single_obs": torch.zeros(1, 45)}, [1]))


def test_concatenating_ppo_callback_is_rejected():
    def concatenating_callback(obs=None, actions=None, env=None):
        mirrored_obs, mirrored_actions = reflect(obs, actions, env)
        return (
            torch.cat([obs, mirrored_obs]) if obs is not None else None,
            torch.cat([actions, mirrored_actions]) if actions is not None else None,
        )
    config = symmetry_config()
    config["data_augmentation_func"] = concatenating_callback
    algorithm = make_algorithm(config)
    with pytest.raises(ValueError, match="without concatenating"):
        algorithm._mirror_consistency_loss(observations(), train_actor=True)
