"""Export an A2 CTS checkpoint without launching Isaac Sim.

Reads the adjacent params/agent.yaml by default. Example:
  python scripts/a2_stairs/export_checkpoint.py --checkpoint logs/.../model_199.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from tensordict import TensorDict
import yaml

from rsl_rl.modules import ActorCriticMoECTS
from rsl_rl.utils.exporter_cts import export_cts_policy_as_jit


class ConfigLoader(yaml.SafeLoader):
    """Support tuples emitted by IsaacLab, without constructing Python objects."""


ConfigLoader.add_constructor("tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node)))


def export_checkpoint(checkpoint: Path, config: Path | None = None, output: Path | None = None) -> Path:
    checkpoint = checkpoint.resolve()
    config = config or checkpoint.parent / "params/agent.yaml"
    output = output or checkpoint.parent / "exported" / f"{checkpoint.stem}.pt"
    with config.open() as stream:
        policy_cfg = yaml.load(stream, Loader=ConfigLoader)["policy"]
    if policy_cfg.pop("class_name") != "ActorCriticMoECTS":
        raise ValueError("Expected an ActorCriticMoECTS checkpoint")
    saved = torch.load(checkpoint, map_location="cpu", weights_only=False)
    state = saved["model_state_dict"]
    critic_dim = state["teacher_encoder.0.network.0.weight"].shape[1]
    obs = TensorDict({"policy": torch.zeros(1, 450), "single_obs": torch.zeros(1, 45),
                      "critic": torch.zeros(1, critic_dim)}, batch_size=[1])
    policy = ActorCriticMoECTS(obs=obs, obs_groups={"policy": ["policy"], "critic": ["critic"]},
                              num_actions=12, **policy_cfg)
    policy.load_state_dict(state, strict=True)
    policy.eval()
    export_cts_policy_as_jit(policy, policy.actor_obs_normalizer, policy.single_obs_normalizer,
                             str(output.parent), output.name)
    # Check the deployment wrapper against direct student inference over a real
    # term-major rolling history, including reset behavior.
    deployed = torch.jit.load(str(output)).eval()
    generator = torch.Generator().manual_seed(42)
    history = torch.zeros(1, 450)
    with torch.inference_mode():
        for _ in range(20):
            single = torch.randn(1, 45, generator=generator)
            ho = so = 0
            for dim in (3, 3, 3, 12, 12, 12):
                end = ho + dim * 10
                history[:, ho:end] = torch.cat((history[:, ho + dim:end], single[:, so:so + dim]), -1)
                ho, so = end, so + dim
            current = TensorDict({"policy": history, "single_obs": single}, batch_size=[1])
            expected = policy.act_inference(current)
            torch.testing.assert_close(deployed(single), expected, rtol=1e-5, atol=1e-6)
        deployed.reset()
        if torch.count_nonzero(deployed.obs_history):
            raise AssertionError("Deployment history reset failed")
    print(f"Exported and verified: {output}")
    return output


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    export_checkpoint(args.checkpoint, args.config, args.output)
