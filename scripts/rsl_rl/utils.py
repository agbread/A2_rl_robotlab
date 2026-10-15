# base version: IsaacLab/source/isaaclab_rl/isaaclab_rl/rsl_rl/exporter.py

import os
import torch
import re
import sys

from rsl_rl.utils.exporter_cts import _TorchPolicyExporter, export_cts_policy_as_jit

"Script to log terminal output to a file, stripping ANSI escape codes."
class Logger:
    def __init__(self, filename):
        self.terminal = sys.stdout
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        self.log = open(filename, 'w', encoding='utf-8')

        self.ansi_escape = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')

    def write(self, message):
        clean_message = self.ansi_escape.sub('', message)

        self.terminal.write(message)
        self.log.write(clean_message)
        self.log.flush()

    def flush(self):
        self.terminal.flush()
        self.log.flush()


def export_cts_policy_as_onnx(
    policy: object, path: str, actor_obs_normalizer: object | None = None, single_obs_normalizer: object | None = None, filename="policy.onnx", verbose=False
):
    """Export CTS policy into a Torch ONNX file.

    Args:
        policy: The CTS policy torch module.
        actor_obs_normalizer: The empirical normalizer module for actor observations. If None, Identity is used.
        single_obs_normalizer: The empirical normalizer module for single observations. If None, Identity is used.
        path: The path to the saving directory.
        filename: The name of exported ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    policy_exporter = _OnnxPolicyExporter(policy, actor_obs_normalizer, single_obs_normalizer, verbose)
    policy_exporter.export(path, filename)


"""
Helper Classes - Private.
"""


class _OnnxPolicyExporter(_TorchPolicyExporter):
    """Exporter of actor-critic into ONNX file."""

    def __init__(self, policy, actor_obs_normalizer=None, single_obs_normalizer=None, verbose=False):
        super().__init__(policy, actor_obs_normalizer, single_obs_normalizer)
        self.verbose = verbose

    def _extract_single_obs_from_history(self, history: torch.Tensor) -> torch.Tensor:
        if history.dim() == 1:
            history = history.unsqueeze(0)
        if history.shape[-1] != self.num_actor_obs:
            raise ValueError(
                f"Expected history last dimension {self.num_actor_obs}, got {history.shape[-1]}."
            )

        single_obs_terms = []
        offset = 0
        for dim in self.feature_dims:
            end = offset + dim * self.history_len
            single_obs_terms.append(history[:, end - dim:end])
            offset = end
        return torch.cat(single_obs_terms, dim=-1)

    def forward(self, history):
        if history.dim() == 1:
            history = history.unsqueeze(0)
        single_obs = self._extract_single_obs_from_history(history)
        single_obs = self.single_obs_normalizer(single_obs)
        obs_a = self.actor_obs_normalizer(history)
        latent, _ = self.student_moe_encoder(obs_a)
        latent_and_obs = torch.cat([latent, single_obs], dim=-1)
        if self.state_dependent_std:
            return self.actor(latent_and_obs)[..., 0, :]
        else:
            return self.actor(latent_and_obs)

    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        opset_version = 18  # was 11, but it caused problems with linux-aarch, and 18 worked well across all systems.
        torch.onnx.export(
            self,
            torch.zeros(1, self.num_actor_obs),
            os.path.join(path, filename),
            export_params=True,
            opset_version=opset_version,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )
