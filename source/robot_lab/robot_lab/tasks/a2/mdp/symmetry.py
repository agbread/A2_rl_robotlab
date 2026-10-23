# Copyright (c) 2026
# SPDX-License-Identifier: BSD-3-Clause

"""Sagittal reflection for the A2 MoE-CTS deployed policy observations.

The callback returns one reflected batch, not original-plus-reflected PPO data.
Only policy and single_obs are reflected; privileged critic observations
are carried through unchanged and MUST NOT be used as mirrored value inputs.
No simulator imports are needed, so the transform can also be audited offline.
"""

from __future__ import annotations

import torch
from tensordict import TensorDict


# Policy order: FL, FR, RL, RR; each leg is hip, thigh, calf.
_JOINT_PERMUTATION = (3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8)
_JOINT_SIGNS = (-1.0, 1.0, 1.0) * 4
_TERM_WIDTHS = (3, 3, 3, 12, 12, 12)


def mirror_a2_joints(values: torch.Tensor, *, signed: bool = True) -> torch.Tensor:
    """Reflect joint positions relative to default, velocities, or actions.

    A2's default hip angles (-0.1 left, +0.1 right) and common thigh/calf
    defaults already respect this reflection, so relative positions need no
    extra offset. For action standard deviations, use signed=False:
    standard deviations are permuted but never multiplied by negative signs.
    """
    if values.shape[-1] != 12:
        raise ValueError(f"A2 joint reflection requires 12 values, got {values.shape[-1]}.")
    permutation = torch.tensor(_JOINT_PERMUTATION, device=values.device)
    result = values.index_select(-1, permutation)
    if signed:
        result = result * values.new_tensor(_JOINT_SIGNS)
    return result


def _mirror_observation(values: torch.Tensor, history_length: int) -> torch.Tensor:
    expected = 45 * history_length
    if values.shape[-1] != expected:
        raise ValueError(f"A2 observation reflection requires {expected} values, got {values.shape[-1]}.")
    # IsaacLab concatenates flattened histories TERM BY TERM. Inside each term
    # the history is [time, feature], not [feature, time] or one global [time,45].
    chunks = values.split([width * history_length for width in _TERM_WIDTHS], dim=-1)
    reflected = []
    for index, (chunk, width) in enumerate(zip(chunks, _TERM_WIDTHS)):
        frames = chunk.reshape(*chunk.shape[:-1], history_length, width)
        if index == 0:  # Angular velocity is an axial vector.
            frames = frames * values.new_tensor((-1.0, 1.0, -1.0))
        elif index == 1:  # Projected gravity is a polar vector.
            frames = frames * values.new_tensor((1.0, -1.0, 1.0))
        elif index == 2:  # Planar velocity and axial yaw-rate command.
            frames = frames * values.new_tensor((1.0, -1.0, -1.0))
        else:  # Relative joint position, velocity, and previous action.
            frames = mirror_a2_joints(frames)
        reflected.append(frames.flatten(-2))
    return torch.cat(reflected, dim=-1)


def mirror_a2_observations(
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
    env=None,
    obs_type: str | None = None,
) -> tuple[TensorDict | None, torch.Tensor | None]:
    """Reflect raw 450-D policy history, independent 45-D current obs, and actions.

    env is accepted for the RSL-RL symmetry callback contract. This function
    deliberately does not read simulator state, change noise/normalizer state,
    concatenate batches, or invent a transformation for privileged observations.
    """
    if obs_type not in (None, "policy"):
        raise ValueError("A2 MoE-CTS reflection supports deployed policy observations only.")
    mirrored_obs = None
    if obs is not None:
        if not isinstance(obs, TensorDict):
            raise TypeError("A2 MoE-CTS reflection expects a TensorDict.")
        if "policy" not in obs or "single_obs" not in obs:
            raise ValueError("A2 reflection requires policy and single_obs observation groups.")
        mirrored_obs = obs.clone(recurse=False)
        mirrored_obs["policy"] = _mirror_observation(obs["policy"], history_length=10)
        mirrored_obs["single_obs"] = _mirror_observation(obs["single_obs"], history_length=1)
    mirrored_actions = mirror_a2_joints(actions) if actions is not None else None
    return mirrored_obs, mirrored_actions
