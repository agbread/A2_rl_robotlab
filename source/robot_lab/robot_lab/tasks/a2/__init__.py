# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Package containing task implementations for various robotic environments."""

import os
import toml
import gymnasium as gym

from isaaclab_tasks.utils import import_packages

##
# Register Gym environments.
##
gym.register(
    id="RobotLab-A2-v0",
    entry_point="robot_lab.tasks.a2.env.a2_env:A2Env",
    # entry_point="robot_lab.tasks.a2.env.a2_env:ActionDelayA2Env", # using motor-level delay, so disabled env-level delay
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.env_cfg:A2EnvCfg",
        "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:MoECTSRunnerCfg",
    },
)

# Matched stair experiments: identical environment, optional mirror loss.
for task_id, runner_cfg in (
    ("RobotLab-A2-Stairs-v0", "StairsMoECTSRunnerCfg"),
    ("RobotLab-A2-Stairs-Sym-v0", "StairsSymMoECTSRunnerCfg"),
):
    gym.register(
        id=task_id,
        entry_point="robot_lab.tasks.a2.env.stairs_env:A2StairsEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.stairs_env_cfg:A2StairsEnvCfg",
            "rsl_rl_cfg_entry_point": f"{__name__}.rsl_rl_cfg:{runner_cfg}",
        },
    )

# The blacklist is used to prevent importing configs from sub-packages
_BLACKLIST_PKGS = ["utils"]
# Import all configs in this package
import_packages(__name__, _BLACKLIST_PKGS)
