"""Dedicated A2 stair finetuning configuration; the rough-terrain task is intact."""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from .env_cfg import A2EnvCfg, A2SceneCfg, CommandsCfg, CurriculumCfg, TerminationsCfg
from .mdp.stairs import (
    StairVelocityCommandCfg, make_stair_terrain_cfg, reset_stair_root,
    stair_completed, stair_failed, stair_completion_curriculum, legacy_critic_contact_forces,
)


@configclass
class StairCommandsCfg(CommandsCfg):
    base_velocity = StairVelocityCommandCfg()


@configclass
class StairCurriculumCfg(CurriculumCfg):
    terrain_levels = CurrTerm(func=stair_completion_curriculum)
    # Preserve the mature 219k reward weights rather than restarting their ramps.
    base_linear_velocity = None
    base_height_l2 = None


@configclass
class StairTerminationsCfg(TerminationsCfg):
    # Completion is an artificial rollout cutoff, not a physical terminal failure.
    # Keep value bootstrapping consistent with the continuing locomotion objective.
    course_completed = DoneTerm(func=stair_completed, time_out=True)
    course_failed = DoneTerm(func=stair_failed)


@configclass
class A2StairsEnvCfg(A2EnvCfg):
    scene = A2SceneCfg(num_envs=4096, env_spacing=0.5)
    commands = StairCommandsCfg()
    curriculum = StairCurriculumCfg()
    terminations = StairTerminationsCfg()
    stair_curriculum_mode: str = "steps"

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_generator = make_stair_terrain_cfg(self.stair_curriculum_mode)
        self.scene.terrain.max_init_terrain_level = 0
        self.episode_length_s = 30.0
        # Four physics frames per command step, for force-block statistics.
        # Keep the critic at 275 values: it sees the original latest three frames.
        self.scene.contact_forces.history_length = self.decimation
        self.observations.critic.contact_force.func = legacy_critic_contact_forces
        self.events.reset_base.func = reset_stair_root
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.base_height_l2.weight = -10.0
