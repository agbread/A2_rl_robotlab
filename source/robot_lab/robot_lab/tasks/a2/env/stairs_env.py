"""A2 stair environment with checkpointable cohort curriculum state."""

import torch
from .a2_env import A2Env


class A2StairsEnv(A2Env):
    def get_training_state(self):
        term = self.command_manager.get_term("base_velocity")
        terrain = self.scene.terrain
        return {
            "version": 2,
            "require_settled_completion": term.cfg.require_settled_completion,
            "num_envs": self.num_envs,
            "curriculum_mode": self.cfg.stair_curriculum_mode,
            "terrain_levels": terrain.terrain_levels.detach().cpu().clone(),
            "cohort_history": term.cohort.history.detach().cpu().clone(),
            "cohort_counts": term.cohort.counts.detach().cpu().clone(),
            "cohort_positions": term.cohort.positions.detach().cpu().clone(),
            "unlocked_level": term.cohort.unlocked_level,
            "completed_episodes": term.completed_episodes.detach().cpu().clone(),
            "successful_episodes": term.successful_episodes.detach().cpu().clone(),
            "settled_episodes": term.settled_episodes.detach().cpu().clone(),
            "ascent_episodes": term.ascent_episodes.detach().cpu().clone(),
            "failed_episodes": term.failed_episodes.detach().cpu().clone(),
            "assigned_seconds_total": term.assigned_seconds_total.detach().cpu().clone(),
            "contact_seconds_total": term.contact_seconds_total.detach().cpu().clone(),
            "foot_contact_seconds_total": term.foot_contact_seconds_total.detach().cpu().clone(),
        }

    def set_training_state(self, state):
        if state.get("version") != 2 or state.get("num_envs") != self.num_envs:
            raise ValueError("Stair resume requires matching state version and environment count; use warm-start otherwise")
        if state.get("curriculum_mode") != self.cfg.stair_curriculum_mode:
            raise ValueError("Stair resume curriculum differs; use warm-start to change it")
        term = self.command_manager.get_term("base_velocity")
        terrain = self.scene.terrain
        if state.get("require_settled_completion") != term.cfg.require_settled_completion:
            raise ValueError("Stair resume completion criterion differs; use warm-start to change it")
        pairs = ((terrain.terrain_levels, "terrain_levels"),
                 (term.cohort.history, "cohort_history"), (term.cohort.counts, "cohort_counts"),
                 (term.cohort.positions, "cohort_positions"),
                 (term.completed_episodes, "completed_episodes"), (term.successful_episodes, "successful_episodes"),
                 (term.settled_episodes, "settled_episodes"),
                 (term.ascent_episodes, "ascent_episodes"), (term.failed_episodes, "failed_episodes"),
                 (term.assigned_seconds_total, "assigned_seconds_total"),
                 (term.contact_seconds_total, "contact_seconds_total"),
                 (term.foot_contact_seconds_total, "foot_contact_seconds_total"))
        for tensor, name in pairs:
            value = state[name]
            if tuple(value.shape) != tuple(tensor.shape):
                raise ValueError(f"Stair resume shape mismatch for {name}")
        levels = state["terrain_levels"]
        unlocked = int(state["unlocked_level"])
        if (levels < 0).any() or (levels >= terrain.max_terrain_level).any() or not 0 <= unlocked < terrain.max_terrain_level:
            raise ValueError("Invalid checkpoint terrain levels")
        for tensor, name in pairs:
            tensor.copy_(state[name].to(device=self.device, dtype=tensor.dtype))
        term.cohort.unlocked_level = unlocked
        terrain.env_origins[:] = terrain.terrain_origins[terrain.terrain_levels, terrain.terrain_types]
        # Resume the curriculum, not a partially simulated physical episode.
        self.episode_length_buf.zero_()
        self.reset()
