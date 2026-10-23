"""Torch-only stair completion and curriculum logic, testable without Isaac Sim."""

from __future__ import annotations

from dataclasses import dataclass
import math
import torch


@dataclass(frozen=True)
class StairCriteria:
    foot_radius: float = 0.032
    height_tolerance: float = 0.02
    contact_force: float = 10.0
    settle_seconds: float = 0.4
    settle_tilt_degrees: float = 30.0
    failure_tilt_degrees: float = 60.0
    lateral_margin: float = 0.10


class StairCourseTracker:
    """Track each episode using current feet and force samples.

    ``specs`` columns: num_steps, rise, tread, approach, landing, width.
    Contact duration uses all supplied physics force samples. Surface membership
    uses current foot positions at the control rate.
    """

    def __init__(self, num_envs: int, device: str, criteria: StairCriteria | None = None):
        self.criteria = criteria or StairCriteria()
        self.stop = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.success = torch.zeros_like(self.stop)
        self.ascent_completed = torch.zeros_like(self.stop)
        self.top_contact_seen = torch.zeros((num_envs, 4), dtype=torch.bool, device=device)
        self.failed = torch.zeros_like(self.stop)
        self.settle_time = torch.zeros(num_envs, device=device)
        self.progress_steps = torch.zeros_like(self.settle_time)
        self.contact_time_s = torch.zeros_like(self.settle_time)
        self.foot_contact_time_s = torch.zeros_like(self.settle_time)
        self.elapsed_s = torch.zeros_like(self.settle_time)
        self.stage_roll_sum = torch.zeros((num_envs, 3), device=device)
        self.stage_roll_abs_sum = torch.zeros_like(self.stage_roll_sum)
        self.stage_samples = torch.zeros_like(self.stage_roll_sum)

    def reset(self, ids):
        for value in vars(self).values():
            if isinstance(value, torch.Tensor):
                value[ids] = 0

    def update(self, feet: torch.Tensor, force_z: torch.Tensor, root: torch.Tensor,
               gravity_b: torch.Tensor, specs: torch.Tensor, active: torch.Tensor, dt: float):
        c = self.criteria
        force_samples = force_z if force_z.ndim == 3 else force_z[:, None, :]
        contact_samples = force_samples > c.contact_force
        force_z = force_samples.mean(dim=1)
        stance_fraction = contact_samples.to(feet.dtype).mean(dim=1)
        n, rise, tread, approach, landing, width = specs.unbind(-1)
        top_start = approach + (n - 1) * tread
        top_end = approach + n * tread + landing
        top_height = n * rise
        x, y, z = feet.unbind(-1)
        inside_y = y.abs() <= (width[:, None] / 2 - c.foot_radius)
        inside_top = ((x >= top_start[:, None] + c.foot_radius)
                      & (x <= top_end[:, None] - c.foot_radius) & inside_y)
        top_height_ok = (z - (top_height[:, None] + c.foot_radius)).abs() <= c.height_tolerance
        on_top = inside_top & top_height_ok
        # A walking gait need not have four simultaneous contacts. Remember each
        # valid top contact, then stop only when all feet have reached the platform.
        self.top_contact_seen |= on_top & (force_z > c.contact_force) & active[:, None]
        reached = active & self.top_contact_seen.all(dim=1) & inside_top.all(dim=1)
        reached &= (z >= top_height[:, None] + c.foot_radius - c.height_tolerance).all(dim=1)
        self.ascent_completed |= reached
        self.stop |= reached
        tilt_cos = -gravity_b[:, 2]
        stable = (active & self.stop & on_top.all(dim=1)
                  & (force_z > c.contact_force).all(dim=1)
                  & (stance_fraction >= 0.5).all(dim=1)
                  & (tilt_cos > math.cos(math.radians(c.settle_tilt_degrees))))
        self.settle_time[:] = torch.where(stable, self.settle_time + dt, 0.0)
        lateral = (y.abs() + c.foot_radius > width[:, None] / 2).any(dim=1)
        fallen = tilt_cos < math.cos(math.radians(c.failure_tilt_degrees))
        overrun = (x + c.foot_radius > top_end[:, None]).any(dim=1)
        nonfinite = ~(torch.isfinite(root).all(dim=1) & torch.isfinite(feet).all(dim=(1, 2)))
        self.failed |= active & (lateral | fallen | overrun | nonfinite)
        self.success |= active & ~self.failed & (self.settle_time >= c.settle_seconds - 1e-6)

        crossed = torch.floor((torch.nan_to_num(x) - c.foot_radius - approach[:, None]) / tread[:, None]) + 1
        crossed = torch.minimum(crossed.clamp(min=0), n[:, None]).amin(dim=1)
        self.progress_steps[:] = torch.maximum(self.progress_steps, torch.where(active, crossed, 0.0))
        local_steps = torch.floor((x - approach[:, None]) / tread[:, None]) + 1
        local_steps = torch.minimum(local_steps.clamp(min=0), n[:, None])
        surface = local_steps * rise[:, None]
        on_stair = ((x >= approach[:, None]) & (x <= top_end[:, None]) & inside_y
                    & ((z - surface - c.foot_radius).abs() <= c.height_tolerance)
                    & active[:, None])
        stair_contacts = on_stair[:, None, :] & contact_samples
        self.contact_time_s += stair_contacts.any(dim=2).to(root.dtype).mean(dim=1) * dt
        self.foot_contact_time_s += stair_contacts.sum(dim=2).to(root.dtype).mean(dim=1) * dt
        self.elapsed_s += active.to(root.dtype) * dt
        # Signed and absolute roll at early/middle/late stair progress, not just an episode mean.
        roll = torch.nan_to_num(torch.atan2(-gravity_b[:, 1], -gravity_b[:, 2]))
        stage = torch.minimum((3 * crossed / n.clamp(min=1)).long(), torch.full_like(n.long(), 2))
        climbing = active & (crossed > 0) & ~self.success
        for i in range(3):
            mask = climbing & (stage == i)
            self.stage_roll_sum[:, i] += torch.where(mask, roll, 0.0)
            self.stage_roll_abs_sum[:, i] += torch.where(mask, roll.abs(), 0.0)
            self.stage_samples[:, i] += mask.to(root.dtype)


def resolve_stair_completion(ascent_completed, settled_success, failed, require_settled_completion=False):
    """Select the task's success criterion, always allowing failure to veto it."""
    completed = settled_success if require_settled_completion else ascent_completed
    return completed & ~failed


class StairCohortCurriculum:
    """Unlock the next level after 90% success over the latest 100 completed episodes.

    Windows pool environments at the same level. Origins only change on reset;
    fixed-target and neighboring terrain environments never enter this object.
    """
    def __init__(self, num_levels, device, window_size=100, success_threshold=0.90):
        if num_levels < 1 or window_size < 1 or not 0 < success_threshold <= 1:
            raise ValueError("Invalid cohort curriculum parameters")
        self.history = torch.zeros((num_levels, window_size), dtype=torch.bool, device=device)
        self.counts = torch.zeros(num_levels, dtype=torch.long, device=device)
        self.positions = torch.zeros_like(self.counts)
        self.unlocked_level = 0
        self.window_size = window_size
        self.success_threshold = success_threshold

    def record(self, levels, successes):
        for level in range(self.history.shape[0]):
            values = successes[levels == level].bool()
            total = values.numel()
            if not total:
                continue
            pos = int(self.positions[level].item())
            if total >= self.window_size:
                self.history[level] = values[-self.window_size:]
                self.positions[level] = 0
            else:
                idx = (torch.arange(total, device=levels.device) + pos) % self.window_size
                self.history[level, idx] = values
                self.positions[level] = (pos + total) % self.window_size
            self.counts[level] += total
            if (level == self.unlocked_level and level < self.history.shape[0]-1
                    and self.counts[level] >= self.window_size
                    and self.history[level].float().mean() >= self.success_threshold):
                self.unlocked_level += 1
        return self.unlocked_level
