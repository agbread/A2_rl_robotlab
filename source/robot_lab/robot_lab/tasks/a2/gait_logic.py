"""Simulator-independent gait statistics. Foot order is FL, FR, RL, RR."""
from __future__ import annotations

import torch


class GaitStatistics:
    """Bounded rolling duty statistics and *completed* swing durations.

    An episode starting airborne is censored until that foot first contacts.
    No incomplete swing is counted as a completed one. Duty detects indefinitely
    raised feet independently of the completed-swing estimator.
    """

    def __init__(self, count, dt, device="cpu", window_s=2.0):
        self.dt = float(dt)
        self.window = round(window_s / dt)
        if self.window < 1:
            raise ValueError("window must contain at least one control step")
        self.history = torch.zeros(count, self.window, 4, device=device)
        self.samples = torch.zeros(count, dtype=torch.long, device=device)
        self.air = torch.zeros(count, 4, device=device)
        self.last_air = torch.zeros_like(self.air)
        self.seen_contact = torch.zeros_like(self.air, dtype=torch.bool)
        self.has_swing = torch.zeros_like(self.seen_contact)
        self.was_contact = torch.zeros_like(self.seen_contact)
        self.commands = torch.full((count, 3), float("nan"), device=device)
        self.cursor = 0

    def reset(self, ids):
        for value in (self.history, self.samples, self.air, self.last_air,
                      self.seen_contact, self.has_swing, self.was_contact):
            value[ids] = 0
        self.commands[ids] = float("nan")

    def update(self, fractions, commands, flat):
        if fractions.shape != self.air.shape or commands.shape != self.commands.shape:
            raise ValueError("Expected per-foot contact fractions and three commands")
        if not torch.isfinite(fractions).all() or not torch.isfinite(commands).all():
            raise ValueError("Nonfinite gait inputs")
        changed = ~torch.isclose(commands, self.commands, atol=1e-6, rtol=0).all(-1)
        self.reset(changed)
        self.commands.copy_(commands)
        contact = fractions >= 0.5
        landed = contact & ~self.was_contact & self.seen_contact & (self.air > 0)
        self.last_air[landed] = self.air[landed]
        self.has_swing |= landed
        self.air = torch.where(contact, 0.0, self.air + self.dt)
        self.seen_contact |= contact
        self.was_contact.copy_(contact)
        self.history[:, self.cursor] = fractions
        self.cursor = (self.cursor + 1) % self.window
        self.samples += 1
        eligible = (flat & (commands[:, 0] > .2) & (commands[:, 1].abs() < .05)
                    & (commands[:, 2].abs() < .05) & (self.samples >= self.window))
        duty = self.history.mean(1)
        duty_loss = ((duty[:, [0, 2]] - duty[:, [1, 3]]) ** 2).mean(-1)
        valid_pair = self.has_swing[:, [0, 2]] & self.has_swing[:, [1, 3]]
        swing_loss = (((self.last_air[:, [0, 2]] - self.last_air[:, [1, 3]]) / .4) ** 2
                      * valid_pair).mean(-1)
        return swing_loss * eligible, duty_loss * eligible
