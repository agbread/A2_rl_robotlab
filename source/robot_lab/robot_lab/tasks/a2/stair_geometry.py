"""Simulator-independent straight-stair geometry shared by training and evaluation.

Coordinates are relative to the robot's nominal start: x=0, y=0, ground z=0.
``num_steps`` counts vertical risers, including the rise onto the top tread.
The last tread and the added landing are coplanar, so the top platform has
length ``tread + landing``. Box sizes below are full extents (not half sizes).
"""

from __future__ import annotations

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class StairSpec:
    num_steps: int = 12
    rise: float = 0.15
    tread: float = 0.20
    approach: float = 1.5
    landing: float = 1.2
    width: float = 3.0

    def __post_init__(self):
        if isinstance(self.num_steps, bool) or not isinstance(self.num_steps, int) or self.num_steps < 1:
            raise ValueError("num_steps must be a positive integer riser count")
        for name in ("rise", "tread", "approach", "landing", "width"):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive, got {value}")

    @property
    def top_height(self) -> float:
        return self.num_steps * self.rise

    @property
    def top_start(self) -> float:
        return self.approach + (self.num_steps - 1) * self.tread

    @property
    def top_end(self) -> float:
        return self.approach + self.num_steps * self.tread + self.landing

    @property
    def down_end(self) -> float:
        """Ground-level exit when an optional symmetric descent is appended."""
        return self.top_end + (self.num_steps - 1) * self.tread

    @property
    def riser_positions(self) -> tuple[float, ...]:
        return tuple(self.approach + i * self.tread for i in range(self.num_steps))


@dataclass(frozen=True)
class StairBox:
    name: str
    center: tuple[float, float, float]
    size: tuple[float, float, float]

    @property
    def bounds(self) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
        return (tuple(c - s / 2 for c, s in zip(self.center, self.size)),
                tuple(c + s / 2 for c, s in zip(self.center, self.size)))


def get_stair_spec(num_steps: int = 12, rise: float = 0.15, tread: float = 0.20,
                   approach: float = 1.5, landing: float = 1.2, width: float = 3.0) -> StairSpec:
    return StairSpec(num_steps, rise, tread, approach, landing, width)


def build_stair_boxes(spec: StairSpec, include_descent: bool = False) -> tuple[StairBox, ...]:
    """Return solid boxes standing on z=0; no height-field quantization occurs."""
    boxes = []
    for i in range(spec.num_steps):
        height = (i + 1) * spec.rise
        boxes.append(StairBox(f"step_up_{i + 1:02d}",
                              (spec.approach + (i + 0.5) * spec.tread, 0.0, height / 2),
                              (spec.tread, spec.width, height)))
    boxes.append(StairBox("landing", (spec.top_end - spec.landing / 2, 0.0, spec.top_height / 2),
                          (spec.landing, spec.width, spec.top_height)))
    if include_descent:
        for i in range(spec.num_steps - 1):
            height = (spec.num_steps - i - 1) * spec.rise
            boxes.append(StairBox(f"step_down_{i + 1:02d}",
                                  (spec.top_end + (i + 0.5) * spec.tread, 0.0, height / 2),
                                  (spec.tread, spec.width, height)))
    return tuple(boxes)


def height_at(spec: StairSpec, x: float, y: float = 0.0, include_descent: bool = False) -> float:
    """Analytical top surface, with half-open x intervals at vertical risers."""
    if abs(y) > spec.width / 2 or x < spec.approach:
        return 0.0
    if x < spec.top_start:
        return min(spec.num_steps, math.floor((x - spec.approach) / spec.tread + 1e-10) + 1) * spec.rise
    if x < spec.top_end:
        return spec.top_height
    if include_descent and x < spec.down_end:
        return max(0, spec.num_steps - 1 - math.floor((x - spec.top_end) / spec.tread + 1e-10)) * spec.rise
    return 0.0


def curriculum_spec(level: int, *, mode: str = "steps", step_counts: tuple[int, ...] = (6, 8, 10, 12),
                    heights: tuple[float, ...] = (0.12, 0.135, 0.15),
                    num_steps: int = 12, rise: float = 0.15, tread: float = 0.20,
                    approach: float = 1.5, landing: float = 1.2, width: float = 3.0) -> StairSpec:
    """Resolve one discrete curriculum level, changing only its selected axis."""
    sequence = step_counts if mode == "steps" else heights if mode == "height" else None
    if sequence is None:
        raise ValueError(f"Unknown stair curriculum mode: {mode}")
    if not sequence or not 0 <= level < len(sequence):
        raise ValueError(f"Invalid level {level} for {mode} curriculum")
    if mode == "steps":
        num_steps = sequence[level]
    else:
        rise = sequence[level]
    return get_stair_spec(num_steps, rise, tread, approach, landing, width)


def validate_stair_boxes(spec: StairSpec, boxes: tuple[StairBox, ...], *, atol: float = 1e-9) -> None:
    """Check every ascent riser/tread, landing and solid-box lower surface."""
    if len(boxes) < spec.num_steps + 1:
        raise ValueError("Missing stair or landing boxes")
    for i, box in enumerate(boxes[:spec.num_steps]):
        lo, hi = box.bounds
        expected = (spec.approach + i * spec.tread, (i + 1) * spec.rise)
        values = ((lo[0], expected[0]), (hi[0] - lo[0], spec.tread),
                  (lo[2], 0.0), (hi[2], expected[1]), (hi[1] - lo[1], spec.width))
        if any(not math.isclose(a, b, abs_tol=atol, rel_tol=0.0) for a, b in values):
            raise ValueError(f"Incorrect dimensions for {box.name}")
    lo, hi = boxes[spec.num_steps].bounds
    if not (math.isclose(lo[0], spec.approach + spec.num_steps * spec.tread, abs_tol=atol)
            and math.isclose(hi[0], spec.top_end, abs_tol=atol)
            and math.isclose(hi[2], spec.top_height, abs_tol=atol)):
        raise ValueError("Incorrect landing dimensions")
