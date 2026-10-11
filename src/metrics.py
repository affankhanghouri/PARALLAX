"""Per-pair timing and tracking statistics."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class PairMetric:
    pair_index: int
    extraction_ms: float
    matching_ms: float
    geometry_ms: float
    core_total_ms: float
    visualization_ms: float
    matches: int
    inliers: int
    success: bool
    reason: str
    rotation_current_from_previous: list[float] | None = None
    translation_direction_current_frame: list[float] | None = None
    state: str = "VALID_MOTION"
    diagnostics: dict[str, float | int | None] | None = None


class Metrics:
    def __init__(self) -> None:
        self.pairs: list[PairMetric] = []

    def add(self, metric: PairMetric) -> None:
        self.pairs.append(metric)

    def summary(self) -> dict[str, float | int]:
        core = np.asarray([item.core_total_ms for item in self.pairs], dtype=float)
        successes = sum(item.success for item in self.pairs)
        states = {state: sum(item.state == state for item in self.pairs) for state in (
            "VALID_MOTION", "ROTATION_ONLY", "DEGENERATE", "TRACKING_LOST"
        )}
        if core.size == 0:
            return {"pairs": 0, "successes": 0, "failures": 0, "average_fps": 0.0}
        return {
            "pairs": len(self.pairs),
            "successes": successes,
            "failures": len(self.pairs) - successes,
            "average_fps": float(1000.0 / np.mean(core)) if np.mean(core) > 0 else 0.0,
            "core_latency_mean_ms": float(np.mean(core)),
            "core_latency_p50_ms": float(np.percentile(core, 50)),
            "core_latency_p95_ms": float(np.percentile(core, 95)),
            "extraction_mean_ms": float(np.mean([item.extraction_ms for item in self.pairs])),
            "matching_mean_ms": float(np.mean([item.matching_ms for item in self.pairs])),
            "geometry_mean_ms": float(np.mean([item.geometry_ms for item in self.pairs])),
            "visualization_mean_ms": float(np.mean([item.visualization_ms for item in self.pairs])),
            "inliers_mean": float(np.mean([item.inliers for item in self.pairs])),
            **{f"state_{key.lower()}": value for key, value in states.items()},
        }

    def save(self, directory: Path) -> None:
        payload = {"summary": self.summary(), "pairs": [asdict(item) for item in self.pairs]}
        (directory / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
