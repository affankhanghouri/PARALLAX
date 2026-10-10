"""Configuration loading and camera calibration validation."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml


@dataclass(frozen=True)
class CameraConfig:
    matrix: np.ndarray


@dataclass(frozen=True)
class AppConfig:
    image_dir: Path
    output_dir: Path
    camera: CameraConfig
    max_features: int = 2500
    ratio_test: float = 0.75
    min_matches: int = 20
    min_inliers: int = 15
    ransac_threshold_px: float = 1.0
    ransac_probability: float = 0.999
    save_match_every: int = 1
    max_frames: int | None = None


def validate_camera_matrix(value: Any) -> np.ndarray:
    """Return a validated 3x3 pinhole intrinsic matrix."""
    matrix = np.asarray(value, dtype=np.float64)
    if matrix.shape != (3, 3):
        raise ValueError(f"camera.matrix must have shape (3, 3), got {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("camera.matrix must contain only finite values")
    if matrix[0, 0] <= 0 or matrix[1, 1] <= 0:
        raise ValueError("camera focal lengths fx and fy must be positive")
    if not np.allclose(matrix[2], [0.0, 0.0, 1.0], atol=1e-9):
        raise ValueError("camera.matrix last row must be [0, 0, 1]")
    return matrix


def load_config(path: str | Path) -> AppConfig:
    """Load YAML configuration, resolving paths relative to the YAML file."""
    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    with config_path.open("r", encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    if not isinstance(raw, dict):
        raise ValueError("Configuration root must be a mapping")
    try:
        matrix = validate_camera_matrix(raw["camera"]["matrix"])
        image_dir_raw = raw["input"]["image_dir"]
    except KeyError as exc:
        raise ValueError(f"Missing required configuration key: {exc}") from exc

    base = config_path.parent

    def resolved(value: str) -> Path:
        candidate = Path(value).expanduser()
        return candidate.resolve() if candidate.is_absolute() else (base / candidate).resolve()

    features = raw.get("features", {})
    geometry = raw.get("geometry", {})
    output = raw.get("output", {})
    runtime = raw.get("runtime", {})
    app = AppConfig(
        image_dir=resolved(str(image_dir_raw)),
        output_dir=resolved(str(output.get("directory", "../outputs"))),
        camera=CameraConfig(matrix),
        max_features=int(features.get("max_features", 2500)),
        ratio_test=float(features.get("ratio_test", 0.75)),
        min_matches=int(features.get("min_matches", 20)),
        min_inliers=int(geometry.get("min_inliers", 15)),
        ransac_threshold_px=float(geometry.get("ransac_threshold_px", 1.0)),
        ransac_probability=float(geometry.get("ransac_probability", 0.999)),
        save_match_every=int(output.get("save_match_every", 1)),
        max_frames=(None if runtime.get("max_frames") is None else int(runtime["max_frames"])),
    )
    if not 0.0 < app.ratio_test < 1.0:
        raise ValueError("features.ratio_test must be between 0 and 1")
    if app.max_features <= 0 or app.min_matches < 5 or app.min_inliers < 5:
        raise ValueError("max_features must be positive; min_matches/min_inliers must be at least 5")
    if app.ransac_threshold_px <= 0 or not 0.0 < app.ransac_probability < 1.0:
        raise ValueError("Invalid RANSAC threshold or probability")
    if app.save_match_every < 0:
        raise ValueError("output.save_match_every cannot be negative")
    return app

