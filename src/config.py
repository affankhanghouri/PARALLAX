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
    geometric_reliability: bool = False
    rotation_only_residual_degrees: float = 0.2
    rotation_only_min_ratio: float = 0.6
    rotation_innovation_degrees: float = 2.0
    suspicious_parallax_p10_degrees: float = 1.0
    rotation_history_size: int = 5
    initialization_min_matches: int = 50
    initialization_min_inliers: int = 50
    initialization_min_median_parallax_degrees: float = 1.0
    initialization_min_p25_parallax_degrees: float = 0.5
    initialization_grid_rows: int = 4
    initialization_grid_columns: int = 6
    initialization_min_grid_ratio: float = 1.0 / 3.0
    triangulation_max_reprojection_error_px: float = 2.0
    triangulation_min_angle_degrees: float = 1.0
    triangulation_max_depth: float = 100.0
    pnp_max_descriptor_distance: float = 64.0
    pnp_min_correspondences: int = 8
    pnp_min_inliers: int = 6
    pnp_min_inlier_ratio: float = 0.5
    pnp_ransac_reprojection_error_px: float = 3.0
    pnp_max_median_reprojection_error_px: float = 2.0
    pnp_max_p95_reprojection_error_px: float = 4.0
    pnp_min_occupied_grid_cells: int = 4
    pnp_max_rotation_jump_degrees: float = 15.0
    pnp_max_translation_jump_arbitrary: float = 2.0
    association_mode: str = "global"
    projection_search_radius_px: float = 35.0
    association_descriptor_reference_limit: int = 0
    descriptor_update_policy: str = "fixed"
    descriptor_bank_max_size: int = 3
    optical_flow_enabled: bool = False
    optical_flow_max_forward_backward_error_px: float = 1.5
    optical_flow_max_lk_error: float = 20.0
    landmark_quality_enabled: bool = False
    landmark_max_active: int = 1000
    landmark_max_visible_misses: int = 1000
    mapping_enabled: bool = False
    keyframe_min_frame_gap: int = 2
    keyframe_max_frame_gap: int = 10
    keyframe_min_translation_arbitrary: float = 0.3
    keyframe_min_rotation_degrees: float = 2.0
    keyframe_support_ratio_trigger: float = 0.5
    keyframe_critical_support_ratio: float = 0.1
    keyframe_critical_min_translation_arbitrary: float = 0.3
    keyframe_critical_min_frame_gap: int = 2
    mapping_min_matches: int = 20
    mapping_epipolar_threshold_px: float = 1.5
    mapping_max_reprojection_error_px: float = 2.0
    mapping_min_triangulation_angle_degrees: float = 1.0
    mapping_max_depth_arbitrary: float = 100.0
    mapping_min_new_landmarks: int = 12
    mapping_min_occupied_grid_cells: int = 6
    mapping_duplicate_radius_px: float = 3.0
    mapping_max_landmarks: int = 1000
    mapping_max_new_landmarks_per_keyframe: int = 1000
    mapping_max_new_landmarks_per_grid_cell: int = 1000
    mapping_efficient_candidate_reverse: bool = False


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
    initialization = raw.get("initialization", {})
    triangulation = raw.get("triangulation", {})
    tracking = raw.get("tracking", {})
    mapping = raw.get("mapping", {})
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
        geometric_reliability=bool(geometry.get("reliability_checks", False)),
        rotation_only_residual_degrees=float(geometry.get("rotation_only_residual_degrees", 0.2)),
        rotation_only_min_ratio=float(geometry.get("rotation_only_min_ratio", 0.6)),
        rotation_innovation_degrees=float(geometry.get("rotation_innovation_degrees", 2.0)),
        suspicious_parallax_p10_degrees=float(geometry.get("suspicious_parallax_p10_degrees", 1.0)),
        rotation_history_size=int(geometry.get("rotation_history_size", 5)),
        initialization_min_matches=int(initialization.get("min_matches", 50)),
        initialization_min_inliers=int(initialization.get("min_geometric_inliers", 50)),
        initialization_min_median_parallax_degrees=float(initialization.get("min_median_parallax_degrees", 1.0)),
        initialization_min_p25_parallax_degrees=float(initialization.get("min_p25_parallax_degrees", 0.5)),
        initialization_grid_rows=int(initialization.get("grid_rows", 4)),
        initialization_grid_columns=int(initialization.get("grid_columns", 6)),
        initialization_min_grid_ratio=float(initialization.get("min_occupied_grid_ratio", 1.0 / 3.0)),
        triangulation_max_reprojection_error_px=float(triangulation.get("max_reprojection_error_px", 2.0)),
        triangulation_min_angle_degrees=float(triangulation.get("min_angle_degrees", 1.0)),
        triangulation_max_depth=float(triangulation.get("max_depth_arbitrary_units", 100.0)),
        pnp_max_descriptor_distance=float(tracking.get("max_descriptor_distance", 64.0)),
        pnp_min_correspondences=int(tracking.get("min_correspondences", 8)),
        pnp_min_inliers=int(tracking.get("min_inliers", 6)),
        pnp_min_inlier_ratio=float(tracking.get("min_inlier_ratio", 0.5)),
        pnp_ransac_reprojection_error_px=float(tracking.get("ransac_reprojection_error_px", 3.0)),
        pnp_max_median_reprojection_error_px=float(tracking.get("max_median_reprojection_error_px", 2.0)),
        pnp_max_p95_reprojection_error_px=float(tracking.get("max_p95_reprojection_error_px", 4.0)),
        pnp_min_occupied_grid_cells=int(tracking.get("min_occupied_grid_cells", 4)),
        pnp_max_rotation_jump_degrees=float(tracking.get("max_rotation_jump_degrees", 15.0)),
        pnp_max_translation_jump_arbitrary=float(tracking.get("max_translation_jump_arbitrary", 2.0)),
        association_mode=str(tracking.get("association_mode", "global")),
        projection_search_radius_px=float(tracking.get("projection_search_radius_px", 35.0)),
        association_descriptor_reference_limit=int(tracking.get("association_descriptor_reference_limit", 0)),
        descriptor_update_policy=str(tracking.get("descriptor_update_policy", "fixed")),
        descriptor_bank_max_size=int(tracking.get("descriptor_bank_max_size", 3)),
        optical_flow_enabled=bool(tracking.get("optical_flow_enabled", False)),
        optical_flow_max_forward_backward_error_px=float(tracking.get("optical_flow_max_forward_backward_error_px", 1.5)),
        optical_flow_max_lk_error=float(tracking.get("optical_flow_max_lk_error", 20.0)),
        landmark_quality_enabled=bool(tracking.get("landmark_quality_enabled", False)),
        landmark_max_active=int(tracking.get("landmark_max_active", 1000)),
        landmark_max_visible_misses=int(tracking.get("landmark_max_visible_misses", 1000)),
        mapping_enabled=bool(mapping.get("enabled", False)),
        keyframe_min_frame_gap=int(mapping.get("keyframe_min_frame_gap", 2)),
        keyframe_max_frame_gap=int(mapping.get("keyframe_max_frame_gap", 10)),
        keyframe_min_translation_arbitrary=float(mapping.get("keyframe_min_translation_arbitrary", 0.3)),
        keyframe_min_rotation_degrees=float(mapping.get("keyframe_min_rotation_degrees", 2.0)),
        keyframe_support_ratio_trigger=float(mapping.get("keyframe_support_ratio_trigger", 0.5)),
        keyframe_critical_support_ratio=float(mapping.get("keyframe_critical_support_ratio", 0.1)),
        keyframe_critical_min_translation_arbitrary=float(mapping.get("keyframe_critical_min_translation_arbitrary", 0.3)),
        keyframe_critical_min_frame_gap=int(mapping.get("keyframe_critical_min_frame_gap", 2)),
        mapping_min_matches=int(mapping.get("min_matches", 20)),
        mapping_epipolar_threshold_px=float(mapping.get("epipolar_threshold_px", 1.5)),
        mapping_max_reprojection_error_px=float(mapping.get("max_reprojection_error_px", 2.0)),
        mapping_min_triangulation_angle_degrees=float(mapping.get("min_triangulation_angle_degrees", 1.0)),
        mapping_max_depth_arbitrary=float(mapping.get("max_depth_arbitrary", 100.0)),
        mapping_min_new_landmarks=int(mapping.get("min_new_landmarks", 12)),
        mapping_min_occupied_grid_cells=int(mapping.get("min_occupied_grid_cells", 6)),
        mapping_duplicate_radius_px=float(mapping.get("duplicate_radius_px", 3.0)),
        mapping_max_landmarks=int(mapping.get("max_landmarks", 1000)),
        mapping_max_new_landmarks_per_keyframe=int(mapping.get("max_new_landmarks_per_keyframe", 1000)),
        mapping_max_new_landmarks_per_grid_cell=int(mapping.get("max_new_landmarks_per_grid_cell", 1000)),
        mapping_efficient_candidate_reverse=bool(mapping.get("efficient_candidate_reverse", False)),
    )
    if not 0.0 < app.ratio_test < 1.0:
        raise ValueError("features.ratio_test must be between 0 and 1")
    if app.max_features <= 0 or app.min_matches < 5 or app.min_inliers < 5:
        raise ValueError("max_features must be positive; min_matches/min_inliers must be at least 5")
    if app.ransac_threshold_px <= 0 or not 0.0 < app.ransac_probability < 1.0:
        raise ValueError("Invalid RANSAC threshold or probability")
    if app.save_match_every < 0:
        raise ValueError("output.save_match_every cannot be negative")
    if not 0.0 < app.rotation_only_residual_degrees < 5.0:
        raise ValueError("rotation_only_residual_degrees must be between 0 and 5")
    if not 0.0 < app.rotation_only_min_ratio <= 1.0:
        raise ValueError("rotation_only_min_ratio must be between 0 and 1")
    if app.rotation_innovation_degrees <= 0 or app.suspicious_parallax_p10_degrees <= 0:
        raise ValueError("rotation innovation and parallax thresholds must be positive")
    if app.rotation_history_size < 1:
        raise ValueError("rotation_history_size must be positive")
    if app.initialization_min_matches < 5 or app.initialization_min_inliers < 5:
        raise ValueError("initialization match/inlier minimums must be at least 5")
    if app.initialization_min_median_parallax_degrees <= 0 or app.initialization_min_p25_parallax_degrees <= 0:
        raise ValueError("initialization parallax thresholds must be positive")
    if app.initialization_grid_rows < 1 or app.initialization_grid_columns < 1:
        raise ValueError("initialization grid dimensions must be positive")
    if not 0.0 < app.initialization_min_grid_ratio <= 1.0:
        raise ValueError("initialization min_occupied_grid_ratio must be between 0 and 1")
    if app.triangulation_max_reprojection_error_px <= 0 or app.triangulation_min_angle_degrees <= 0:
        raise ValueError("triangulation reprojection and angle thresholds must be positive")
    if app.triangulation_max_depth <= 0:
        raise ValueError("triangulation max depth must be positive")
    if app.pnp_min_correspondences < 4 or app.pnp_min_inliers < 4:
        raise ValueError("PnP correspondence/inlier minimums must be at least 4")
    if not 0.0 < app.pnp_min_inlier_ratio <= 1.0:
        raise ValueError("PnP min_inlier_ratio must be between 0 and 1")
    if min(
        app.pnp_max_descriptor_distance,
        app.pnp_ransac_reprojection_error_px,
        app.pnp_max_median_reprojection_error_px,
        app.pnp_max_p95_reprojection_error_px,
        app.pnp_max_rotation_jump_degrees,
        app.pnp_max_translation_jump_arbitrary,
    ) <= 0:
        raise ValueError("PnP distance, error, and motion sanity thresholds must be positive")
    if app.pnp_min_occupied_grid_cells < 1:
        raise ValueError("PnP min_occupied_grid_cells must be positive")
    if app.association_mode not in {"global", "projected"}:
        raise ValueError("tracking.association_mode must be global or projected")
    if app.descriptor_update_policy not in {"fixed", "latest", "bounded_recent"}:
        raise ValueError("tracking.descriptor_update_policy must be fixed, latest, or bounded_recent")
    if app.projection_search_radius_px <= 0 or app.descriptor_bank_max_size < 1:
        raise ValueError("projection search radius and descriptor bank size must be positive")
    if app.association_descriptor_reference_limit < 0:
        raise ValueError("association descriptor reference limit cannot be negative")
    if min(
        app.optical_flow_max_forward_backward_error_px,
        app.optical_flow_max_lk_error,
    ) <= 0:
        raise ValueError("optical-flow error thresholds must be positive")
    if app.landmark_max_active < 4 or app.landmark_max_visible_misses < 1:
        raise ValueError("landmark active budget must be at least four and miss limit positive")
    if app.keyframe_min_frame_gap < 1 or app.keyframe_max_frame_gap < app.keyframe_min_frame_gap:
        raise ValueError("mapping keyframe frame-gap limits are invalid")
    if not 1 <= app.keyframe_critical_min_frame_gap <= app.keyframe_min_frame_gap:
        raise ValueError("mapping critical frame gap must be between one and the normal minimum")
    if not 0.0 < app.keyframe_support_ratio_trigger <= 1.0:
        raise ValueError("mapping keyframe_support_ratio_trigger must be between 0 and 1")
    if not 0.0 < app.keyframe_critical_support_ratio <= app.keyframe_support_ratio_trigger:
        raise ValueError("mapping critical support ratio must be positive and no larger than the normal trigger")
    if min(
        app.keyframe_min_translation_arbitrary,
        app.keyframe_critical_min_translation_arbitrary,
        app.keyframe_min_rotation_degrees,
        app.mapping_epipolar_threshold_px,
        app.mapping_max_reprojection_error_px,
        app.mapping_min_triangulation_angle_degrees,
        app.mapping_max_depth_arbitrary,
        app.mapping_duplicate_radius_px,
    ) <= 0:
        raise ValueError("mapping geometry thresholds must be positive")
    if app.mapping_min_matches < 5 or app.mapping_min_new_landmarks < 1:
        raise ValueError("mapping match and new-landmark minimums are invalid")
    if app.mapping_min_occupied_grid_cells < 1 or app.mapping_max_landmarks < 1:
        raise ValueError("mapping grid and landmark limits must be positive")
    if app.mapping_max_new_landmarks_per_keyframe < 1 or app.mapping_max_new_landmarks_per_grid_cell < 1:
        raise ValueError("mapping landmark insertion budgets must be positive")
    return app
