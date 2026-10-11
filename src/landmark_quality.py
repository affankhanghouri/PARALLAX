"""Runtime landmark quality evidence and reversible active-set selection."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .local_map import Landmark, LandmarkStatus, MapSegment
from .map_tracking import TrackingResult


@dataclass(frozen=True)
class QualityUpdate:
    eligible: int
    predicted_visible: int
    selected_active: int
    temporarily_inactive: int
    deactivated_for_budget: int
    deactivated_for_misses: int
    reactivated: int


def _project_visible(
    landmarks: list[Landmark], pose_camera_to_segment: np.ndarray,
    camera_matrix: np.ndarray, image_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    if not landmarks:
        return np.empty((0, 2)), np.empty(0, dtype=bool)
    world_to_camera = np.linalg.inv(pose_camera_to_segment)
    points = np.asarray([item.position for item in landmarks], dtype=np.float64)
    camera = (world_to_camera[:3, :3] @ points.T).T + world_to_camera[:3, 3]
    homogeneous = (camera_matrix @ camera.T).T
    pixels = np.full((len(points), 2), np.nan)
    finite = np.isfinite(homogeneous).all(axis=1) & (np.abs(homogeneous[:, 2]) > 1e-12)
    pixels[finite] = homogeneous[finite, :2] / homogeneous[finite, 2:3]
    height, width = image_shape[:2]
    visible = (
        finite & (camera[:, 2] > 0) & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    return pixels, visible


def _score(landmark: Landmark) -> tuple[float, float, float, int]:
    """Rank by validated runtime evidence, then static geometric quality."""
    attempts = max(landmark.projection_attempts, 1)
    inlier_rate = landmark.pnp_inlier_observations / attempts
    association_rate = landmark.association_successes / attempts
    error = landmark.reprojection_error_ema_px
    error_score = 1.0 / (1.0 + (error if error is not None else 2.0))
    return (
        3.0 * inlier_rate + association_rate + error_score,
        float(landmark.quality.triangulation_angle_degrees),
        -float(landmark.consecutive_visible_misses),
        landmark.last_observed_frame_index,
    )


def select_active_landmarks(
    segment: MapSegment, pose_camera_to_segment: np.ndarray,
    camera_matrix: np.ndarray, image_shape: tuple[int, ...], *,
    max_active: int, max_visible_misses: int, grid_rows: int = 4,
    grid_columns: int = 6,
) -> tuple[QualityUpdate, set[str]]:
    """Select a bounded, spatially balanced active set without deleting landmarks.

    Visibility uses only the previous accepted pose. A landmark is inactive for
    this frame when outside the predicted view, repeatedly missed, or displaced
    by the active budget. All non-invalid landmarks are reconsidered every frame.
    """
    eligible = [item for item in segment.landmarks.values() if item.status != LandmarkStatus.INVALID]
    old_active = {item.landmark_id for item in eligible if item.status == LandmarkStatus.ACTIVE}
    pixels, visible = _project_visible(
        eligible, pose_camera_to_segment, camera_matrix, image_shape,
    )
    visible_indices = [index for index in np.flatnonzero(visible)]
    miss_filtered = [
        index for index in visible_indices
        if eligible[index].consecutive_visible_misses < max_visible_misses
    ]
    height, width = image_shape[:2]
    cells: dict[tuple[int, int], list[int]] = {}
    for index in miss_filtered:
        x = min(grid_columns - 1, max(0, int(pixels[index, 0] / width * grid_columns)))
        y = min(grid_rows - 1, max(0, int(pixels[index, 1] / height * grid_rows)))
        cells.setdefault((y, x), []).append(index)
    for indices in cells.values():
        indices.sort(key=lambda index: _score(eligible[index]), reverse=True)
    selected: list[int] = []
    while len(selected) < max_active and any(cells.values()):
        for cell in sorted(cells):
            if cells[cell] and len(selected) < max_active:
                selected.append(cells[cell].pop(0))
    selected_ids = {eligible[index].landmark_id for index in selected}
    for landmark in eligible:
        landmark.status = (
            LandmarkStatus.ACTIVE
            if landmark.landmark_id in selected_ids
            else LandmarkStatus.TEMPORARILY_INACTIVE
        )
    new_active = selected_ids
    missed = len(visible_indices) - len(miss_filtered)
    budget = max(0, len(miss_filtered) - len(selected))
    update = QualityUpdate(
        len(eligible), len(visible_indices), len(selected), len(eligible) - len(selected),
        budget, missed, len(new_active - old_active),
    )
    return update, {eligible[index].landmark_id for index in visible_indices}


def update_landmark_quality(
    segment: MapSegment, result: TrackingResult, visible_landmark_ids: set[str],
    source_frame_index: int,
) -> None:
    """Update evidence only after pose estimation; no ground truth is involved."""
    associated = {item.landmark_id for item in result.associations}
    inlier_errors = {
        result.associations[int(association_index)].landmark_id:
        float(result.inlier_reprojection_errors_px[error_index])
        for error_index, association_index in enumerate(result.inlier_association_indices)
    }
    for landmark_id in visible_landmark_ids:
        landmark = segment.landmarks[landmark_id]
        landmark.projection_attempts += 1
        if landmark_id in associated:
            landmark.association_successes += 1
        if result.accepted and landmark_id in inlier_errors:
            landmark.pnp_inlier_observations += 1
            landmark.consecutive_visible_misses = 0
            landmark.last_observed_frame_index = source_frame_index
            error = inlier_errors[landmark_id]
            landmark.reprojection_error_ema_px = (
                error if landmark.reprojection_error_ema_px is None
                else 0.8 * landmark.reprojection_error_ema_px + 0.2 * error
            )
        else:
            landmark.consecutive_visible_misses += 1
