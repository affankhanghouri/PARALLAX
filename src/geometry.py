"""Calibrated two-view geometry."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np


class MotionState(str, Enum):
    VALID_MOTION = "VALID_MOTION"
    ROTATION_ONLY = "ROTATION_ONLY"
    DEGENERATE = "DEGENERATE"
    TRACKING_LOST = "TRACKING_LOST"


@dataclass(frozen=True)
class GeometryDiagnostics:
    essential_inlier_ratio: float
    homography_inlier_ratio: float | None
    compensated_parallax_p10_degrees: float | None
    compensated_parallax_median_degrees: float | None
    rotation_only_inlier_count: int
    rotation_only_inlier_ratio: float
    rotation_only_median_residual_degrees: float | None
    rotation_model_disagreement_degrees: float | None
    rotation_innovation_degrees: float | None


@dataclass(frozen=True)
class PoseEstimate:
    success: bool
    rotation: np.ndarray | None
    translation: np.ndarray | None
    inlier_mask: np.ndarray
    inlier_count: int
    reason: str = ""
    state: MotionState = MotionState.VALID_MOTION
    diagnostics: GeometryDiagnostics | None = None


def _rotation_angle_degrees(rotation: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))


def _project_to_rotation(matrix: np.ndarray) -> np.ndarray:
    left, _, right = np.linalg.svd(matrix)
    rotation = left @ right
    if np.linalg.det(rotation) < 0:
        left[:, -1] *= -1
        rotation = left @ right
    return rotation


def _unit_bearings(points: np.ndarray, camera_matrix: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    rays = (np.linalg.inv(camera_matrix) @ homogeneous.T).T
    return rays / np.linalg.norm(rays, axis=1, keepdims=True)


def _bearing_residuals_degrees(rotation: np.ndarray, previous: np.ndarray, current: np.ndarray) -> np.ndarray:
    rotated = (rotation @ previous.T).T
    return np.degrees(np.arccos(np.clip(np.sum(rotated * current, axis=1), -1.0, 1.0)))


def _rotation_only_model(
    points_previous: np.ndarray,
    points_current: np.ndarray,
    camera_matrix: np.ndarray,
    residual_threshold_degrees: float,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, float]:
    """Fit a constrained bearing rotation, initialized by a robust homography.

    Homography support is not used as proof of pure rotation. Its only role is to
    initialize SO(3); acceptance uses angular residuals of the constrained model.
    """
    count = len(points_previous)
    empty = np.zeros(count, dtype=bool)
    homography, homography_mask = cv2.findHomography(
        points_previous, points_current, cv2.RANSAC, 1.0
    )
    if homography is None or homography_mask is None or not np.all(np.isfinite(homography)):
        return None, empty, empty, float("inf")
    calibrated = np.linalg.inv(camera_matrix) @ homography @ camera_matrix
    rotation = _project_to_rotation(calibrated)
    previous_bearings = _unit_bearings(points_previous, camera_matrix)
    current_bearings = _unit_bearings(points_current, camera_matrix)
    inliers = empty
    for _ in range(5):
        residuals = _bearing_residuals_degrees(rotation, previous_bearings, current_bearings)
        updated = residuals < residual_threshold_degrees
        if np.count_nonzero(updated) < 3:
            break
        rotation = _project_to_rotation(current_bearings[updated].T @ previous_bearings[updated])
        if np.array_equal(updated, inliers):
            break
        inliers = updated
    residuals = _bearing_residuals_degrees(rotation, previous_bearings, current_bearings)
    inliers = residuals < residual_threshold_degrees
    median = float(np.median(residuals[inliers])) if np.any(inliers) else float("inf")
    return rotation, inliers, homography_mask.reshape(-1).astype(bool), median


def _median_recent_rotation(recent_rotations: list[np.ndarray]) -> np.ndarray | None:
    if not recent_rotations:
        return None
    vectors = [cv2.Rodrigues(rotation)[0].reshape(3) for rotation in recent_rotations]
    vector = np.median(np.asarray(vectors), axis=0)
    return cv2.Rodrigues(vector)[0]


def estimate_relative_pose(
    points_previous: np.ndarray,
    points_current: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    min_inliers: int = 15,
    ransac_probability: float = 0.999,
    ransac_threshold_px: float = 1.0,
) -> PoseEstimate:
    """Estimate X_current = R X_previous + t with calibrated epipolar geometry."""
    count = len(points_previous)
    empty_mask = np.zeros(count, dtype=bool)
    if count < 5 or len(points_current) != count:
        return PoseEstimate(False, None, None, empty_mask, 0, "insufficient or mismatched points", MotionState.TRACKING_LOST)
    if not np.all(np.isfinite(points_previous)) or not np.all(np.isfinite(points_current)):
        return PoseEstimate(False, None, None, empty_mask, 0, "non-finite feature coordinates", MotionState.TRACKING_LOST)
    try:
        essential, ransac_mask = cv2.findEssentialMat(
            points_previous,
            points_current,
            camera_matrix,
            method=cv2.RANSAC,
            prob=ransac_probability,
            threshold=ransac_threshold_px,
        )
        if essential is None or ransac_mask is None:
            return PoseEstimate(False, None, None, empty_mask, 0, "essential matrix estimation failed", MotionState.DEGENERATE)
        # findEssentialMat may stack several 3x3 candidates. Test each candidate and
        # retain the one with the most cheirality-consistent recoverPose inliers.
        candidates = [essential[i : i + 3] for i in range(0, essential.shape[0], 3)]
        best: tuple[int, np.ndarray, np.ndarray, np.ndarray] | None = None
        for candidate in candidates:
            pose_count, rotation, translation, pose_mask = cv2.recoverPose(
                candidate,
                points_previous,
                points_current,
                camera_matrix,
                mask=ransac_mask.copy(),
            )
            if best is None or pose_count > best[0]:
                best = (int(pose_count), rotation, translation, pose_mask)
        assert best is not None
        pose_count, rotation, translation, pose_mask = best
        mask = pose_mask.reshape(-1).astype(bool)
        if pose_count < min_inliers:
            return PoseEstimate(False, None, None, mask, pose_count, f"only {pose_count} geometric inliers", MotionState.DEGENERATE)
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            return PoseEstimate(False, None, None, mask, pose_count, "non-finite pose", MotionState.DEGENERATE)
        determinant = float(np.linalg.det(rotation))
        norm = float(np.linalg.norm(translation))
        if abs(determinant - 1.0) > 1e-3 or norm < 1e-9:
            return PoseEstimate(False, None, None, mask, pose_count, "invalid rotation or translation", MotionState.DEGENERATE)
        return PoseEstimate(True, rotation, translation.reshape(3) / norm, mask, pose_count)
    except cv2.error as exc:
        return PoseEstimate(False, None, None, empty_mask, 0, f"OpenCV geometry error: {exc}", MotionState.DEGENERATE)


def estimate_reliable_motion(
    points_previous: np.ndarray,
    points_current: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    recent_rotations: list[np.ndarray],
    min_inliers: int = 15,
    ransac_probability: float = 0.999,
    ransac_threshold_px: float = 1.0,
    rotation_only_residual_degrees: float = 0.2,
    rotation_only_min_ratio: float = 0.6,
    rotation_innovation_degrees: float = 2.0,
    suspicious_parallax_p10_degrees: float = 1.0,
) -> PoseEstimate:
    """Estimate motion and reject two-view geometry that lacks observability.

    The decision uses image measurements only. Ground truth is deliberately not
    accepted by this API and is confined to offline evaluation.
    """
    baseline = estimate_relative_pose(
        points_previous, points_current, camera_matrix,
        min_inliers=min_inliers, ransac_probability=ransac_probability,
        ransac_threshold_px=ransac_threshold_px,
    )
    count = len(points_previous)
    if count < 5:
        return baseline
    essential_ratio = baseline.inlier_count / count
    p10 = median_parallax = disagreement = innovation = None
    if baseline.rotation is not None and np.any(baseline.inlier_mask):
        previous_bearings = _unit_bearings(points_previous[baseline.inlier_mask], camera_matrix)
        current_bearings = _unit_bearings(points_current[baseline.inlier_mask], camera_matrix)
        residuals = _bearing_residuals_degrees(baseline.rotation, previous_bearings, current_bearings)
        p10 = float(np.percentile(residuals, 10))
        median_parallax = float(np.median(residuals))
    recent = _median_recent_rotation(recent_rotations)
    if baseline.rotation is not None and recent is not None:
        innovation = _rotation_angle_degrees(recent.T @ baseline.rotation)
    questionable = (
        innovation is not None
        and innovation > rotation_innovation_degrees
        and p10 is not None
        and p10 > suspicious_parallax_p10_degrees
    )
    rotation_only = None
    rotation_mask = np.zeros(count, dtype=bool)
    homography_ratio = None
    rotation_residual = None
    try:
        rotation_only, rotation_mask, homography_mask, model_residual = _rotation_only_model(
            points_previous, points_current, camera_matrix, rotation_only_residual_degrees
        )
        homography_ratio = float(np.count_nonzero(homography_mask) / count)
        rotation_residual = model_residual if np.isfinite(model_residual) else None
    except (cv2.error, np.linalg.LinAlgError):
        pass
    rotation_count = int(np.count_nonzero(rotation_mask))
    rotation_ratio = rotation_count / count
    if baseline.rotation is not None and rotation_only is not None:
        disagreement = _rotation_angle_degrees(rotation_only.T @ baseline.rotation)
    diagnostics = GeometryDiagnostics(
        essential_ratio, homography_ratio, p10, median_parallax,
        rotation_count, rotation_ratio, rotation_residual, disagreement, innovation,
    )

    # A constrained SO(3) model explaining most correspondences means image
    # translation is below the chosen angular observability floor. Preserve the
    # supported rotation, but never manufacture a unit translation.
    if rotation_only is not None and rotation_ratio >= rotation_only_min_ratio:
        return PoseEstimate(
            True, rotation_only, None, rotation_mask, rotation_count,
            f"translation unobservable: rotation-only consensus {rotation_count}/{count}",
            MotionState.ROTATION_ONLY, diagnostics,
        )

    if not baseline.success:
        return PoseEstimate(
            False, baseline.rotation, baseline.translation, baseline.inlier_mask,
            baseline.inlier_count, baseline.reason, baseline.state, diagnostics,
        )

    # Large temporal rotation innovations are only rejected when the essential
    # rotation also leaves uniformly large compensated bearing residuals. This
    # conjunction avoids treating ordinary acceleration as degeneracy.
    if questionable:
        return PoseEstimate(
            False, baseline.rotation, baseline.translation, baseline.inlier_mask,
            baseline.inlier_count,
            f"questionable rotation: {innovation:.3f} deg innovation and {p10:.3f} deg p10 compensated parallax",
            MotionState.DEGENERATE, diagnostics,
        )
    return PoseEstimate(
        True, baseline.rotation, baseline.translation, baseline.inlier_mask,
        baseline.inlier_count, "", MotionState.VALID_MOTION, diagnostics,
    )
