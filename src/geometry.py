"""Calibrated two-view geometry."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class PoseEstimate:
    success: bool
    rotation: np.ndarray | None
    translation: np.ndarray | None
    inlier_mask: np.ndarray
    inlier_count: int
    reason: str = ""


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
        return PoseEstimate(False, None, None, empty_mask, 0, "insufficient or mismatched points")
    if not np.all(np.isfinite(points_previous)) or not np.all(np.isfinite(points_current)):
        return PoseEstimate(False, None, None, empty_mask, 0, "non-finite feature coordinates")
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
            return PoseEstimate(False, None, None, empty_mask, 0, "essential matrix estimation failed")
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
            return PoseEstimate(False, None, None, mask, pose_count, f"only {pose_count} geometric inliers")
        if not np.all(np.isfinite(rotation)) or not np.all(np.isfinite(translation)):
            return PoseEstimate(False, None, None, mask, pose_count, "non-finite pose")
        determinant = float(np.linalg.det(rotation))
        norm = float(np.linalg.norm(translation))
        if abs(determinant - 1.0) > 1e-3 or norm < 1e-9:
            return PoseEstimate(False, None, None, mask, pose_count, "invalid rotation or translation")
        return PoseEstimate(True, rotation, translation.reshape(3) / norm, mask, pose_count)
    except cv2.error as exc:
        return PoseEstimate(False, None, None, empty_mask, 0, f"OpenCV geometry error: {exc}")

