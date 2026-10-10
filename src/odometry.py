"""Camera-to-world pose accumulation and tracking state."""

from __future__ import annotations

import numpy as np

from .geometry import PoseEstimate


class VisualOdometry:
    """Accumulate camera poses using one arbitrary translation unit per valid pair.

    recoverPose returns X_current = R X_previous + t. If T_wc is a
    camera-to-world transform, T_wc_current = T_wc_previous @ inv(T_current_previous).
    Thus R_wc' = R_wc @ R.T and p_wc' = p_wc - R_wc @ R.T @ t.
    """

    def __init__(self) -> None:
        self.pose_camera_to_world = np.eye(4, dtype=np.float64)
        self.trajectory: list[np.ndarray] = [np.zeros(3, dtype=np.float64)]
        self.successes = 0
        self.failures = 0
        self.tracking_reliable = True

    def update(self, estimate: PoseEstimate) -> bool:
        if not estimate.success or estimate.rotation is None or estimate.translation is None:
            self.mark_failure()
            return False
        relative = np.eye(4, dtype=np.float64)
        relative[:3, :3] = estimate.rotation
        relative[:3, 3] = estimate.translation
        candidate = self.pose_camera_to_world @ np.linalg.inv(relative)
        if not np.all(np.isfinite(candidate)):
            self.mark_failure()
            return False
        self.pose_camera_to_world = candidate
        self.trajectory.append(candidate[:3, 3].copy())
        self.successes += 1
        self.tracking_reliable = True
        return True

    def mark_failure(self) -> None:
        self.failures += 1
        self.tracking_reliable = False

