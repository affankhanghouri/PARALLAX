"""Camera-to-world pose accumulation and tracking state."""

from __future__ import annotations

import numpy as np

from .geometry import MotionState, PoseEstimate


class VisualOdometry:
    """Accumulate camera poses using one arbitrary translation unit per valid pair.

    recoverPose returns X_current = R X_previous + t. If T_wc is a
    camera-to-world transform, T_wc_current = T_wc_previous @ inv(T_current_previous).
    Thus R_wc' = R_wc @ R.T and p_wc' = p_wc - R_wc @ R.T @ t.
    """

    def __init__(self) -> None:
        self.pose_camera_to_world = np.eye(4, dtype=np.float64)
        self.trajectory: list[np.ndarray] = [np.zeros(3, dtype=np.float64)]
        self.poses_camera_to_world: list[np.ndarray] = [self.pose_camera_to_world.copy()]
        self.successes = 0
        self.failures = 0
        self.tracking_reliable = True
        self.state_counts = {state.value: 0 for state in MotionState}

    def update(self, estimate: PoseEstimate) -> bool:
        self.state_counts[estimate.state.value] += 1
        if not estimate.success or estimate.rotation is None:
            self.mark_failure()
            return False
        relative = np.eye(4, dtype=np.float64)
        relative[:3, :3] = estimate.rotation
        if estimate.state == MotionState.VALID_MOTION and estimate.translation is not None:
            relative[:3, 3] = estimate.translation
        elif estimate.state == MotionState.ROTATION_ONLY:
            # Orientation is observable; position is held rather than inventing
            # the baseline engine's arbitrary one-unit translation.
            self.tracking_reliable = False
        else:
            self.mark_failure()
            return False
        candidate = self.pose_camera_to_world @ np.linalg.inv(relative)
        if not np.all(np.isfinite(candidate)):
            self.mark_failure()
            return False
        self.pose_camera_to_world = candidate
        self.trajectory.append(candidate[:3, 3].copy())
        self.poses_camera_to_world.append(candidate.copy())
        self.successes += 1
        return True

    def mark_failure(self) -> None:
        self.failures += 1
        self.tracking_reliable = False
