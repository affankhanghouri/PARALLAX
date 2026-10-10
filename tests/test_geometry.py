import cv2
import numpy as np

from src.geometry import estimate_relative_pose


def _project(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray, camera: np.ndarray) -> np.ndarray:
    camera_points = (rotation @ points.T).T + translation
    pixels = (camera @ camera_points.T).T
    return pixels[:, :2] / pixels[:, 2:]


def test_relative_pose_from_synthetic_calibrated_points() -> None:
    rng = np.random.default_rng(42)
    camera = np.array([[700.0, 0, 320], [0, 700.0, 240], [0, 0, 1]])
    points = np.column_stack(
        [rng.uniform(-2, 2, 200), rng.uniform(-1.5, 1.5, 200), rng.uniform(5, 12, 200)]
    )
    rotation_vector = np.array([0.015, -0.04, 0.01])
    true_rotation, _ = cv2.Rodrigues(rotation_vector)
    true_translation = np.array([0.7, 0.08, 0.05])
    previous = _project(points, np.eye(3), np.zeros(3), camera)
    current = _project(points, true_rotation, true_translation, camera)

    estimate = estimate_relative_pose(previous, current, camera, min_inliers=100, ransac_threshold_px=0.5)

    assert estimate.success, estimate.reason
    assert estimate.inlier_count >= 100
    rotation_error = estimate.rotation @ true_rotation.T
    angle = np.arccos(np.clip((np.trace(rotation_error) - 1) / 2, -1, 1))
    assert angle < 0.01
    direction_cosine = float(np.dot(estimate.translation, true_translation / np.linalg.norm(true_translation)))
    assert direction_cosine > 0.99


def test_relative_pose_rejects_insufficient_points() -> None:
    estimate = estimate_relative_pose(np.zeros((4, 2)), np.zeros((4, 2)), np.eye(3))
    assert not estimate.success
    assert "insufficient" in estimate.reason

