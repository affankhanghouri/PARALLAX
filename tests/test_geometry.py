import cv2
import numpy as np

from src.geometry import MotionState, estimate_relative_pose, estimate_reliable_motion


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


def _reliable_estimate(points: np.ndarray, rotation_vector: np.ndarray, translation: np.ndarray):
    camera = np.array([[700.0, 0, 320], [0, 700.0, 240], [0, 0, 1]])
    rotation, _ = cv2.Rodrigues(rotation_vector.astype(float))
    previous = _project(points, np.eye(3), np.zeros(3), camera)
    current = _project(points, rotation, translation, camera)
    return estimate_reliable_motion(
        previous, current, camera, recent_rotations=[], min_inliers=30,
        ransac_threshold_px=0.5,
    )


def test_pure_rotation_is_explicitly_rotation_only() -> None:
    rng = np.random.default_rng(10)
    points = np.column_stack([
        rng.uniform(-3, 3, 200), rng.uniform(-2, 2, 200), rng.uniform(6, 20, 200)
    ])
    estimate = _reliable_estimate(points, np.array([0.01, -0.035, 0.005]), np.zeros(3))
    assert estimate.success
    assert estimate.state == MotionState.ROTATION_ONLY
    assert estimate.translation is None


def test_very_small_translation_does_not_create_unit_step() -> None:
    rng = np.random.default_rng(11)
    points = np.column_stack([
        rng.uniform(-3, 3, 200), rng.uniform(-2, 2, 200), rng.uniform(8, 25, 200)
    ])
    estimate = _reliable_estimate(points, np.array([0.005, -0.02, 0.002]), np.array([0.002, 0, 0.001]))
    assert estimate.state == MotionState.ROTATION_ONLY
    assert estimate.translation is None


def test_normal_translation_with_depth_variation_is_valid_motion() -> None:
    rng = np.random.default_rng(12)
    points = np.column_stack([
        rng.uniform(-3, 3, 300), rng.uniform(-2, 2, 300), rng.uniform(5, 25, 300)
    ])
    estimate = _reliable_estimate(points, np.array([0.005, -0.02, 0.002]), np.array([0.7, 0.03, 0.1]))
    assert estimate.success, estimate.reason
    assert estimate.state == MotionState.VALID_MOTION
    assert estimate.translation is not None


def test_planar_scene_is_not_called_rotation_only_from_homography_alone() -> None:
    rng = np.random.default_rng(13)
    camera = np.array([[700.0, 0, 320], [0, 700.0, 240], [0, 0, 1]])
    points = np.column_stack([
        rng.uniform(-3, 3, 300), rng.uniform(-2, 2, 300), np.full(300, 8.0)
    ])
    rotation, _ = cv2.Rodrigues(np.array([0.005, -0.02, 0.002]))
    previous = _project(points, np.eye(3), np.zeros(3), camera)
    current = _project(points, rotation, np.array([0.8, 0.04, 0.1]), camera)
    _, homography_mask = cv2.findHomography(previous, current, cv2.RANSAC, 0.5)
    estimate = estimate_reliable_motion(
        previous, current, camera, recent_rotations=[], min_inliers=30,
        ransac_threshold_px=0.5,
    )
    assert estimate.state != MotionState.ROTATION_ONLY
    assert np.count_nonzero(homography_mask) / len(points) > 0.9
