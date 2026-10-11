import cv2
import numpy as np

from src.geometry import MotionState, PoseEstimate
from src.odometry import VisualOdometry


def _success(rotation: np.ndarray, translation: np.ndarray) -> PoseEstimate:
    return PoseEstimate(True, rotation, translation, np.ones(20, dtype=bool), 20)


def test_camera_to_world_pose_composition() -> None:
    odometry = VisualOdometry()
    angle = np.pi / 2
    rotation = np.array([[np.cos(angle), 0, np.sin(angle)], [0, 1, 0], [-np.sin(angle), 0, np.cos(angle)]])
    assert odometry.update(_success(rotation, np.array([0.0, 0.0, 1.0])))

    expected = np.eye(4)
    relative = np.eye(4)
    relative[:3, :3] = rotation
    relative[:3, 3] = [0, 0, 1]
    expected = expected @ np.linalg.inv(relative)
    assert np.allclose(odometry.pose_camera_to_world, expected)
    assert np.allclose(odometry.trajectory[-1], expected[:3, 3])
    assert np.allclose(odometry.poses_camera_to_world[-1], expected)


def test_failure_does_not_change_pose_or_trajectory() -> None:
    odometry = VisualOdometry()
    before = odometry.pose_camera_to_world.copy()
    failed = PoseEstimate(False, None, None, np.zeros(0, dtype=bool), 0, "test failure", MotionState.TRACKING_LOST)
    assert not odometry.update(failed)
    assert np.array_equal(odometry.pose_camera_to_world, before)
    assert len(odometry.trajectory) == 1
    assert len(odometry.poses_camera_to_world) == 1
    assert odometry.failures == 1
    assert not odometry.tracking_reliable


def test_tracking_reliability_does_not_silently_recover() -> None:
    odometry = VisualOdometry()
    failed = PoseEstimate(False, None, None, np.zeros(0, dtype=bool), 0, "lost", MotionState.TRACKING_LOST)
    odometry.update(failed)
    odometry.update(_success(np.eye(3), np.array([0.0, 0.0, 1.0])))
    assert odometry.successes == 1
    assert not odometry.tracking_reliable


def test_rotation_only_updates_orientation_without_fictional_translation() -> None:
    odometry = VisualOdometry()
    rotation, _ = cv2.Rodrigues(np.array([0.0, 0.1, 0.0]))
    estimate = PoseEstimate(
        True, rotation, None, np.ones(30, dtype=bool), 30,
        "translation unobservable", MotionState.ROTATION_ONLY,
    )
    assert odometry.update(estimate)
    assert np.allclose(odometry.trajectory[-1], np.zeros(3))
    assert np.allclose(odometry.pose_camera_to_world[:3, :3], rotation.T)
    assert not odometry.tracking_reliable
