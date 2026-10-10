import numpy as np

from src.geometry import PoseEstimate
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


def test_failure_does_not_change_pose_or_trajectory() -> None:
    odometry = VisualOdometry()
    before = odometry.pose_camera_to_world.copy()
    failed = PoseEstimate(False, None, None, np.zeros(0, dtype=bool), 0, "test failure")
    assert not odometry.update(failed)
    assert np.array_equal(odometry.pose_camera_to_world, before)
    assert len(odometry.trajectory) == 1
    assert odometry.failures == 1
    assert not odometry.tracking_reliable

