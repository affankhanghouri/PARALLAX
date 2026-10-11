import cv2
import numpy as np
import pytest

from src.reconstruction import TriangulationThresholds, triangulate_two_view


CAMERA = np.array([[700.0, 0, 320], [0, 700.0, 240], [0, 0, 1]])


def _project(points: np.ndarray, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    camera_points = (rotation @ points.T).T + translation
    pixels = (CAMERA @ camera_points.T).T
    return pixels[:, :2] / pixels[:, 2:]


def _scene(seed: int = 20, count: int = 300, planar: bool = False) -> np.ndarray:
    rng = np.random.default_rng(seed)
    depth = np.full(count, 8.0) if planar else rng.uniform(4.0, 15.0, count)
    return np.column_stack([rng.uniform(-3, 3, count), rng.uniform(-2, 2, count), depth])


def test_normal_scene_recovers_unit_baseline_geometry() -> None:
    points = _scene()
    rotation, _ = cv2.Rodrigues(np.array([0.01, -0.025, 0.004]))
    translation = np.array([0.8, 0.05, 0.1])
    first = _project(points, np.eye(3), np.zeros(3))
    second = _project(points, rotation, translation)

    result = triangulate_two_view(first, second, CAMERA, rotation, translation)

    scale = 1.0 / np.linalg.norm(translation)
    assert result.stats.accepted > 280
    assert np.allclose(result.points_first_camera[result.accepted_mask], points[result.accepted_mask] * scale, atol=1e-5)
    assert np.isclose(np.linalg.norm(result.camera_center_second), 1.0)
    assert result.summary()["reprojection_first_px"]["max"] < 1e-5


def test_small_baseline_is_rejected_by_triangulation_angle() -> None:
    points = _scene(seed=21)
    rotation = np.eye(3)
    translation = np.array([0.005, 0.0, 0.0])
    first = _project(points, rotation, np.zeros(3))
    second = _project(points, rotation, translation)

    result = triangulate_two_view(first, second, CAMERA, rotation, translation)

    assert result.stats.reprojection_consistent == len(points)
    assert result.stats.sufficient_angle == 0
    assert result.stats.accepted == 0


def test_pure_rotation_cannot_be_triangulated() -> None:
    points = _scene(seed=22)
    rotation, _ = cv2.Rodrigues(np.array([0.0, 0.05, 0.0]))
    first = _project(points, np.eye(3), np.zeros(3))
    second = _project(points, rotation, np.zeros(3))
    with pytest.raises(ValueError, match="nonzero translation"):
        triangulate_two_view(first, second, CAMERA, rotation, np.zeros(3))


def test_noisy_correspondences_retain_well_conditioned_majority() -> None:
    rng = np.random.default_rng(23)
    points = _scene(seed=23)
    rotation, _ = cv2.Rodrigues(np.array([0.005, -0.02, 0.003]))
    translation = np.array([0.9, 0.04, 0.1])
    first = _project(points, np.eye(3), np.zeros(3)) + rng.normal(0, 0.35, (len(points), 2))
    second = _project(points, rotation, translation) + rng.normal(0, 0.35, (len(points), 2))

    result = triangulate_two_view(first, second, CAMERA, rotation, translation)

    assert result.stats.accepted > 250
    accepted_errors = result.reprojection_error_first_px[result.accepted_mask]
    assert np.percentile(accepted_errors, 95) < 1.0


def test_planar_scene_is_valid_when_baseline_is_observable() -> None:
    points = _scene(seed=24, planar=True)
    rotation, _ = cv2.Rodrigues(np.array([0.004, -0.015, 0.002]))
    translation = np.array([0.8, 0.03, 0.1])
    result = triangulate_two_view(
        _project(points, np.eye(3), np.zeros(3)),
        _project(points, rotation, translation),
        CAMERA, rotation, translation,
    )
    assert result.stats.accepted > 290


def test_incorrect_correspondences_are_mostly_rejected() -> None:
    points = _scene(seed=25)
    rotation, _ = cv2.Rodrigues(np.array([0.004, -0.02, 0.003]))
    translation = np.array([0.8, 0.02, 0.1])
    first = _project(points, np.eye(3), np.zeros(3))
    second = np.roll(_project(points, rotation, translation), 37, axis=0)
    result = triangulate_two_view(first, second, CAMERA, rotation, translation)
    assert result.stats.accepted < len(points) * 0.15


def test_nonfinite_correspondence_is_counted_and_rejected() -> None:
    points = _scene(seed=26, count=20)
    rotation = np.eye(3)
    translation = np.array([1.0, 0.0, 0.0])
    first = _project(points, rotation, np.zeros(3))
    second = _project(points, rotation, translation)
    first[0] = np.nan
    result = triangulate_two_view(first, second, CAMERA, rotation, translation)
    assert result.stats.candidates == 20
    assert result.stats.finite_homogeneous == 19
    assert not result.accepted_mask[0]


def test_max_depth_filters_extreme_unit_baseline_depth() -> None:
    points = _scene(seed=27)
    rotation = np.eye(3)
    translation = np.array([1.0, 0.0, 0.0])
    result = triangulate_two_view(
        _project(points, rotation, np.zeros(3)),
        _project(points, rotation, translation),
        CAMERA, rotation, translation,
        TriangulationThresholds(max_depth=6.0, min_triangulation_angle_degrees=0.1),
    )
    assert result.stats.bounded_depth < result.stats.sufficient_angle
