"""Reliable two-view sparse reconstruction in an arbitrary-scale local frame."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class TriangulationThresholds:
    """Quality limits expressed in pixels, degrees, and unit-baseline depth."""

    max_reprojection_error_px: float = 2.0
    min_triangulation_angle_degrees: float = 1.0
    min_depth: float = 1e-6
    max_depth: float = 100.0
    homogeneous_epsilon: float = 1e-10


@dataclass(frozen=True)
class LandmarkFilterStats:
    candidates: int
    finite_homogeneous: int
    positive_depth_both: int
    reprojection_consistent: int
    sufficient_angle: int
    bounded_depth: int
    accepted: int


@dataclass(frozen=True)
class TwoViewReconstruction:
    """Filtered landmarks and diagnostics in the first-camera coordinate frame."""

    points_first_camera: np.ndarray
    accepted_mask: np.ndarray
    depth_first: np.ndarray
    depth_second: np.ndarray
    reprojection_error_first_px: np.ndarray
    reprojection_error_second_px: np.ndarray
    triangulation_angle_degrees: np.ndarray
    stats: LandmarkFilterStats
    camera_center_first: np.ndarray
    camera_center_second: np.ndarray
    rotation_second_from_first: np.ndarray
    translation_second_from_first: np.ndarray

    def summary(self) -> dict[str, object]:
        accepted = self.accepted_mask

        def distribution(values: np.ndarray) -> dict[str, float | None]:
            selected = values[accepted & np.isfinite(values)]
            if selected.size == 0:
                return {"median": None, "p95": None, "max": None}
            return {
                "median": float(np.median(selected)),
                "p95": float(np.percentile(selected, 95)),
                "max": float(np.max(selected)),
            }

        return {
            "filter_counts": asdict(self.stats),
            "accepted_ratio": self.stats.accepted / max(self.stats.candidates, 1),
            "positive_depth_ratio": self.stats.positive_depth_both / max(self.stats.candidates, 1),
            "reprojection_first_px": distribution(self.reprojection_error_first_px),
            "reprojection_second_px": distribution(self.reprojection_error_second_px),
            "triangulation_angle_degrees": distribution(self.triangulation_angle_degrees),
            "depth_first_arbitrary_units": distribution(self.depth_first),
            "depth_second_arbitrary_units": distribution(self.depth_second),
            "baseline_arbitrary_units": float(np.linalg.norm(
                self.camera_center_second - self.camera_center_first
            )),
        }


def _project(points: np.ndarray, projection: np.ndarray) -> np.ndarray:
    homogeneous = np.column_stack([points, np.ones(len(points), dtype=np.float64)])
    pixels = (projection @ homogeneous.T).T
    result = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(pixels).all(axis=1) & (np.abs(pixels[:, 2]) > 1e-12)
    result[valid] = pixels[valid, :2] / pixels[valid, 2:3]
    return result


def triangulation_angles_degrees(
    points_first_camera: np.ndarray,
    camera_center_first: np.ndarray,
    camera_center_second: np.ndarray,
) -> np.ndarray:
    ray_first = points_first_camera - camera_center_first
    ray_second = points_first_camera - camera_center_second
    denominator = np.linalg.norm(ray_first, axis=1) * np.linalg.norm(ray_second, axis=1)
    cosine = np.full(len(points_first_camera), np.nan, dtype=np.float64)
    valid = denominator > 1e-12
    cosine[valid] = np.sum(ray_first[valid] * ray_second[valid], axis=1) / denominator[valid]
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def triangulate_two_view(
    points_first: np.ndarray,
    points_second: np.ndarray,
    camera_matrix: np.ndarray,
    rotation_second_from_first: np.ndarray,
    translation_second_from_first: np.ndarray,
    thresholds: TriangulationThresholds = TriangulationThresholds(),
) -> TwoViewReconstruction:
    """Triangulate into camera one's frame using a one-unit monocular baseline.

    The relative transform follows ``X_second = R X_first + t``. Projection
    matrices are therefore ``K[I|0]`` and ``K[R|t]``. Camera two's center in
    camera-one coordinates is ``-R.T @ t``.
    """
    first = np.asarray(points_first, dtype=np.float64)
    second = np.asarray(points_second, dtype=np.float64)
    camera = np.asarray(camera_matrix, dtype=np.float64)
    rotation = np.asarray(rotation_second_from_first, dtype=np.float64)
    translation = np.asarray(translation_second_from_first, dtype=np.float64).reshape(3)
    if first.ndim != 2 or first.shape[1:] != (2,) or second.shape != first.shape:
        raise ValueError("point correspondences must be matching N x 2 arrays")
    if camera.shape != (3, 3) or rotation.shape != (3, 3):
        raise ValueError("camera and rotation matrices must be 3 x 3")
    if not np.all(np.isfinite(camera)) or not np.all(np.isfinite(rotation)):
        raise ValueError("camera and rotation matrices must be finite")
    translation_norm = float(np.linalg.norm(translation))
    if not np.isfinite(translation_norm) or translation_norm < 1e-12:
        raise ValueError("triangulation requires a finite nonzero translation")
    translation = translation / translation_norm
    count = len(first)
    projection_first = camera @ np.column_stack([np.eye(3), np.zeros(3)])
    projection_second = camera @ np.column_stack([rotation, translation])

    valid_input = np.isfinite(first).all(axis=1) & np.isfinite(second).all(axis=1)
    homogeneous = np.full((4, count), np.nan, dtype=np.float64)
    if np.any(valid_input):
        homogeneous[:, valid_input] = cv2.triangulatePoints(
            projection_first, projection_second,
            first[valid_input].T, second[valid_input].T,
        )
    finite_homogeneous = (
        valid_input
        & np.isfinite(homogeneous).all(axis=0)
        & (np.abs(homogeneous[3]) > thresholds.homogeneous_epsilon)
    )
    points = np.full((count, 3), np.nan, dtype=np.float64)
    points[finite_homogeneous] = (
        homogeneous[:3, finite_homogeneous] / homogeneous[3, finite_homogeneous]
    ).T
    points_second_frame = (rotation @ points.T).T + translation
    depth_first = points[:, 2]
    depth_second = points_second_frame[:, 2]
    positive = (
        finite_homogeneous
        & (depth_first > thresholds.min_depth)
        & (depth_second > thresholds.min_depth)
    )

    reprojection_first = _project(points, projection_first)
    reprojection_second = _project(points, projection_second)
    error_first = np.linalg.norm(reprojection_first - first, axis=1)
    error_second = np.linalg.norm(reprojection_second - second, axis=1)
    reprojection_consistent = (
        positive
        & np.isfinite(error_first)
        & np.isfinite(error_second)
        & (error_first <= thresholds.max_reprojection_error_px)
        & (error_second <= thresholds.max_reprojection_error_px)
    )

    center_first = np.zeros(3, dtype=np.float64)
    center_second = -rotation.T @ translation
    angles = triangulation_angles_degrees(points, center_first, center_second)
    sufficient_angle = (
        reprojection_consistent
        & np.isfinite(angles)
        & (angles >= thresholds.min_triangulation_angle_degrees)
    )
    bounded_depth = (
        sufficient_angle
        & (depth_first <= thresholds.max_depth)
        & (depth_second <= thresholds.max_depth)
    )
    accepted = bounded_depth
    stats = LandmarkFilterStats(
        candidates=count,
        finite_homogeneous=int(np.count_nonzero(finite_homogeneous)),
        positive_depth_both=int(np.count_nonzero(positive)),
        reprojection_consistent=int(np.count_nonzero(reprojection_consistent)),
        sufficient_angle=int(np.count_nonzero(sufficient_angle)),
        bounded_depth=int(np.count_nonzero(bounded_depth)),
        accepted=int(np.count_nonzero(accepted)),
    )
    return TwoViewReconstruction(
        points, accepted, depth_first, depth_second, error_first, error_second,
        angles, stats, center_first, center_second, rotation, translation,
    )
