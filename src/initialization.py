"""Image-only keyframe-pair selection for two-view map initialization."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

import numpy as np

from .features import Features, ORBFeatures, matched_points
from .frame_source import Frame
from .geometry import MotionState, PoseEstimate, estimate_reliable_motion


@dataclass(frozen=True)
class PairSelectionThresholds:
    min_matches: int = 50
    min_geometric_inliers: int = 50
    min_median_compensated_parallax_degrees: float = 1.0
    min_p25_compensated_parallax_degrees: float = 0.5
    grid_rows: int = 4
    grid_columns: int = 6
    min_occupied_grid_ratio: float = 1.0 / 3.0


@dataclass(frozen=True)
class PairAssessment:
    anchor_index: int
    candidate_index: int
    suitable: bool
    state: MotionState
    reasons: tuple[str, ...]
    matches: int
    median_match_distance: float | None
    geometric_inliers: int
    median_compensated_parallax_degrees: float | None
    p25_compensated_parallax_degrees: float | None
    occupied_grid_cells: int
    total_grid_cells: int
    occupied_grid_ratio: float
    processing_ms: float


@dataclass(frozen=True)
class SelectedPair:
    anchor: Frame
    candidate: Frame
    assessment: PairAssessment
    estimate: PoseEstimate
    points_anchor: np.ndarray
    points_candidate: np.ndarray
    anchor_feature_indices: np.ndarray
    candidate_feature_indices: np.ndarray
    anchor_features: Features
    candidate_features: Features


@dataclass(frozen=True)
class PairSelectionResult:
    selected: SelectedPair | None
    assessments: tuple[PairAssessment, ...]


def _compensated_parallax_degrees(
    points_first: np.ndarray,
    points_second: np.ndarray,
    camera_matrix: np.ndarray,
    rotation_second_from_first: np.ndarray,
) -> np.ndarray:
    inverse_camera = np.linalg.inv(camera_matrix)
    first_h = np.column_stack([points_first, np.ones(len(points_first))])
    second_h = np.column_stack([points_second, np.ones(len(points_second))])
    first_bearings = (inverse_camera @ first_h.T).T
    second_bearings = (inverse_camera @ second_h.T).T
    first_bearings /= np.linalg.norm(first_bearings, axis=1, keepdims=True)
    second_bearings /= np.linalg.norm(second_bearings, axis=1, keepdims=True)
    rotated_first = (rotation_second_from_first @ first_bearings.T).T
    cosine = np.sum(rotated_first * second_bearings, axis=1)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _occupied_grid_cells(
    points: np.ndarray,
    image_shape: tuple[int, ...],
    rows: int,
    columns: int,
) -> int:
    if len(points) == 0:
        return 0
    height, width = image_shape[:2]
    column = np.clip((points[:, 0] / max(width, 1) * columns).astype(int), 0, columns - 1)
    row = np.clip((points[:, 1] / max(height, 1) * rows).astype(int), 0, rows - 1)
    return len(set(zip(row.tolist(), column.tolist())))


class TwoViewPairSelector:
    """Select the first candidate satisfying observable two-view geometry."""

    def __init__(
        self,
        frontend: ORBFeatures,
        camera_matrix: np.ndarray,
        *,
        thresholds: PairSelectionThresholds = PairSelectionThresholds(),
        min_pose_inliers: int = 15,
        ransac_probability: float = 0.999,
        ransac_threshold_px: float = 1.0,
        rotation_only_residual_degrees: float = 0.2,
        rotation_only_min_ratio: float = 0.6,
        rotation_innovation_degrees: float = 2.0,
        suspicious_parallax_p10_degrees: float = 1.0,
    ) -> None:
        self.frontend = frontend
        self.camera_matrix = np.asarray(camera_matrix, dtype=np.float64)
        self.thresholds = thresholds
        self.pose_options = {
            "min_inliers": min_pose_inliers,
            "ransac_probability": ransac_probability,
            "ransac_threshold_px": ransac_threshold_px,
            "rotation_only_residual_degrees": rotation_only_residual_degrees,
            "rotation_only_min_ratio": rotation_only_min_ratio,
            "rotation_innovation_degrees": rotation_innovation_degrees,
            "suspicious_parallax_p10_degrees": suspicious_parallax_p10_degrees,
        }

    def _evaluate(
        self,
        anchor: Frame,
        anchor_features: Features,
        candidate: Frame,
    ) -> tuple[PairAssessment, PoseEstimate | None, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Features | None]:
        started = perf_counter()
        empty = np.empty((0, 2), dtype=np.float64)
        empty_indices = np.empty(0, dtype=int)
        if candidate.image is None:
            assessment = PairAssessment(
                anchor.index, candidate.index, False, MotionState.TRACKING_LOST,
                ("candidate image is unreadable",), 0, None, 0, None, None, 0,
                self.thresholds.grid_rows * self.thresholds.grid_columns, 0.0,
                (perf_counter() - started) * 1000.0,
            )
            return assessment, None, empty, empty, empty_indices, empty_indices, None
        features = self.frontend.extract(candidate.image)
        matches = self.frontend.match(anchor_features, features)
        if len(matches) < self.thresholds.min_matches:
            assessment = PairAssessment(
                anchor.index, candidate.index, False, MotionState.TRACKING_LOST,
                (f"only {len(matches)} matches",), len(matches),
                (float(np.median([match.distance for match in matches])) if matches else None),
                0, None, None, 0,
                self.thresholds.grid_rows * self.thresholds.grid_columns, 0.0,
                (perf_counter() - started) * 1000.0,
            )
            return assessment, None, empty, empty, empty_indices, empty_indices, features

        points_anchor, points_candidate = matched_points(anchor_features, features, matches)
        estimate = estimate_reliable_motion(
            points_anchor, points_candidate, self.camera_matrix,
            recent_rotations=[], **self.pose_options,
        )
        mask = estimate.inlier_mask
        inlier_anchor = points_anchor[mask]
        inlier_candidate = points_candidate[mask]
        anchor_indices = np.asarray([match.queryIdx for match in matches], dtype=int)[mask]
        candidate_indices = np.asarray([match.trainIdx for match in matches], dtype=int)[mask]
        median = p25 = None
        if estimate.rotation is not None and len(inlier_anchor):
            parallax = _compensated_parallax_degrees(
                inlier_anchor, inlier_candidate, self.camera_matrix, estimate.rotation,
            )
            median = float(np.median(parallax))
            p25 = float(np.percentile(parallax, 25))
        occupied_anchor = _occupied_grid_cells(
            inlier_anchor, anchor.image.shape,
            self.thresholds.grid_rows, self.thresholds.grid_columns,
        )
        occupied_candidate = _occupied_grid_cells(
            inlier_candidate, candidate.image.shape,
            self.thresholds.grid_rows, self.thresholds.grid_columns,
        )
        occupied = min(occupied_anchor, occupied_candidate)
        total_cells = self.thresholds.grid_rows * self.thresholds.grid_columns
        coverage = occupied / total_cells
        reasons: list[str] = []
        if estimate.state != MotionState.VALID_MOTION:
            reasons.append(f"motion state is {estimate.state.value}")
        if estimate.inlier_count < self.thresholds.min_geometric_inliers:
            reasons.append(f"only {estimate.inlier_count} geometric inliers")
        if median is None or median < self.thresholds.min_median_compensated_parallax_degrees:
            reasons.append(f"median compensated parallax {median} below threshold")
        if p25 is None or p25 < self.thresholds.min_p25_compensated_parallax_degrees:
            reasons.append(f"p25 compensated parallax {p25} below threshold")
        if coverage < self.thresholds.min_occupied_grid_ratio:
            reasons.append(f"feature grid coverage {coverage:.3f} below threshold")
        suitable = not reasons
        assessment = PairAssessment(
            anchor.index, candidate.index, suitable, estimate.state, tuple(reasons),
            len(matches), float(np.median([match.distance for match in matches])),
            estimate.inlier_count, median, p25, occupied, total_cells,
            coverage, (perf_counter() - started) * 1000.0,
        )
        return assessment, estimate, inlier_anchor, inlier_candidate, anchor_indices, candidate_indices, features

    def select(self, anchor: Frame, candidates: Iterable[Frame]) -> PairSelectionResult:
        if anchor.image is None:
            raise ValueError("anchor image is unreadable")
        anchor_features = self.frontend.extract(anchor.image)
        assessments: list[PairAssessment] = []
        for candidate in candidates:
            assessment, estimate, first, second, first_indices, second_indices, candidate_features = self._evaluate(
                anchor, anchor_features, candidate,
            )
            assessments.append(assessment)
            if assessment.suitable:
                assert estimate is not None
                assert candidate_features is not None
                return PairSelectionResult(
                    SelectedPair(
                        anchor, candidate, assessment, estimate, first, second,
                        first_indices, second_indices, anchor_features,
                        candidate_features,
                    ),
                    tuple(assessments),
                )
        return PairSelectionResult(None, tuple(assessments))
