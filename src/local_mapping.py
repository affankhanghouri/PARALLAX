"""Conservative keyframe insertion and fixed-segment landmark replenishment.

All poses are ``T_segment_camera``. Triangulation projection matrices use their
inverse, ``T_camera_segment``, so new points are created directly in the
existing segment frame and inherit its arbitrary scale.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from time import perf_counter

import cv2
import numpy as np

from .features import Features
from .local_map import (
    Keyframe,
    Landmark,
    LandmarkQuality,
    MapSegment,
    Observation,
    PoseProvenance,
)
from .map_tracking import TrackingResult
from .reconstruction import triangulation_angles_degrees


@dataclass(frozen=True)
class KeyframePolicy:
    min_frame_gap: int = 2
    max_frame_gap: int = 10
    min_translation_arbitrary: float = 0.3
    min_rotation_degrees: float = 2.0
    support_ratio_trigger: float = 0.5
    critical_support_ratio: float = 0.1
    critical_min_translation_arbitrary: float = 0.3
    critical_min_frame_gap: int = 2


@dataclass(frozen=True)
class MappingThresholds:
    ratio_test: float = 0.75
    max_descriptor_distance: float = 64.0
    min_matches: int = 20
    epipolar_threshold_px: float = 1.5
    max_reprojection_error_px: float = 2.0
    min_triangulation_angle_degrees: float = 1.0
    max_depth_arbitrary: float = 100.0
    min_new_landmarks: int = 12
    grid_rows: int = 4
    grid_columns: int = 6
    min_occupied_grid_cells: int = 6
    duplicate_radius_px: float = 3.0
    max_landmarks: int = 1000
    max_new_landmarks_per_keyframe: int = 1000
    max_new_landmarks_per_grid_cell: int = 1000
    efficient_candidate_reverse: bool = False


@dataclass(frozen=True)
class KeyframeDecision:
    insert: bool
    reason: str
    frame_gap: int
    translation_arbitrary: float
    rotation_degrees: float
    support_ratio: float


@dataclass(frozen=True)
class MappingStats:
    unmatched_reference_features: int
    unmatched_current_features: int
    descriptor_matches: int
    epipolar_consistent: int
    nonduplicate_candidates: int
    finite_homogeneous: int
    positive_depth_both: int
    reprojection_consistent: int
    sufficient_angle: int
    bounded_depth: int
    occupied_grid_cells: int
    accepted_candidates: int
    inserted_landmarks: int


@dataclass(frozen=True)
class MappingResult:
    accepted: bool
    reason: str
    keyframe_id: str
    reference_keyframe_id: str
    stats: MappingStats
    processing_ms: float
    descriptor_matching_ms: float
    triangulation_ms: float

    def summary(self) -> dict[str, object]:
        return {
            "accepted": self.accepted,
            "reason": self.reason,
            "keyframe_id": self.keyframe_id,
            "reference_keyframe_id": self.reference_keyframe_id,
            "processing_ms": self.processing_ms,
            "descriptor_matching_ms": self.descriptor_matching_ms,
            "triangulation_ms": self.triangulation_ms,
            **asdict(self.stats),
        }


def _rotation_angle_degrees(rotation: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))


def decide_keyframe(
    segment: MapSegment,
    source_frame_index: int,
    pose_camera_to_segment: np.ndarray,
    tracked_inliers: int,
    active_landmarks: int,
    policy: KeyframePolicy = KeyframePolicy(),
) -> KeyframeDecision:
    """Select an accepted pose only when motion and mapping need are present."""
    if not segment.keyframes:
        return KeyframeDecision(False, "segment has no reference keyframe", 0, 0.0, 0.0, 0.0)
    reference = max(segment.keyframes.values(), key=lambda item: item.source_frame_index)
    frame_gap = source_frame_index - reference.source_frame_index
    translation = float(np.linalg.norm(
        np.asarray(pose_camera_to_segment)[:3, 3] - reference.pose_camera_to_segment[:3, 3]
    ))
    relative_rotation = reference.pose_camera_to_segment[:3, :3].T @ np.asarray(pose_camera_to_segment)[:3, :3]
    rotation = _rotation_angle_degrees(relative_rotation)
    support = tracked_inliers / max(active_landmarks, 1)
    frame_gap_threshold = (
        policy.critical_min_frame_gap
        if support <= policy.critical_support_ratio
        else policy.min_frame_gap
    )
    if frame_gap < frame_gap_threshold:
        reason = f"frame gap {frame_gap} below {frame_gap_threshold}"
        return KeyframeDecision(False, reason, frame_gap, translation, rotation, support)
    translation_threshold = (
        policy.critical_min_translation_arbitrary
        if support <= policy.critical_support_ratio
        else policy.min_translation_arbitrary
    )
    geometry_ready = (
        translation >= translation_threshold
        or rotation >= policy.min_rotation_degrees
    )
    if not geometry_ready:
        reason = "insufficient viewpoint change"
        return KeyframeDecision(False, reason, frame_gap, translation, rotation, support)
    mapping_needed = support <= policy.support_ratio_trigger or frame_gap >= policy.max_frame_gap
    if not mapping_needed:
        reason = "tracking support remains strong"
        return KeyframeDecision(False, reason, frame_gap, translation, rotation, support)
    trigger = "support decline" if support <= policy.support_ratio_trigger else "maximum keyframe gap"
    reason = (
        f"{trigger}: gap={frame_gap}, translation={translation:.3f}, "
        f"rotation={rotation:.3f} deg, support={support:.3f}, "
        f"translation-threshold={translation_threshold:.3f}, "
        f"gap-threshold={frame_gap_threshold}"
    )
    return KeyframeDecision(True, reason, frame_gap, translation, rotation, support)


def create_mapping_keyframe(
    segment: MapSegment,
    frame_id: str,
    source_frame_index: int,
    pose_camera_to_segment: np.ndarray,
    features: Features,
    tracking: TrackingResult,
    insertion_reason: str,
) -> Keyframe:
    """Create a keyframe from a pose that was already accepted by map PnP."""
    if not tracking.accepted or tracking.pose_camera_to_segment is None:
        raise ValueError("a rejected tracking result cannot create a keyframe")
    if features.descriptors is None:
        raise ValueError("a mapping keyframe requires descriptors")
    observations = {
        tracking.associations[int(index)].feature_index:
        tracking.associations[int(index)].landmark_id
        for index in tracking.inlier_association_indices
        if tracking.associations[int(index)].feature_index is not None
    }
    keyframe = Keyframe(
        frame_id, segment.segment_id, source_frame_index,
        np.asarray(pose_camera_to_segment, dtype=np.float64).copy(),
        np.asarray([keypoint.pt for keypoint in features.keypoints], dtype=np.float64),
        features.descriptors.copy(), observations, PoseProvenance.MAP_PNP,
        "VALIDATED", insertion_reason,
    )
    segment.keyframes[keyframe.keyframe_id] = keyframe
    return keyframe


def _skew(vector: np.ndarray) -> np.ndarray:
    x, y, z = np.asarray(vector, dtype=np.float64).reshape(3)
    return np.asarray([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])


def _symmetric_epipolar_errors(
    first: np.ndarray,
    second: np.ndarray,
    camera_matrix: np.ndarray,
    pose_first_camera_to_segment: np.ndarray,
    pose_second_camera_to_segment: np.ndarray,
) -> np.ndarray:
    relative_second_first = (
        np.linalg.inv(pose_second_camera_to_segment) @ pose_first_camera_to_segment
    )
    essential = _skew(relative_second_first[:3, 3]) @ relative_second_first[:3, :3]
    inverse_camera = np.linalg.inv(camera_matrix)
    fundamental = inverse_camera.T @ essential @ inverse_camera
    first_h = np.column_stack([first, np.ones(len(first))])
    second_h = np.column_stack([second, np.ones(len(second))])
    lines_second = (fundamental @ first_h.T).T
    lines_first = (fundamental.T @ second_h.T).T
    numerator = np.abs(np.sum(second_h * lines_second, axis=1))
    distance_second = numerator / np.maximum(np.linalg.norm(lines_second[:, :2], axis=1), 1e-12)
    distance_first = numerator / np.maximum(np.linalg.norm(lines_first[:, :2], axis=1), 1e-12)
    return np.sqrt(0.5 * (distance_first ** 2 + distance_second ** 2))


def _project_segment_points(
    points: np.ndarray,
    pose_camera_to_segment: np.ndarray,
    camera_matrix: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    world_to_camera = np.linalg.inv(pose_camera_to_segment)
    camera_points = (world_to_camera[:3, :3] @ points.T).T + world_to_camera[:3, 3]
    homogeneous = (camera_matrix @ camera_points.T).T
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    valid = np.isfinite(homogeneous).all(axis=1) & (np.abs(homogeneous[:, 2]) > 1e-12)
    pixels[valid] = homogeneous[valid, :2] / homogeneous[valid, 2:3]
    return pixels, camera_points[:, 2]


def _occupied_cells(points: np.ndarray, shape: tuple[int, int], rows: int, columns: int) -> int:
    if len(points) == 0:
        return 0
    height, width = shape
    xs = np.clip((points[:, 0] / width * columns).astype(int), 0, columns - 1)
    ys = np.clip((points[:, 1] / height * rows).astype(int), 0, rows - 1)
    return len(set(zip(ys.tolist(), xs.tolist())))


def _match_unmapped_features(
    reference: Keyframe,
    current: Keyframe,
    thresholds: MappingThresholds,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    reference_indices = np.asarray([
        index for index in range(len(reference.descriptors))
        if index not in reference.landmark_observations
    ], dtype=int)
    current_indices = np.asarray([
        index for index in range(len(current.descriptors))
        if index not in current.landmark_observations
    ], dtype=int)
    if len(reference_indices) < 2 or len(current_indices) < 2:
        return reference_indices, current_indices, np.empty((0, 3), dtype=float)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    first_descriptors = reference.descriptors[reference_indices]
    second_descriptors = current.descriptors[current_indices]
    forward = matcher.knnMatch(first_descriptors, second_descriptors, k=2)
    preliminary = []
    for pair in forward:
        if len(pair) != 2:
            continue
        best, second = pair
        if (
            best.distance >= thresholds.ratio_test * second.distance
            or best.distance > thresholds.max_descriptor_distance
        ):
            continue
        preliminary.append((best.queryIdx, best.trainIdx, float(best.distance)))
    if not preliminary:
        return reference_indices, current_indices, np.empty((0, 3), dtype=float)
    # Mutual consistency is needed only for current descriptors that survived
    # the forward ratio/distance filters. Matching every unmapped current
    # feature back to the reference was equivalent but needlessly expensive.
    reverse_candidates = (
        np.asarray(sorted({item[1] for item in preliminary}), dtype=int)
        if thresholds.efficient_candidate_reverse
        else np.arange(len(second_descriptors), dtype=int)
    )
    reverse = matcher.match(second_descriptors[reverse_candidates], first_descriptors)
    reverse_lookup = {
        int(reverse_candidates[match.queryIdx]): match.trainIdx for match in reverse
    }
    matches = []
    for query_index, train_index, distance in preliminary:
        if reverse_lookup.get(train_index) != query_index:
            continue
        matches.append((
            int(reference_indices[query_index]),
            int(current_indices[train_index]),
            distance,
        ))
    return reference_indices, current_indices, np.asarray(matches, dtype=float).reshape(-1, 3)


def replenish_landmarks(
    segment: MapSegment,
    reference: Keyframe,
    current: Keyframe,
    camera_matrix: np.ndarray,
    image_shape: tuple[int, ...],
    thresholds: MappingThresholds = MappingThresholds(),
) -> MappingResult:
    """Triangulate unmatched observations using two already accepted poses."""
    started = perf_counter()
    reference_indices, current_indices, matches = _match_unmapped_features(
        reference, current, thresholds,
    )
    descriptor_matching_ms = (perf_counter() - started) * 1000.0
    triangulation_ms = 0.0

    def finish(reason: str, stats: MappingStats, accepted: bool = False) -> MappingResult:
        return MappingResult(
            accepted, reason, current.keyframe_id, reference.keyframe_id,
            stats, (perf_counter() - started) * 1000.0,
            descriptor_matching_ms, triangulation_ms,
        )

    empty_stats = MappingStats(
        len(reference_indices), len(current_indices), len(matches),
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
    )
    if len(matches) < thresholds.min_matches:
        return finish(f"only {len(matches)} unmapped descriptor matches", empty_stats)

    first_feature_indices = matches[:, 0].astype(int)
    second_feature_indices = matches[:, 1].astype(int)
    first_pixels = reference.keypoint_pixels[first_feature_indices]
    second_pixels = current.keypoint_pixels[second_feature_indices]
    epipolar_errors = _symmetric_epipolar_errors(
        first_pixels, second_pixels, camera_matrix,
        reference.pose_camera_to_segment, current.pose_camera_to_segment,
    )
    epipolar_mask = np.isfinite(epipolar_errors) & (
        epipolar_errors <= thresholds.epipolar_threshold_px
    )

    active_points = np.asarray([item.position for item in segment.active_landmarks()], dtype=np.float64)
    duplicate_mask = np.zeros(len(matches), dtype=bool)
    if len(active_points):
        projected_first, depth_first_existing = _project_segment_points(
            active_points, reference.pose_camera_to_segment, camera_matrix,
        )
        projected_second, depth_second_existing = _project_segment_points(
            active_points, current.pose_camera_to_segment, camera_matrix,
        )
        # Vectorize candidate-to-map proximity. The former Python loop executed
        # two full landmark scans per match and dominated mapping-heavy frames.
        valid_first = depth_first_existing > 0
        valid_second = depth_second_existing > 0
        distance_first_sq = np.sum(
            (first_pixels[:, None, :] - projected_first[None, :, :]) ** 2, axis=2,
        )
        distance_second_sq = np.sum(
            (second_pixels[:, None, :] - projected_second[None, :, :]) ** 2, axis=2,
        )
        radius_sq = thresholds.duplicate_radius_px ** 2
        duplicate_mask = (
            np.any((distance_first_sq <= radius_sq) & valid_first[None, :], axis=1)
            | np.any((distance_second_sq <= radius_sq) & valid_second[None, :], axis=1)
        )
    candidate_mask = epipolar_mask & ~duplicate_mask

    projection_first = camera_matrix @ np.linalg.inv(reference.pose_camera_to_segment)[:3]
    projection_second = camera_matrix @ np.linalg.inv(current.pose_camera_to_segment)[:3]
    homogeneous = np.full((4, len(matches)), np.nan, dtype=np.float64)
    if np.any(candidate_mask):
        triangulation_started = perf_counter()
        homogeneous[:, candidate_mask] = cv2.triangulatePoints(
            projection_first, projection_second,
            first_pixels[candidate_mask].T, second_pixels[candidate_mask].T,
        )
        triangulation_ms = (perf_counter() - triangulation_started) * 1000.0
    finite = (
        candidate_mask & np.isfinite(homogeneous).all(axis=0)
        & (np.abs(homogeneous[3]) > 1e-10)
    )
    points = np.full((len(matches), 3), np.nan, dtype=np.float64)
    points[finite] = (homogeneous[:3, finite] / homogeneous[3, finite]).T
    reprojection_first, depth_first = _project_segment_points(
        points, reference.pose_camera_to_segment, camera_matrix,
    )
    reprojection_second, depth_second = _project_segment_points(
        points, current.pose_camera_to_segment, camera_matrix,
    )
    positive = finite & (depth_first > 1e-6) & (depth_second > 1e-6)
    error_first = np.linalg.norm(reprojection_first - first_pixels, axis=1)
    error_second = np.linalg.norm(reprojection_second - second_pixels, axis=1)
    reprojection = (
        positive & np.isfinite(error_first) & np.isfinite(error_second)
        & (error_first <= thresholds.max_reprojection_error_px)
        & (error_second <= thresholds.max_reprojection_error_px)
    )
    angles = triangulation_angles_degrees(
        points, reference.pose_camera_to_segment[:3, 3], current.pose_camera_to_segment[:3, 3],
    )
    sufficient_angle = (
        reprojection & np.isfinite(angles)
        & (angles >= thresholds.min_triangulation_angle_degrees)
    )
    bounded = (
        sufficient_angle
        & (depth_first <= thresholds.max_depth_arbitrary)
        & (depth_second <= thresholds.max_depth_arbitrary)
    )
    accepted_indices = np.flatnonzero(bounded)
    occupied = _occupied_cells(
        second_pixels[accepted_indices], image_shape[:2],
        thresholds.grid_rows, thresholds.grid_columns,
    )
    stats = MappingStats(
        len(reference_indices), len(current_indices), len(matches),
        int(np.count_nonzero(epipolar_mask)), int(np.count_nonzero(candidate_mask)),
        int(np.count_nonzero(finite)), int(np.count_nonzero(positive)),
        int(np.count_nonzero(reprojection)), int(np.count_nonzero(sufficient_angle)),
        int(np.count_nonzero(bounded)), occupied, len(accepted_indices), 0,
    )
    if len(accepted_indices) < thresholds.min_new_landmarks:
        return finish(
            f"only {len(accepted_indices)} geometrically valid new landmarks", stats,
        )
    if occupied < thresholds.min_occupied_grid_cells:
        return finish(f"new landmarks occupy only {occupied} image grid cells", stats)
    capacity = max(0, thresholds.max_landmarks - len(segment.landmarks))
    if capacity == 0:
        return finish("map landmark capacity reached", stats)
    # Prefer geometrically strong points while enforcing a per-cell budget.
    # This bounds map growth and prevents a textured region from monopolizing
    # the active map. Thresholds default to effectively unlimited for V2.2.
    ranked = sorted(
        accepted_indices.tolist(),
        key=lambda index: (
            -float(angles[index]),
            float(max(error_first[index], error_second[index])),
            float(matches[index, 2]),
        ),
    )
    cell_counts: dict[tuple[int, int], int] = {}
    balanced: list[int] = []
    height, width = image_shape[:2]
    limit = min(capacity, thresholds.max_new_landmarks_per_keyframe)
    for index in ranked:
        x = min(thresholds.grid_columns - 1, max(0, int(second_pixels[index, 0] / width * thresholds.grid_columns)))
        y = min(thresholds.grid_rows - 1, max(0, int(second_pixels[index, 1] / height * thresholds.grid_rows)))
        cell = (y, x)
        if cell_counts.get(cell, 0) >= thresholds.max_new_landmarks_per_grid_cell:
            continue
        balanced.append(index)
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
        if len(balanced) >= limit:
            break
    accepted_indices = np.asarray(balanced, dtype=int)
    selected_occupied = _occupied_cells(
        second_pixels[accepted_indices], image_shape[:2],
        thresholds.grid_rows, thresholds.grid_columns,
    )
    if len(accepted_indices) < thresholds.min_new_landmarks:
        return finish(f"only {len(accepted_indices)} new landmarks remain after spatial budget", stats)
    if selected_occupied < thresholds.min_occupied_grid_cells:
        return finish(f"budgeted new landmarks occupy only {selected_occupied} image grid cells", stats)
    inserted = 0
    for match_index in accepted_indices:
        first_feature = int(first_feature_indices[match_index])
        second_feature = int(second_feature_indices[match_index])
        landmark_id = f"{segment.segment_id}:landmark-{len(segment.landmarks):06d}"
        quality = LandmarkQuality(
            float(angles[match_index]), float(error_first[match_index]),
            float(error_second[match_index]), float(depth_first[match_index]),
            float(depth_second[match_index]),
        )
        observations = [
            Observation(
                reference.keyframe_id, landmark_id, first_feature,
                first_pixels[match_index].copy(), float(error_first[match_index]), True,
            ),
            Observation(
                current.keyframe_id, landmark_id, second_feature,
                second_pixels[match_index].copy(), float(error_second[match_index]), True,
            ),
        ]
        segment.landmarks[landmark_id] = Landmark(
            landmark_id, segment.segment_id, points[match_index].copy(),
            current.descriptors[second_feature][None, :].copy(),
            (current.keyframe_id,), quality, observations,
            created_frame_index=current.source_frame_index,
            last_observed_frame_index=current.source_frame_index,
            association_successes=2,
            pnp_inlier_observations=2,
            reprojection_error_ema_px=float(
                0.5 * (error_first[match_index] + error_second[match_index])
            ),
        )
        reference.landmark_observations[first_feature] = landmark_id
        current.landmark_observations[second_feature] = landmark_id
        inserted += 1
    final_stats = MappingStats(
        stats.unmatched_reference_features,
        stats.unmatched_current_features,
        stats.descriptor_matches,
        stats.epipolar_consistent,
        stats.nonduplicate_candidates,
        stats.finite_homogeneous,
        stats.positive_depth_both,
        stats.reprojection_consistent,
        stats.sufficient_angle,
        stats.bounded_depth,
        stats.occupied_grid_cells,
        stats.accepted_candidates,
        inserted,
    )
    return finish("", final_stats, True)
