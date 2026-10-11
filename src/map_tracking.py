"""Descriptor association and robust calibrated PnP for a local map segment."""

from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter

import cv2
import numpy as np

from .features import Features
from .local_map import (
    FramePose,
    Landmark,
    MapSegment,
    MapSegmentStatus,
    Observation,
    PoseProvenance,
)


_HAMMING_BYTE_COUNTS = np.unpackbits(
    np.arange(256, dtype=np.uint8)[:, None], axis=1,
).sum(axis=1)


@dataclass(frozen=True)
class PnPThresholds:
    ratio_test: float = 0.75
    max_descriptor_distance: float = 64.0
    min_correspondences: int = 8
    min_inliers: int = 6
    min_inlier_ratio: float = 0.5
    ransac_reprojection_error_px: float = 3.0
    max_median_reprojection_error_px: float = 2.0
    max_p95_reprojection_error_px: float = 4.0
    grid_rows: int = 4
    grid_columns: int = 6
    min_occupied_grid_cells: int = 4
    max_rotation_jump_degrees: float = 15.0
    max_translation_jump_arbitrary: float = 2.0


@dataclass(frozen=True)
class LandmarkAssociation:
    landmark_id: str
    feature_index: int | None
    distance: float
    point_segment: np.ndarray
    pixel: np.ndarray
    source: str = "DESCRIPTOR"


@dataclass(frozen=True)
class OpticalFlowDiagnostics:
    attempted: int
    forward_valid: int
    backward_valid: int
    forward_backward_pass: int
    lk_error_pass: int
    projection_pass: int
    accepted: int
    forward_backward_median_px: float | None


@dataclass(frozen=True)
class AssociationDiagnostics:
    mode: str
    active_landmarks: int
    predicted_visible_landmarks: int
    landmarks_with_two_candidates: int
    distance_pass: int
    ratio_pass: int
    consistency_pass: int
    unique_associations: int
    nearest_distance_median: float | None
    nearest_distance_p95: float | None
    nearest_ratio_median: float | None
    projection_ms: float = 0.0
    descriptor_matching_ms: float = 0.0


@dataclass(frozen=True)
class TrackingResult:
    accepted: bool
    reason: str
    pose_camera_to_segment: np.ndarray | None
    associations: tuple[LandmarkAssociation, ...]
    inlier_association_indices: np.ndarray
    inlier_reprojection_errors_px: np.ndarray
    ransac_inliers: int
    inlier_ratio: float
    reprojection_median_px: float | None
    reprojection_p95_px: float | None
    occupied_grid_cells: int
    rotation_jump_degrees: float | None
    translation_jump_arbitrary: float | None
    association_ms: float
    pnp_ms: float
    validation_ms: float
    association_diagnostics: AssociationDiagnostics


def _distribution(values: list[float], percentile: float) -> float | None:
    return float(np.percentile(values, percentile)) if values else None


def _project_landmarks(
    landmarks: list[Landmark],
    pose_camera_to_segment: np.ndarray,
    camera_matrix: np.ndarray,
    image_shape: tuple[int, ...],
) -> tuple[np.ndarray, np.ndarray]:
    """Project segment-frame landmarks using ``T_camera_segment``.

    ``pose_camera_to_segment`` is ``T_segment_camera`` and is inverted before
    projection. Returned visibility requires positive depth and image bounds.
    """
    if not landmarks:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=bool)
    world_to_camera = np.linalg.inv(np.asarray(pose_camera_to_segment, dtype=np.float64))
    points = np.asarray([item.position for item in landmarks], dtype=np.float64)
    camera_points = (world_to_camera[:3, :3] @ points.T).T + world_to_camera[:3, 3]
    homogeneous = (np.asarray(camera_matrix, dtype=np.float64) @ camera_points.T).T
    pixels = np.full((len(points), 2), np.nan, dtype=np.float64)
    finite = np.isfinite(homogeneous).all(axis=1) & (np.abs(homogeneous[:, 2]) > 1e-12)
    pixels[finite] = homogeneous[finite, :2] / homogeneous[finite, 2:3]
    height, width = image_shape[:2]
    visible = (
        finite & (camera_points[:, 2] > 0)
        & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    return pixels, visible


def _associate_global(
    landmarks: list[Landmark],
    features: Features,
    ratio_test: float,
    max_descriptor_distance: float,
    predicted_visible: int,
) -> tuple[list[LandmarkAssociation], AssociationDiagnostics]:
    descriptor_rows = []
    descriptor_landmark_indices = []
    for landmark_index, landmark in enumerate(landmarks):
        for descriptor in np.atleast_2d(landmark.descriptor_bank):
            descriptor_rows.append(descriptor)
            descriptor_landmark_indices.append(landmark_index)
    descriptors = np.asarray(descriptor_rows, dtype=np.uint8)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    forward = matcher.knnMatch(descriptors, features.descriptors, k=2)
    reverse = matcher.match(features.descriptors, descriptors)
    reverse_landmark = {match.queryIdx: match.trainIdx for match in reverse}
    candidates: list[tuple[float, int, int]] = []
    nearest_distances: list[float] = []
    nearest_ratios: list[float] = []
    distance_landmarks: set[int] = set()
    ratio_landmarks: set[int] = set()
    consistency_landmarks: set[int] = set()
    for descriptor_index, pair in enumerate(forward):
        if len(pair) != 2:
            continue
        first, second = pair
        landmark_index = descriptor_landmark_indices[descriptor_index]
        nearest_distances.append(float(first.distance))
        nearest_ratios.append(float(first.distance / max(second.distance, 1e-12)))
        if first.distance <= max_descriptor_distance:
            distance_landmarks.add(landmark_index)
        if first.distance >= ratio_test * second.distance or first.distance > max_descriptor_distance:
            continue
        ratio_landmarks.add(landmark_index)
        reverse_descriptor = reverse_landmark.get(first.trainIdx)
        if (
            reverse_descriptor is None
            or descriptor_landmark_indices[reverse_descriptor] != landmark_index
        ):
            continue
        consistency_landmarks.add(landmark_index)
        candidates.append((float(first.distance), landmark_index, first.trainIdx))
    result = _unique_associations(candidates, landmarks, features)
    diagnostics = AssociationDiagnostics(
        "global", len(landmarks), predicted_visible, len(landmarks),
        len(distance_landmarks), len(ratio_landmarks), len(consistency_landmarks), len(result),
        _distribution(nearest_distances, 50), _distribution(nearest_distances, 95),
        _distribution(nearest_ratios, 50),
    )
    return result, diagnostics


def _unique_associations(
    candidates: list[tuple[float, int, int]],
    landmarks: list[Landmark],
    features: Features,
) -> list[LandmarkAssociation]:
    """Greedily enforce one landmark and one image feature per association."""
    result: list[LandmarkAssociation] = []
    used_landmarks: set[int] = set()
    used_features: set[int] = set()
    for distance, landmark_index, feature_index in sorted(candidates):
        if landmark_index in used_landmarks or feature_index in used_features:
            continue
        used_landmarks.add(landmark_index)
        used_features.add(feature_index)
        landmark = landmarks[landmark_index]
        result.append(LandmarkAssociation(
            landmark.landmark_id, feature_index, distance,
            landmark.position.copy(), np.asarray(features.keypoints[feature_index].pt, dtype=np.float64),
        ))
    return result


def _associate_projected(
    landmarks: list[Landmark],
    features: Features,
    ratio_test: float,
    max_descriptor_distance: float,
    pose_camera_to_segment: np.ndarray,
    camera_matrix: np.ndarray,
    image_shape: tuple[int, ...],
    projection_search_radius_px: float,
    descriptor_reference_limit: int,
) -> tuple[list[LandmarkAssociation], AssociationDiagnostics]:
    projection_started = perf_counter()
    projections, visible = _project_landmarks(
        landmarks, pose_camera_to_segment, camera_matrix, image_shape,
    )
    feature_pixels = np.asarray([item.pt for item in features.keypoints], dtype=np.float64)
    # Spatial bins avoid an O(landmarks × all image features) radius scan.
    cell_size = projection_search_radius_px
    spatial_bins: dict[tuple[int, int], list[int]] = {}
    feature_cells = np.floor(feature_pixels / cell_size).astype(int)
    for feature_index, (cell_x, cell_y) in enumerate(feature_cells):
        spatial_bins.setdefault((int(cell_x), int(cell_y)), []).append(feature_index)
    projection_ms = (perf_counter() - projection_started) * 1000.0
    matching_started = perf_counter()
    candidates: list[tuple[float, int, int]] = []
    nearest_distances: list[float] = []
    nearest_ratios: list[float] = []
    with_two = distance_pass = ratio_pass = 0
    for landmark_index, landmark in enumerate(landmarks):
        if not visible[landmark_index]:
            continue
        cell_x, cell_y = np.floor(projections[landmark_index] / cell_size).astype(int)
        nearby_candidates = [
            feature_index
            for offset_y in (-1, 0, 1)
            for offset_x in (-1, 0, 1)
            for feature_index in spatial_bins.get(
                (int(cell_x + offset_x), int(cell_y + offset_y)), ()
            )
        ]
        if not nearby_candidates:
            continue
        nearby_array = np.asarray(nearby_candidates, dtype=int)
        nearby = nearby_array[
            np.linalg.norm(feature_pixels[nearby_array] - projections[landmark_index], axis=1)
            <= projection_search_radius_px
        ]
        if len(nearby) < 2:
            continue
        with_two += 1
        bank = np.atleast_2d(landmark.descriptor_bank).astype(np.uint8, copy=False)
        if descriptor_reference_limit:
            bank = bank[-descriptor_reference_limit:]
        nearby_descriptors = features.descriptors[nearby]
        xor = np.bitwise_xor(bank[:, None, :], nearby_descriptors[None, :, :])
        distances = _HAMMING_BYTE_COUNTS[xor].sum(axis=2).min(axis=0).astype(float)
        order = np.argsort(distances)
        best = float(distances[order[0]])
        second = float(distances[order[1]])
        nearest_distances.append(best)
        nearest_ratios.append(best / max(second, 1e-12))
        if best <= max_descriptor_distance:
            distance_pass += 1
        if best > max_descriptor_distance or best >= ratio_test * second:
            continue
        ratio_pass += 1
        candidates.append((best, landmark_index, int(nearby[order[0]])))
    result = _unique_associations(candidates, landmarks, features)
    diagnostics = AssociationDiagnostics(
        "projected", len(landmarks), int(np.count_nonzero(visible)), with_two,
        distance_pass, ratio_pass, ratio_pass, len(result),
        _distribution(nearest_distances, 50), _distribution(nearest_distances, 95),
        _distribution(nearest_ratios, 50),
        projection_ms, (perf_counter() - matching_started) * 1000.0,
    )
    return result, diagnostics


def associate_landmarks(
    landmarks: list[Landmark],
    features: Features,
    *,
    ratio_test: float,
    max_descriptor_distance: float,
    mode: str = "global",
    pose_camera_to_segment: np.ndarray | None = None,
    camera_matrix: np.ndarray | None = None,
    image_shape: tuple[int, ...] | None = None,
    projection_search_radius_px: float = 35.0,
    descriptor_reference_limit: int = 0,
) -> list[LandmarkAssociation]:
    """Return ratio-filtered, one-to-one landmark-feature matches."""
    associations, _ = associate_landmarks_with_diagnostics(
        landmarks, features, ratio_test=ratio_test,
        max_descriptor_distance=max_descriptor_distance, mode=mode,
        pose_camera_to_segment=pose_camera_to_segment,
        camera_matrix=camera_matrix, image_shape=image_shape,
        projection_search_radius_px=projection_search_radius_px,
        descriptor_reference_limit=descriptor_reference_limit,
    )
    return associations


def associate_landmarks_with_diagnostics(
    landmarks: list[Landmark],
    features: Features,
    *,
    ratio_test: float,
    max_descriptor_distance: float,
    mode: str = "global",
    pose_camera_to_segment: np.ndarray | None = None,
    camera_matrix: np.ndarray | None = None,
    image_shape: tuple[int, ...] | None = None,
    projection_search_radius_px: float = 35.0,
    descriptor_reference_limit: int = 0,
) -> tuple[list[LandmarkAssociation], AssociationDiagnostics]:
    """Associate landmarks and retain counts for each rejection stage."""
    if not landmarks or features.descriptors is None or len(features.descriptors) < 2:
        diagnostics = AssociationDiagnostics(
            mode, len(landmarks), 0, 0, 0, 0, 0, 0, None, None, None,
        )
        return [], diagnostics
    predicted_visible = len(landmarks)
    if pose_camera_to_segment is not None and camera_matrix is not None and image_shape is not None:
        _, visibility = _project_landmarks(
            landmarks, pose_camera_to_segment, camera_matrix, image_shape,
        )
        predicted_visible = int(np.count_nonzero(visibility))
    if mode == "global":
        return _associate_global(
            landmarks, features, ratio_test, max_descriptor_distance, predicted_visible,
        )
    if mode != "projected":
        raise ValueError(f"unknown landmark association mode: {mode}")
    if pose_camera_to_segment is None or camera_matrix is None or image_shape is None:
        raise ValueError("projected association requires pose, camera matrix, and image shape")
    return _associate_projected(
        landmarks, features, ratio_test, max_descriptor_distance,
        pose_camera_to_segment, camera_matrix, image_shape, projection_search_radius_px,
        descriptor_reference_limit,
    )


def _rotation_angle_degrees(rotation: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))


def _occupied_cells(pixels: np.ndarray, image_shape: tuple[int, ...], rows: int, columns: int) -> int:
    if len(pixels) == 0:
        return 0
    height, width = image_shape[:2]
    xs = np.clip((pixels[:, 0] / width * columns).astype(int), 0, columns - 1)
    ys = np.clip((pixels[:, 1] / height * rows).astype(int), 0, rows - 1)
    return len(set(zip(ys.tolist(), xs.tolist())))


def estimate_map_pose(
    segment: MapSegment,
    features: Features,
    image_shape: tuple[int, ...],
    camera_matrix: np.ndarray,
    previous_pose_camera_to_segment: np.ndarray,
    *,
    thresholds: PnPThresholds = PnPThresholds(),
    association_mode: str = "global",
    projection_search_radius_px: float = 35.0,
    descriptor_reference_limit: int = 0,
    supplemental_associations: tuple[LandmarkAssociation, ...] = (),
) -> TrackingResult:
    association_started = perf_counter()
    associations, association_diagnostics = associate_landmarks_with_diagnostics(
        segment.active_landmarks(), features,
        ratio_test=thresholds.ratio_test,
        max_descriptor_distance=thresholds.max_descriptor_distance,
        mode=association_mode,
        pose_camera_to_segment=previous_pose_camera_to_segment,
        camera_matrix=camera_matrix,
        image_shape=image_shape,
        projection_search_radius_px=projection_search_radius_px,
        descriptor_reference_limit=descriptor_reference_limit,
    )
    used_landmarks = {item.landmark_id for item in associations}
    used_pixels = [item.pixel for item in associations]
    for association in supplemental_associations:
        if association.landmark_id in used_landmarks:
            continue
        if any(np.linalg.norm(association.pixel - pixel) < 2.0 for pixel in used_pixels):
            continue
        associations.append(association)
        used_landmarks.add(association.landmark_id)
        used_pixels.append(association.pixel)
    association_ms = (perf_counter() - association_started) * 1000.0
    empty_indices = np.empty(0, dtype=int)

    def rejected(reason: str, *, pnp_ms: float = 0.0, validation_ms: float = 0.0) -> TrackingResult:
        return TrackingResult(
            False, reason, None, tuple(associations), empty_indices,
            np.empty(0, dtype=float), 0, 0.0, None, None, 0, None, None,
            association_ms, pnp_ms, validation_ms, association_diagnostics,
        )

    if len(associations) < thresholds.min_correspondences:
        return rejected(f"only {len(associations)} unique 3D-2D correspondences")
    object_points = np.asarray([item.point_segment for item in associations], dtype=np.float64)
    image_points = np.asarray([item.pixel for item in associations], dtype=np.float64)
    pnp_started = perf_counter()
    success, rotation_vector, translation, inliers = cv2.solvePnPRansac(
        object_points, image_points, camera_matrix, None,
        iterationsCount=100,
        reprojectionError=thresholds.ransac_reprojection_error_px,
        confidence=0.999,
        flags=cv2.SOLVEPNP_EPNP,
    )
    pnp_ms = (perf_counter() - pnp_started) * 1000.0
    if not success or rotation_vector is None or translation is None or inliers is None:
        return rejected("PnP RANSAC failed", pnp_ms=pnp_ms)
    inlier_indices = inliers.reshape(-1).astype(int)
    if len(inlier_indices) >= 6:
        success_refined, rotation_vector, translation = cv2.solvePnP(
            object_points[inlier_indices], image_points[inlier_indices], camera_matrix, None,
            rotation_vector, translation, True, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not success_refined:
            return rejected("PnP nonlinear refinement failed", pnp_ms=pnp_ms)
    pnp_ms = (perf_counter() - pnp_started) * 1000.0
    validation_started = perf_counter()
    rotation, _ = cv2.Rodrigues(rotation_vector)
    world_to_camera = np.eye(4, dtype=np.float64)
    world_to_camera[:3, :3] = rotation
    world_to_camera[:3, 3] = translation.reshape(3)
    if not np.all(np.isfinite(world_to_camera)) or abs(np.linalg.det(rotation) - 1.0) > 1e-3:
        return rejected("non-finite or invalid PnP transformation", pnp_ms=pnp_ms)
    pose = np.linalg.inv(world_to_camera)
    projected, _ = cv2.projectPoints(
        object_points[inlier_indices], rotation_vector, translation, camera_matrix, None,
    )
    errors = np.linalg.norm(projected.reshape(-1, 2) - image_points[inlier_indices], axis=1)
    median_error = float(np.median(errors))
    p95_error = float(np.percentile(errors, 95))
    ratio = len(inlier_indices) / len(associations)
    occupied = _occupied_cells(
        image_points[inlier_indices], image_shape,
        thresholds.grid_rows, thresholds.grid_columns,
    )
    relative = np.linalg.inv(previous_pose_camera_to_segment) @ pose
    rotation_jump = _rotation_angle_degrees(relative[:3, :3])
    translation_jump = float(np.linalg.norm(relative[:3, 3]))
    reasons = []
    if len(inlier_indices) < thresholds.min_inliers:
        reasons.append(f"only {len(inlier_indices)} PnP inliers")
    if ratio < thresholds.min_inlier_ratio:
        reasons.append(f"PnP inlier ratio {ratio:.3f} below threshold")
    if median_error > thresholds.max_median_reprojection_error_px:
        reasons.append(f"median reprojection error {median_error:.3f} px above threshold")
    if p95_error > thresholds.max_p95_reprojection_error_px:
        reasons.append(f"p95 reprojection error {p95_error:.3f} px above threshold")
    if occupied < thresholds.min_occupied_grid_cells:
        reasons.append(f"only {occupied} occupied image grid cells")
    # These are gross sanity bounds, not a constant-velocity motion model.
    if rotation_jump > thresholds.max_rotation_jump_degrees:
        reasons.append(f"rotation jump {rotation_jump:.3f} deg above sanity bound")
    if translation_jump > thresholds.max_translation_jump_arbitrary:
        reasons.append(f"translation jump {translation_jump:.3f} above sanity bound")
    validation_ms = (perf_counter() - validation_started) * 1000.0
    return TrackingResult(
        not reasons, "; ".join(reasons), (pose if not reasons else None),
        tuple(associations), inlier_indices, errors, len(inlier_indices), ratio,
        median_error, p95_error, occupied, rotation_jump, translation_jump,
        association_ms, pnp_ms, validation_ms, association_diagnostics,
    )


def update_landmark_references(
    segment: MapSegment,
    result: TrackingResult,
    features: Features,
    frame_id: str,
    *,
    policy: str,
    max_bank_size: int,
) -> int:
    """Update references only from inliers of an already accepted map pose.

    This function must be called after pose validation. It never turns an
    association into evidence for the pose that authorized that association.
    """
    if policy == "fixed" or not result.accepted or features.descriptors is None:
        return 0
    if policy not in {"latest", "bounded_recent"}:
        raise ValueError(f"unknown descriptor update policy: {policy}")
    if max_bank_size < 1:
        raise ValueError("descriptor bank size must be positive")
    updated = 0
    for association_index in result.inlier_association_indices:
        association = result.associations[int(association_index)]
        if association.feature_index is None:
            continue
        landmark = segment.landmarks[association.landmark_id]
        descriptor = features.descriptors[association.feature_index][None, :].copy()
        if policy == "latest":
            landmark.descriptor_bank = descriptor
            landmark.reference_frame_ids = (frame_id,)
        else:
            landmark.descriptor_bank = np.vstack([landmark.descriptor_bank, descriptor])[-max_bank_size:]
            landmark.reference_frame_ids = (*landmark.reference_frame_ids, frame_id)[-max_bank_size:]
        updated += 1
    return updated


def track_landmarks_optical_flow(
    segment: MapSegment,
    previous_image: np.ndarray,
    current_image: np.ndarray,
    previous_landmark_pixels: dict[str, np.ndarray],
    predicted_pose_camera_to_segment: np.ndarray,
    camera_matrix: np.ndarray,
    *,
    projection_search_radius_px: float,
    max_forward_backward_error_px: float,
    max_lk_error: float,
) -> tuple[list[LandmarkAssociation], OpticalFlowDiagnostics]:
    """Track only previously validated landmark observations with FB-checked LK."""
    landmarks = [
        segment.landmarks[identifier]
        for identifier in previous_landmark_pixels
        if identifier in segment.landmarks
    ]
    if not landmarks:
        return [], OpticalFlowDiagnostics(0, 0, 0, 0, 0, 0, 0, None)
    previous_points = np.asarray([
        previous_landmark_pixels[item.landmark_id] for item in landmarks
    ], dtype=np.float32).reshape(-1, 1, 2)
    previous_gray = (
        cv2.cvtColor(previous_image, cv2.COLOR_BGR2GRAY)
        if previous_image.ndim == 3 else previous_image
    )
    current_gray = (
        cv2.cvtColor(current_image, cv2.COLOR_BGR2GRAY)
        if current_image.ndim == 3 else current_image
    )
    current_points, forward_status, forward_error = cv2.calcOpticalFlowPyrLK(
        previous_gray, current_gray, previous_points, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if current_points is None or forward_status is None or forward_error is None:
        return [], OpticalFlowDiagnostics(len(landmarks), 0, 0, 0, 0, 0, 0, None)
    backward_points, backward_status, _ = cv2.calcOpticalFlowPyrLK(
        current_gray, previous_gray, current_points, None,
        winSize=(21, 21), maxLevel=3,
        criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
    )
    if backward_points is None or backward_status is None:
        return [], OpticalFlowDiagnostics(
            len(landmarks), int(np.count_nonzero(forward_status)), 0, 0, 0, 0, 0, None,
        )
    current_flat = current_points.reshape(-1, 2).astype(np.float64)
    backward_flat = backward_points.reshape(-1, 2).astype(np.float64)
    fb_error = np.linalg.norm(backward_flat - previous_points.reshape(-1, 2), axis=1)
    forward_valid = forward_status.reshape(-1).astype(bool)
    backward_valid = backward_status.reshape(-1).astype(bool)
    fb_pass = forward_valid & backward_valid & np.isfinite(fb_error) & (
        fb_error <= max_forward_backward_error_px
    )
    lk_error = forward_error.reshape(-1)
    error_pass = fb_pass & np.isfinite(lk_error) & (lk_error <= max_lk_error)
    projections, visible = _project_landmarks(
        landmarks, predicted_pose_camera_to_segment, camera_matrix, current_image.shape,
    )
    height, width = current_image.shape[:2]
    in_image = (
        np.isfinite(current_flat).all(axis=1)
        & (current_flat[:, 0] >= 0) & (current_flat[:, 0] < width)
        & (current_flat[:, 1] >= 0) & (current_flat[:, 1] < height)
    )
    projection_pass = (
        error_pass & visible & in_image
        & (np.linalg.norm(current_flat - projections, axis=1) <= projection_search_radius_px)
    )
    associations = [
        LandmarkAssociation(
            landmark.landmark_id, None, float(fb_error[index]),
            landmark.position.copy(), current_flat[index].copy(), "OPTICAL_FLOW",
        )
        for index, landmark in enumerate(landmarks)
        if projection_pass[index]
    ]
    valid_fb = fb_error[fb_pass]
    diagnostics = OpticalFlowDiagnostics(
        len(landmarks), int(np.count_nonzero(forward_valid)),
        int(np.count_nonzero(forward_valid & backward_valid)),
        int(np.count_nonzero(fb_pass)), int(np.count_nonzero(error_pass)),
        int(np.count_nonzero(projection_pass)), len(associations),
        float(np.median(valid_fb)) if len(valid_fb) else None,
    )
    return associations, diagnostics


def update_segment_with_tracking(
    segment: MapSegment,
    frame_id: str,
    source_frame_index: int,
    result: TrackingResult,
) -> None:
    """Record a pose/observations; rejected frames never receive fictional poses."""
    if not result.accepted or result.pose_camera_to_segment is None:
        segment.trajectory_continuous = False
        segment.status = MapSegmentStatus.INTERRUPTED
        segment.frame_poses.append(FramePose(
            frame_id, source_frame_index, None, False, result.reason,
            PoseProvenance.MAP_PNP,
        ))
        return
    segment.frame_poses.append(FramePose(
        frame_id, source_frame_index, result.pose_camera_to_segment.copy(), True,
        "accepted map PnP", PoseProvenance.MAP_PNP,
    ))
    for error_index, association_index in enumerate(result.inlier_association_indices):
        association = result.associations[int(association_index)]
        landmark = segment.landmarks[association.landmark_id]
        landmark.observations.append(Observation(
            frame_id, landmark.landmark_id, association.feature_index,
            association.pixel.copy(),
            float(result.inlier_reprojection_errors_px[error_index]), True,
            "ORB" if association.feature_index is not None else association.source,
        ))
