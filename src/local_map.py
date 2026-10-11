"""Minimal in-memory map structures for one arbitrary-scale local segment."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import numpy as np

from .initialization import SelectedPair
from .reconstruction import TwoViewReconstruction


class MapSegmentStatus(str, Enum):
    ACTIVE = "ACTIVE"
    INTERRUPTED = "INTERRUPTED"
    LOST = "LOST"


class LandmarkStatus(str, Enum):
    ACTIVE = "ACTIVE"
    TEMPORARILY_INACTIVE = "TEMPORARILY_INACTIVE"
    INVALID = "INVALID"


class PoseProvenance(str, Enum):
    TWO_VIEW_INITIALIZATION = "TWO_VIEW_INITIALIZATION"
    MAP_PNP = "MAP_PNP"


@dataclass
class Observation:
    frame_id: str
    landmark_id: str
    feature_index: int | None
    pixel: np.ndarray
    reprojection_error_px: float
    inlier: bool
    feature_kind: str = "ORB"


@dataclass
class LandmarkQuality:
    triangulation_angle_degrees: float
    reprojection_error_first_px: float
    reprojection_error_second_px: float
    depth_first_arbitrary: float
    depth_second_arbitrary: float


@dataclass
class Landmark:
    landmark_id: str
    segment_id: str
    position: np.ndarray
    descriptor_bank: np.ndarray
    reference_frame_ids: tuple[str, ...]
    quality: LandmarkQuality
    observations: list[Observation] = field(default_factory=list)
    status: LandmarkStatus = LandmarkStatus.ACTIVE
    created_frame_index: int = 0
    last_observed_frame_index: int = 0
    projection_attempts: int = 0
    association_successes: int = 0
    pnp_inlier_observations: int = 0
    consecutive_visible_misses: int = 0
    reprojection_error_ema_px: float | None = None


@dataclass
class Keyframe:
    keyframe_id: str
    segment_id: str
    source_frame_index: int
    pose_camera_to_segment: np.ndarray
    keypoint_pixels: np.ndarray
    descriptors: np.ndarray
    landmark_observations: dict[int, str]
    provenance: PoseProvenance
    quality_status: str
    insertion_reason: str = ""


@dataclass
class FramePose:
    frame_id: str
    source_frame_index: int
    pose_camera_to_segment: np.ndarray | None
    accepted: bool
    reason: str
    provenance: PoseProvenance


@dataclass
class MapSegment:
    segment_id: str
    coordinate_system: str
    status: MapSegmentStatus
    trajectory_continuous: bool
    keyframes: dict[str, Keyframe] = field(default_factory=dict)
    landmarks: dict[str, Landmark] = field(default_factory=dict)
    frame_poses: list[FramePose] = field(default_factory=list)

    def active_landmarks(self) -> list[Landmark]:
        return [item for item in self.landmarks.values() if item.status == LandmarkStatus.ACTIVE]


def validate_map_integrity(segment: MapSegment) -> tuple[str, ...]:
    """Return reproducibility/ownership invariant violations for one segment."""
    errors: list[str] = []
    frame_ids = set(segment.keyframes) | {item.frame_id for item in segment.frame_poses}
    for keyframe_id, keyframe in segment.keyframes.items():
        if keyframe.keyframe_id != keyframe_id or keyframe.segment_id != segment.segment_id:
            errors.append(f"keyframe identity/segment mismatch: {keyframe_id}")
        if not np.all(np.isfinite(keyframe.pose_camera_to_segment)):
            errors.append(f"non-finite keyframe pose: {keyframe_id}")
        for feature_index, landmark_id in keyframe.landmark_observations.items():
            if landmark_id not in segment.landmarks:
                errors.append(f"keyframe references missing landmark: {landmark_id}")
            if not 0 <= feature_index < len(keyframe.descriptors):
                errors.append(f"keyframe feature index out of bounds: {keyframe_id}:{feature_index}")
    for landmark_id, landmark in segment.landmarks.items():
        if landmark.landmark_id != landmark_id or landmark.segment_id != segment.segment_id:
            errors.append(f"landmark identity/segment mismatch: {landmark_id}")
        if len(landmark.descriptor_bank) != len(landmark.reference_frame_ids):
            errors.append(f"descriptor/reference count mismatch: {landmark_id}")
        if not np.all(np.isfinite(landmark.position)):
            errors.append(f"non-finite landmark position: {landmark_id}")
        for observation in landmark.observations:
            if observation.landmark_id != landmark_id:
                errors.append(f"observation landmark mismatch: {landmark_id}")
            if observation.frame_id not in frame_ids:
                errors.append(f"observation references unknown frame: {observation.frame_id}")
            if observation.feature_kind == "ORB" and observation.feature_index is None:
                errors.append(f"ORB observation lacks feature index: {landmark_id}")
    for frame_pose in segment.frame_poses:
        if frame_pose.accepted and frame_pose.pose_camera_to_segment is None:
            errors.append(f"accepted frame lacks pose: {frame_pose.frame_id}")
        if not frame_pose.accepted and frame_pose.pose_camera_to_segment is not None:
            errors.append(f"rejected frame has a pose: {frame_pose.frame_id}")
    return tuple(errors)


def _keypoint_pixels(features) -> np.ndarray:
    return np.asarray([keypoint.pt for keypoint in features.keypoints], dtype=np.float64)


def create_initialized_segment(
    selected: SelectedPair,
    reconstruction: TwoViewReconstruction,
    *,
    segment_id: str = "segment-0000",
) -> MapSegment:
    """Convert a validated V2.0 reconstruction into a persistent local map."""
    if selected.estimate.rotation is None or selected.estimate.translation is None:
        raise ValueError("map initialization requires a full relative pose")
    if selected.anchor_features.descriptors is None or selected.candidate_features.descriptors is None:
        raise ValueError("map initialization requires descriptors in both keyframes")
    relative = np.eye(4, dtype=np.float64)
    relative[:3, :3] = selected.estimate.rotation
    relative[:3, 3] = selected.estimate.translation / np.linalg.norm(selected.estimate.translation)
    pose_first = np.eye(4, dtype=np.float64)
    pose_second = np.linalg.inv(relative)
    first_id = f"{segment_id}:keyframe-0000"
    second_id = f"{segment_id}:keyframe-0001"
    first_observations: dict[int, str] = {}
    second_observations: dict[int, str] = {}
    segment = MapSegment(
        segment_id=segment_id,
        coordinate_system="camera 1 local frame; arbitrary unit baseline",
        status=MapSegmentStatus.ACTIVE,
        trajectory_continuous=True,
    )
    for candidate_index in np.flatnonzero(reconstruction.accepted_mask):
        landmark_id = f"{segment_id}:landmark-{len(segment.landmarks):06d}"
        first_feature = int(selected.anchor_feature_indices[candidate_index])
        second_feature = int(selected.candidate_feature_indices[candidate_index])
        quality = LandmarkQuality(
            float(reconstruction.triangulation_angle_degrees[candidate_index]),
            float(reconstruction.reprojection_error_first_px[candidate_index]),
            float(reconstruction.reprojection_error_second_px[candidate_index]),
            float(reconstruction.depth_first[candidate_index]),
            float(reconstruction.depth_second[candidate_index]),
        )
        observations = [
            Observation(
                first_id, landmark_id, first_feature,
                selected.points_anchor[candidate_index].copy(),
                quality.reprojection_error_first_px, True,
            ),
            Observation(
                second_id, landmark_id, second_feature,
                selected.points_candidate[candidate_index].copy(),
                quality.reprojection_error_second_px, True,
            ),
        ]
        segment.landmarks[landmark_id] = Landmark(
            landmark_id, segment_id,
            reconstruction.points_first_camera[candidate_index].copy(),
            selected.candidate_features.descriptors[second_feature][None, :].copy(),
            (second_id,), quality, observations,
            created_frame_index=selected.anchor.index,
            last_observed_frame_index=selected.candidate.index,
            association_successes=2,
            pnp_inlier_observations=2,
            reprojection_error_ema_px=float(
                0.5 * (quality.reprojection_error_first_px + quality.reprojection_error_second_px)
            ),
        )
        first_observations[first_feature] = landmark_id
        second_observations[second_feature] = landmark_id
    segment.keyframes[first_id] = Keyframe(
        first_id, segment_id, selected.anchor.index, pose_first,
        _keypoint_pixels(selected.anchor_features), selected.anchor_features.descriptors.copy(),
        first_observations, PoseProvenance.TWO_VIEW_INITIALIZATION, "VALIDATED",
        "two-view initialization anchor",
    )
    segment.keyframes[second_id] = Keyframe(
        second_id, segment_id, selected.candidate.index, pose_second,
        _keypoint_pixels(selected.candidate_features), selected.candidate_features.descriptors.copy(),
        second_observations, PoseProvenance.TWO_VIEW_INITIALIZATION, "VALIDATED",
        "two-view initialization baseline",
    )
    segment.frame_poses.extend([
        FramePose(first_id, selected.anchor.index, pose_first.copy(), True, "initial camera", PoseProvenance.TWO_VIEW_INITIALIZATION),
        FramePose(second_id, selected.candidate.index, pose_second.copy(), True, "two-view relative pose", PoseProvenance.TWO_VIEW_INITIALIZATION),
    ])
    return segment
