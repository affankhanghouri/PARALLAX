import cv2
import numpy as np
import pytest

from src.features import Features
from src.local_map import (
    Keyframe,
    MapSegment,
    MapSegmentStatus,
    PoseProvenance,
    validate_map_integrity,
)
from src.local_mapping import (
    KeyframePolicy,
    MappingThresholds,
    create_mapping_keyframe,
    decide_keyframe,
    replenish_landmarks,
)
from src.map_tracking import estimate_map_pose


CAMERA = np.array([[700.0, 0, 320], [0, 700.0, 240], [0, 0, 1]])


def _project(points, pose_camera_to_segment):
    world_to_camera = np.linalg.inv(pose_camera_to_segment)
    camera_points = (world_to_camera[:3, :3] @ points.T).T + world_to_camera[:3, 3]
    pixels = (CAMERA @ camera_points.T).T
    return pixels[:, :2] / pixels[:, 2:]


def _keyframe(identifier, index, pose, pixels, descriptors):
    return Keyframe(
        identifier, "segment-map", index, pose, pixels, descriptors, {},
        PoseProvenance.MAP_PNP, "VALIDATED", "unit test",
    )


def test_known_pose_replenishment_preserves_existing_segment_scale_and_identity() -> None:
    rng = np.random.default_rng(60)
    points = np.column_stack([
        rng.uniform(-2.5, 2.5, 80), rng.uniform(-1.2, 1.2, 80), rng.uniform(5, 10, 80),
    ])
    first_pose = np.eye(4)
    second_pose = np.eye(4)
    second_pose[:3, 3] = [1.0, 0.0, 0.0]
    first_pixels = _project(points, first_pose)
    second_pixels = _project(points, second_pose)
    descriptors = rng.integers(0, 256, (80, 32), dtype=np.uint8)
    first = _keyframe("kf-first", 0, first_pose, first_pixels, descriptors)
    second = _keyframe("kf-second", 5, second_pose, second_pixels, descriptors.copy())
    segment = MapSegment("segment-map", "existing arbitrary scale", MapSegmentStatus.ACTIVE, True)
    segment.keyframes = {first.keyframe_id: first, second.keyframe_id: second}
    result = replenish_landmarks(
        segment, first, second, CAMERA, (480, 640, 3),
        MappingThresholds(
            min_matches=20, min_new_landmarks=20,
            min_occupied_grid_cells=4, min_triangulation_angle_degrees=0.5,
        ),
    )
    assert result.accepted, result.reason
    assert result.stats.inserted_landmarks > 60
    recovered = np.asarray([item.position for item in segment.landmarks.values()])
    assert np.median(np.min(np.linalg.norm(recovered[:, None] - points[None, :], axis=2), axis=1)) < 1e-5
    assert all(item.segment_id == segment.segment_id for item in segment.landmarks.values())
    assert all(len(item.observations) == 2 for item in segment.landmarks.values())
    assert len(second.landmark_observations) == result.stats.inserted_landmarks
    assert not validate_map_integrity(segment)
    repeated = replenish_landmarks(segment, first, second, CAMERA, (480, 640, 3))
    assert not repeated.accepted
    assert repeated.stats.inserted_landmarks == 0


def test_keyframe_policy_requires_motion_and_mapping_need() -> None:
    pose = np.eye(4)
    segment = MapSegment("segment-map", "test", MapSegmentStatus.ACTIVE, True)
    segment.keyframes["reference"] = _keyframe(
        "reference", 5, pose, np.empty((0, 2)), np.empty((0, 32), dtype=np.uint8),
    )
    moved = np.eye(4)
    moved[:3, 3] = [0.5, 0.0, 0.0]
    assert not decide_keyframe(segment, 6, moved, 5, 60).insert
    assert decide_keyframe(segment, 8, moved, 5, 60).insert
    assert not decide_keyframe(segment, 8, moved, 55, 60).insert


def test_rejected_pose_cannot_create_mapping_keyframe() -> None:
    segment = MapSegment("segment-map", "test", MapSegmentStatus.ACTIVE, True)
    empty = Features((), None)
    rejected = estimate_map_pose(segment, empty, (480, 640, 3), CAMERA, np.eye(4))
    with pytest.raises(ValueError, match="rejected"):
        create_mapping_keyframe(
            segment, "bad-frame", 7, np.eye(4), empty, rejected, "must fail",
        )


def test_mapping_growth_is_bounded_and_spatially_distributed() -> None:
    rng = np.random.default_rng(61)
    points = np.column_stack([
        rng.uniform(-2.5, 2.5, 80), rng.uniform(-1.2, 1.2, 80), rng.uniform(5, 10, 80),
    ])
    first_pose = np.eye(4)
    second_pose = np.eye(4)
    second_pose[0, 3] = 1.0
    first_pixels = _project(points, first_pose)
    second_pixels = _project(points, second_pose)
    descriptors = rng.integers(0, 256, (80, 32), dtype=np.uint8)
    first = _keyframe("bounded-first", 0, first_pose, first_pixels, descriptors)
    second = _keyframe("bounded-second", 5, second_pose, second_pixels, descriptors.copy())
    segment = MapSegment("segment-map", "test", MapSegmentStatus.ACTIVE, True)
    segment.keyframes = {first.keyframe_id: first, second.keyframe_id: second}
    result = replenish_landmarks(
        segment, first, second, CAMERA, (480, 640, 3),
        MappingThresholds(
            min_matches=20, min_new_landmarks=10, min_occupied_grid_cells=4,
            min_triangulation_angle_degrees=0.5,
            max_new_landmarks_per_keyframe=16,
            max_new_landmarks_per_grid_cell=2,
            efficient_candidate_reverse=True,
        ),
    )
    assert result.accepted, result.reason
    assert result.stats.inserted_landmarks <= 16
    assert result.stats.inserted_landmarks >= 10
    assert len(segment.landmarks) == result.stats.inserted_landmarks
    assert not validate_map_integrity(segment)
