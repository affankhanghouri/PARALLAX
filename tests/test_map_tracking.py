from pathlib import Path

import cv2
import numpy as np

from src.features import Features
from src.frame_source import Frame
from src.geometry import MotionState, PoseEstimate
from src.initialization import PairAssessment, SelectedPair
from src.local_map import (
    Landmark,
    LandmarkQuality,
    MapSegment,
    MapSegmentStatus,
    PoseProvenance,
    create_initialized_segment,
    validate_map_integrity,
)
from src.landmark_quality import select_active_landmarks, update_landmark_quality
from src.map_tracking import (
    PnPThresholds,
    TrackingResult,
    associate_landmarks,
    estimate_map_pose,
    track_landmarks_optical_flow,
    update_landmark_references,
    update_segment_with_tracking,
)
from src.reconstruction import triangulate_two_view


CAMERA = np.array([[700.0, 0, 320], [0, 700.0, 240], [0, 0, 1]])


def _features(pixels: np.ndarray, descriptors: np.ndarray) -> Features:
    keypoints = tuple(cv2.KeyPoint(float(x), float(y), 5) for x, y in pixels)
    return Features(keypoints, descriptors.astype(np.uint8))


def _tracking_fixture(count: int = 40):
    rng = np.random.default_rng(40)
    points = np.column_stack([
        rng.uniform(-2.5, 2.5, count), rng.uniform(-1.5, 1.5, count), rng.uniform(5, 12, count)
    ])
    descriptors = rng.integers(0, 256, (count, 32), dtype=np.uint8)
    quality = LandmarkQuality(3.0, 0.2, 0.2, 6.0, 5.5)
    segment = MapSegment("segment-test", "test frame", MapSegmentStatus.ACTIVE, True)
    for index, (point, descriptor) in enumerate(zip(points, descriptors)):
        identifier = f"segment-test:landmark-{index:06d}"
        segment.landmarks[identifier] = Landmark(
            identifier, segment.segment_id, point, descriptor[None, :], ("reference",), quality,
        )
    rotation, _ = cv2.Rodrigues(np.array([0.015, -0.03, 0.008]))
    camera_center = np.array([0.25, -0.02, -0.35])
    translation = -rotation @ camera_center
    camera_points = (rotation @ points.T).T + translation
    pixels_h = (CAMERA @ camera_points.T).T
    pixels = pixels_h[:, :2] / pixels_h[:, 2:]
    true_pose = np.eye(4)
    true_pose[:3, :3] = rotation.T
    true_pose[:3, 3] = camera_center
    return segment, _features(pixels, descriptors), true_pose


def test_pnp_recovers_camera_to_world_pose_and_inverts_transform() -> None:
    segment, features, true_pose = _tracking_fixture()
    result = estimate_map_pose(segment, features, (480, 640, 3), CAMERA, np.eye(4))
    assert result.accepted, result.reason
    assert result.ransac_inliers == 40
    assert np.allclose(result.pose_camera_to_segment, true_pose, atol=1e-5)


def test_associations_are_unique_and_ambiguous_duplicate_is_removed() -> None:
    segment, features, _ = _tracking_fixture(10)
    landmarks = segment.active_landmarks()
    landmarks[1].descriptor_bank = landmarks[0].descriptor_bank.copy()
    associations = associate_landmarks(
        landmarks, features, ratio_test=0.75, max_descriptor_distance=64,
    )
    feature_indices = [item.feature_index for item in associations]
    landmark_ids = [item.landmark_id for item in associations]
    assert len(feature_indices) == len(set(feature_indices))
    assert len(landmark_ids) == len(set(landmark_ids))
    assert sum(item.feature_index == 0 for item in associations) <= 1


def test_projected_association_uses_pose_gate_without_losing_true_matches() -> None:
    segment, features, true_pose = _tracking_fixture()
    pixels = np.asarray([item.pt for item in features.keypoints])
    rng = np.random.default_rng(42)
    features = _features(
        np.vstack([pixels, pixels + np.array([5.0, 0.0])]),
        np.vstack([features.descriptors, rng.integers(0, 256, features.descriptors.shape, dtype=np.uint8)]),
    )
    associations = associate_landmarks(
        segment.active_landmarks(), features,
        ratio_test=0.75, max_descriptor_distance=64,
        mode="projected", pose_camera_to_segment=true_pose,
        camera_matrix=CAMERA, image_shape=(480, 640, 3),
        projection_search_radius_px=10.0,
    )
    assert len(associations) == 40
    assert len({item.landmark_id for item in associations}) == 40
    assert len({item.feature_index for item in associations}) == 40


def test_landmark_quality_updates_and_active_state_is_reversible() -> None:
    segment, features, true_pose = _tracking_fixture()
    result = estimate_map_pose(segment, features, (480, 640, 3), CAMERA, np.eye(4))
    assert result.accepted
    update, visible = select_active_landmarks(
        segment, true_pose, CAMERA, (480, 640, 3),
        max_active=12, max_visible_misses=3,
    )
    assert update.selected_active == 12
    assert update.temporarily_inactive == 28
    update_landmark_quality(segment, result, visible, 7)
    assert sum(item.projection_attempts for item in segment.landmarks.values()) == 40
    assert sum(item.pnp_inlier_observations for item in segment.landmarks.values()) == 40
    victim = next(iter(segment.landmarks.values()))
    victim.consecutive_visible_misses = 3
    select_active_landmarks(
        segment, true_pose, CAMERA, (480, 640, 3),
        max_active=40, max_visible_misses=3,
    )
    assert victim.status.value == "TEMPORARILY_INACTIVE"
    victim.consecutive_visible_misses = 0
    update, _ = select_active_landmarks(
        segment, true_pose, CAMERA, (480, 640, 3),
        max_active=40, max_visible_misses=3,
    )
    assert victim.status.value == "ACTIVE"
    assert update.reactivated >= 1
    assert not validate_map_integrity(segment)


def test_descriptor_bank_updates_only_after_accepted_pose_and_stays_bounded() -> None:
    segment, features, _ = _tracking_fixture()
    result = estimate_map_pose(segment, features, (480, 640, 3), CAMERA, np.eye(4))
    assert result.accepted
    updated = update_landmark_references(
        segment, result, features, "accepted-a",
        policy="bounded_recent", max_bank_size=2,
    )
    assert updated == 40
    assert all(len(item.descriptor_bank) == 2 for item in segment.landmarks.values())
    update_landmark_references(
        segment, result, features, "accepted-b",
        policy="bounded_recent", max_bank_size=2,
    )
    assert all(len(item.descriptor_bank) == 2 for item in segment.landmarks.values())
    assert all(item.reference_frame_ids[-1] == "accepted-b" for item in segment.landmarks.values())


def test_forward_backward_optical_flow_preserves_landmark_identity() -> None:
    camera = np.array([[100.0, 0, 100], [0, 100.0, 100], [0, 0, 1]])
    previous = np.zeros((200, 200), dtype=np.uint8)
    current = np.zeros_like(previous)
    pixels = np.asarray([(x, y) for y in (50, 80, 110, 140) for x in (50, 80, 110, 140)], dtype=float)
    segment = MapSegment("flow", "test", MapSegmentStatus.ACTIVE, True)
    quality = LandmarkQuality(3.0, 0.1, 0.1, 5.0, 5.0)
    previous_pixels = {}
    for index, pixel in enumerate(pixels):
        cv2.circle(previous, tuple(pixel.astype(int)), 4, 255, -1)
        shifted = pixel + np.array([2.0, 1.0])
        cv2.circle(current, tuple(shifted.astype(int)), 4, 255, -1)
        landmark_id = f"flow:landmark-{index}"
        point = np.array([(pixel[0] - 100) / 20, (pixel[1] - 100) / 20, 5.0])
        segment.landmarks[landmark_id] = Landmark(
            landmark_id, segment.segment_id, point,
            np.zeros((1, 32), dtype=np.uint8), ("reference",), quality,
        )
        previous_pixels[landmark_id] = pixel
    associations, diagnostics = track_landmarks_optical_flow(
        segment, previous, current, previous_pixels, np.eye(4), camera,
        projection_search_radius_px=10.0,
        max_forward_backward_error_px=0.5, max_lk_error=20.0,
    )
    assert diagnostics.accepted >= 12
    assert len({item.landmark_id for item in associations}) == len(associations)
    observed_shift = np.median([
        item.pixel - previous_pixels[item.landmark_id] for item in associations
    ], axis=0)
    assert np.allclose(observed_shift, [2.0, 1.0], atol=0.2)


def test_insufficient_correspondences_reject_pose() -> None:
    segment, features, _ = _tracking_fixture(6)
    result = estimate_map_pose(segment, features, (480, 640, 3), CAMERA, np.eye(4))
    assert not result.accepted
    assert result.pose_camera_to_segment is None
    assert "only 6" in result.reason


def test_incorrect_correspondences_are_rejected() -> None:
    segment, features, _ = _tracking_fixture()
    shifted = np.roll(np.asarray([keypoint.pt for keypoint in features.keypoints]), 13, axis=0)
    bad_features = _features(shifted, features.descriptors)
    result = estimate_map_pose(segment, bad_features, (480, 640, 3), CAMERA, np.eye(4))
    assert not result.accepted
    assert result.pose_camera_to_segment is None


def test_spatially_concentrated_pose_is_rejected() -> None:
    segment, features, _ = _tracking_fixture()
    result = estimate_map_pose(segment, features, (5000, 5000, 3), CAMERA, np.eye(4))
    assert not result.accepted
    assert "occupied image grid cells" in result.reason


def test_tracking_interruption_records_no_fictional_pose() -> None:
    segment, features, _ = _tracking_fixture(6)
    result = estimate_map_pose(segment, features, (480, 640, 3), CAMERA, np.eye(4))
    update_segment_with_tracking(segment, "frame-bad", 7, result)
    assert segment.status == MapSegmentStatus.INTERRUPTED
    assert not segment.trajectory_continuous
    rejected = segment.frame_poses[-1]
    assert rejected.pose_camera_to_segment is None
    assert not rejected.accepted
    assert rejected.provenance == PoseProvenance.MAP_PNP


def test_two_view_map_ids_and_observations_are_consistent() -> None:
    rng = np.random.default_rng(41)
    points = np.column_stack([
        rng.uniform(-2, 2, 30), rng.uniform(-1, 1, 30), rng.uniform(5, 10, 30)
    ])
    rotation = np.eye(3)
    translation = np.array([1.0, 0.0, 0.0])

    def project(value, trans):
        camera_points = value + trans
        pixels = (CAMERA @ camera_points.T).T
        return pixels[:, :2] / pixels[:, 2:]

    first = project(points, np.zeros(3))
    second = project(points, translation)
    descriptors_first = rng.integers(0, 256, (30, 32), dtype=np.uint8)
    descriptors_second = descriptors_first.copy()
    feature_first = _features(first, descriptors_first)
    feature_second = _features(second, descriptors_second)
    estimate = PoseEstimate(
        True, rotation, translation, np.ones(30, dtype=bool), 30,
        state=MotionState.VALID_MOTION,
    )
    assessment = PairAssessment(
        0, 1, True, MotionState.VALID_MOTION, (), 30, 0.0, 30,
        5.0, 3.0, 12, 24, 0.5, 1.0,
    )
    selected = SelectedPair(
        Frame(0, Path("first.png"), np.zeros((480, 640, 3), dtype=np.uint8)),
        Frame(1, Path("second.png"), np.zeros((480, 640, 3), dtype=np.uint8)),
        assessment, estimate, first, second, np.arange(30), np.arange(30),
        feature_first, feature_second,
    )
    reconstruction = triangulate_two_view(first, second, CAMERA, rotation, translation)
    segment = create_initialized_segment(selected, reconstruction, segment_id="segment-unit")
    assert segment.landmarks
    for landmark in segment.landmarks.values():
        assert landmark.segment_id == segment.segment_id
        assert landmark.landmark_id.startswith("segment-unit:landmark-")
        assert len(landmark.observations) == 2
        assert all(observation.landmark_id == landmark.landmark_id for observation in landmark.observations)
    assert not validate_map_integrity(segment)
