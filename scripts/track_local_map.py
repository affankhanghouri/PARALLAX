#!/usr/bin/env python3
"""Initialize one local map and track subsequent frames with landmark PnP."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys
from time import perf_counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.features import ORBFeatures
from src.frame_source import ImageSequence
from src.initialization import PairSelectionThresholds, TwoViewPairSelector
from src.local_map import (
    FramePose,
    MapSegmentStatus,
    PoseProvenance,
    create_initialized_segment,
    validate_map_integrity,
)
from src.local_mapping import (
    KeyframePolicy,
    MappingThresholds,
    create_mapping_keyframe,
    decide_keyframe,
    replenish_landmarks,
)
from src.landmark_quality import select_active_landmarks, update_landmark_quality
from src.map_tracking import (
    OpticalFlowDiagnostics,
    PnPThresholds,
    estimate_map_pose,
    track_landmarks_optical_flow,
    update_landmark_references,
    update_segment_with_tracking,
)
from src.reconstruction import TriangulationThresholds, triangulate_two_view


LOGGER = logging.getLogger("local_map_tracking")


def _occupied_grid_cells(pixels, image_shape, rows=4, columns=6) -> int:
    pixels = np.asarray(pixels, dtype=float).reshape(-1, 2)
    if not len(pixels):
        return 0
    height, width = image_shape[:2]
    xs = np.clip((pixels[:, 0] / width * columns).astype(int), 0, columns - 1)
    ys = np.clip((pixels[:, 1] / height * rows).astype(int), 0, rows - 1)
    return len(set(zip(ys.tolist(), xs.tolist())))


def _visible_landmarks(segment, pose_camera_to_segment, camera_matrix, image_shape) -> int:
    landmarks = segment.active_landmarks()
    if not landmarks:
        return 0
    world_to_camera = np.linalg.inv(pose_camera_to_segment)
    points = np.asarray([item.position for item in landmarks])
    camera_points = (world_to_camera[:3, :3] @ points.T).T + world_to_camera[:3, 3]
    pixels_h = (camera_matrix @ camera_points.T).T
    pixels = pixels_h[:, :2] / np.maximum(pixels_h[:, 2:3], 1e-12)
    height, width = image_shape[:2]
    visible = (
        (camera_points[:, 2] > 0)
        & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
        & (pixels[:, 1] >= 0) & (pixels[:, 1] < height)
    )
    return int(np.count_nonzero(visible))


def _save_plots(segment, frame_metrics, output: Path) -> None:
    landmarks = np.asarray([item.position for item in segment.active_landmarks()])
    accepted_poses = [item for item in segment.frame_poses if item.accepted and item.pose_camera_to_segment is not None]
    positions = np.asarray([item.pose_camera_to_segment[:3, 3] for item in accepted_poses])
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    if len(landmarks):
        axes[0].scatter(landmarks[:, 0], landmarks[:, 2], s=9, alpha=0.5, label="active map landmarks")
    if len(positions):
        axes[0].plot(positions[:, 0], positions[:, 2], "o-", ms=3, label="accepted camera poses")
    axes[0].set_title("Local map and camera trajectory — arbitrary scale")
    axes[0].set_xlabel("segment X [arbitrary units]")
    axes[0].set_ylabel("segment Z [arbitrary units]")
    axes[0].axis("equal")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    frames = [item["frame_offset"] for item in frame_metrics]
    axes[1].plot(frames, [item["associations"] for item in frame_metrics], label="unique associations")
    axes[1].plot(frames, [item["pnp_inliers"] for item in frame_metrics], label="PnP inliers")
    rejected = [item for item in frame_metrics if not item["accepted"]]
    if rejected:
        axes[1].scatter(
            [item["frame_offset"] for item in rejected],
            [item["associations"] for item in rejected], marker="x", color="red", label="rejected",
        )
    axes[1].set_title("Landmark support")
    axes[1].set_xlabel("local frame offset")
    axes[1].set_ylabel("count")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(output / "local_map_tracking.png", dpi=180)
    plt.close(figure)

    diagnostics, axes = plt.subplots(2, 1, figsize=(11, 8), sharex=True)
    axes[0].plot(frames, [item["reprojection_median_px"] for item in frame_metrics], label="median reprojection")
    axes[0].plot(frames, [item["reprojection_p95_px"] for item in frame_metrics], label="p95 reprojection")
    axes[0].set_ylabel("pixels")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    for key, label in [
        ("extraction_ms", "feature extraction"),
        ("association_ms", "association"),
        ("pnp_ms", "PnP RANSAC + refine"),
        ("validation_ms", "pose validation"),
        ("map_update_ms", "map update"),
    ]:
        axes[1].plot(frames, [item[key] for item in frame_metrics], label=label)
    axes[1].set_xlabel("local frame offset")
    axes[1].set_ylabel("milliseconds")
    axes[1].legend(ncol=2)
    axes[1].grid(alpha=0.3)
    diagnostics.tight_layout()
    diagnostics.savefig(output / "tracking_diagnostics.png", dpi=180)
    plt.close(diagnostics)

    association_figure, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
    for key, label in [
        ("active_landmarks", "active map landmarks"),
        ("predicted_visible_landmarks", "projected in image"),
        ("association_distance_pass", "Hamming distance pass"),
        ("association_ratio_pass", "ratio pass"),
        ("associations", "fused associations"),
        ("pnp_inliers", "PnP inliers"),
    ]:
        axes[0].plot(frames, [item[key] for item in frame_metrics], label=label)
    axes[0].set_ylabel("count")
    axes[0].legend(ncol=3)
    axes[0].grid(alpha=0.3)
    axes[1].plot(
        frames, [item["nearest_descriptor_distance_median"] for item in frame_metrics],
        label="median nearest Hamming distance",
    )
    axes[1].plot(
        frames, [item["nearest_descriptor_distance_p95"] for item in frame_metrics],
        label="P95 nearest Hamming distance",
    )
    axes[1].set_ylabel("Hamming distance")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[2].plot(
        frames, [item["optical_flow_accepted"] for item in frame_metrics],
        label="forward-backward flow tracks",
    )
    axes[2].bar(
        frames, [item["new_landmarks"] for item in frame_metrics],
        alpha=0.45, label="new landmarks",
    )
    axes[2].set_xlabel("local frame offset")
    axes[2].set_ylabel("count")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    association_figure.tight_layout()
    association_figure.savefig(output / "association_diagnostics.png", dpi=180)
    plt.close(association_figure)

    spatial, axis = plt.subplots(figsize=(11, 4.5))
    for key, label in [
        ("feature_grid_cells", "ORB feature cells"),
        ("descriptor_association_grid_cells", "descriptor association cells"),
        ("fused_association_grid_cells", "fused association cells"),
        ("occupied_grid_cells", "PnP inlier cells"),
    ]:
        axis.plot(frames, [item.get(key, 0) for item in frame_metrics], label=label)
    axis.axhline(4, color="black", linestyle="--", alpha=0.6, label="PnP acceptance minimum")
    axis.set_xlabel("local frame offset")
    axis.set_ylabel("occupied cells in 4x6 grid")
    axis.set_ylim(0, 25)
    axis.grid(alpha=0.3)
    axis.legend(ncol=2)
    spatial.tight_layout()
    spatial.savefig(output / "spatial_coverage.png", dpi=180)
    plt.close(spatial)


def run(
    config_path: str | Path,
    *,
    anchor_offset: int = 0,
    max_candidates: int = 30,
    max_tracking_frames: int = 50,
    max_consecutive_failures: int = 3,
    output_dir_override: str | Path | None = None,
) -> dict[str, object]:
    # OpenCV RANSAC otherwise changes keyframe/map-growth decisions between
    # identical benchmark runs, invalidating controlled ablations.
    cv2.setRNGSeed(0)
    config = load_config(config_path)
    output = Path(output_dir_override).expanduser().resolve() if output_dir_override else config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    source = ImageSequence(
        config.image_dir, anchor_offset + max_candidates + max_tracking_frames + 1
    )
    iterator = iter(source)
    anchor = None
    for frame in iterator:
        if frame.index == anchor_offset:
            anchor = frame
            break
    if anchor is None:
        raise ValueError("anchor offset is outside the input sequence")
    frontend = ORBFeatures(config.max_features, config.ratio_test)
    selector = TwoViewPairSelector(
        frontend, config.camera.matrix,
        thresholds=PairSelectionThresholds(
            max(config.initialization_min_matches, config.min_matches),
            config.initialization_min_inliers,
            config.initialization_min_median_parallax_degrees,
            config.initialization_min_p25_parallax_degrees,
            config.initialization_grid_rows,
            config.initialization_grid_columns,
            config.initialization_min_grid_ratio,
        ),
        min_pose_inliers=config.min_inliers,
        ransac_probability=config.ransac_probability,
        ransac_threshold_px=config.ransac_threshold_px,
        rotation_only_residual_degrees=config.rotation_only_residual_degrees,
        rotation_only_min_ratio=config.rotation_only_min_ratio,
        rotation_innovation_degrees=config.rotation_innovation_degrees,
        suspicious_parallax_p10_degrees=config.suspicious_parallax_p10_degrees,
    )
    initialization_started = perf_counter()
    selection = selector.select(anchor, iterator)
    if selection.selected is None:
        raise ValueError("no suitable initialization pair found")
    selected = selection.selected
    reconstruction = triangulate_two_view(
        selected.points_anchor, selected.points_candidate, config.camera.matrix,
        selected.estimate.rotation, selected.estimate.translation,
        TriangulationThresholds(
            max_reprojection_error_px=config.triangulation_max_reprojection_error_px,
            min_triangulation_angle_degrees=config.triangulation_min_angle_degrees,
            max_depth=config.triangulation_max_depth,
        ),
    )
    segment = create_initialized_segment(selected, reconstruction)
    initial_landmark_count = len(segment.landmarks)
    initialization_ms = (perf_counter() - initialization_started) * 1000.0
    thresholds = PnPThresholds(
        ratio_test=config.ratio_test,
        max_descriptor_distance=config.pnp_max_descriptor_distance,
        min_correspondences=config.pnp_min_correspondences,
        min_inliers=config.pnp_min_inliers,
        min_inlier_ratio=config.pnp_min_inlier_ratio,
        ransac_reprojection_error_px=config.pnp_ransac_reprojection_error_px,
        max_median_reprojection_error_px=config.pnp_max_median_reprojection_error_px,
        max_p95_reprojection_error_px=config.pnp_max_p95_reprojection_error_px,
        min_occupied_grid_cells=config.pnp_min_occupied_grid_cells,
        max_rotation_jump_degrees=config.pnp_max_rotation_jump_degrees,
        max_translation_jump_arbitrary=config.pnp_max_translation_jump_arbitrary,
    )
    previous_pose = segment.frame_poses[-1].pose_camera_to_segment
    assert previous_pose is not None
    previous_image = selected.candidate.image
    assert previous_image is not None
    previous_landmark_pixels = {
        landmark.landmark_id: landmark.observations[-1].pixel.copy()
        for landmark in segment.active_landmarks()
    }
    frame_metrics: list[dict[str, object]] = []
    mapping_events: list[dict[str, object]] = []
    consecutive_failures = 0
    for frame_number, frame in enumerate(iterator):
        if frame_number >= max_tracking_frames:
            break
        frame_started = perf_counter()
        if frame.image is None:
            segment.status = MapSegmentStatus.INTERRUPTED
            segment.trajectory_continuous = False
            frame_id = f"{segment.segment_id}:frame-{frame.index:06d}"
            segment.frame_poses.append(FramePose(
                frame_id, frame.index, None, False, "unreadable frame", PoseProvenance.MAP_PNP,
            ))
            frame_metrics.append({
                "frame_offset": frame.index, "accepted": False,
                "tracking_state": "REJECTED",
                "pose_provenance": PoseProvenance.MAP_PNP.value,
                "reason": "unreadable frame", "features": 0,
                "predicted_visible_landmarks": 0, "associations": 0,
                "association_distance_pass": 0, "association_ratio_pass": 0,
                "association_consistency_pass": 0,
                "nearest_descriptor_distance_median": None,
                "nearest_descriptor_distance_p95": None,
                "nearest_descriptor_ratio_median": None,
                "pnp_inliers": 0, "inlier_ratio": 0.0,
                "reprojection_median_px": None, "reprojection_p95_px": None,
                "occupied_grid_cells": 0, "rotation_jump_degrees": None,
                "translation_jump_arbitrary": None, "extraction_ms": 0.0,
                "association_ms": 0.0, "pnp_ms": 0.0, "validation_ms": 0.0,
                "optical_flow_ms": 0.0, "optical_flow_attempted": 0,
                "optical_flow_accepted": 0,
                "optical_flow_forward_valid": 0,
                "optical_flow_backward_valid": 0,
                "optical_flow_forward_backward_pass": 0,
                "optical_flow_lk_error_pass": 0,
                "optical_flow_projection_pass": 0,
                "optical_flow_forward_backward_median_px": None,
                "map_update_ms": 0.0, "mapping_ms": 0.0,
                "descriptor_references_updated": 0, "keyframe_inserted": False,
                "keyframe_reason": "unreadable frame", "new_landmarks": 0,
                "active_landmarks": len(segment.active_landmarks()), "total_ms": 0.0,
            })
            consecutive_failures += 1
            continue
        visible = _visible_landmarks(segment, previous_pose, config.camera.matrix, frame.image.shape)
        quality_update = None
        visible_landmark_ids: set[str] = set()
        quality_started = perf_counter()
        if config.landmark_quality_enabled:
            quality_update, visible_landmark_ids = select_active_landmarks(
                segment, previous_pose, config.camera.matrix, frame.image.shape,
                max_active=config.landmark_max_active,
                max_visible_misses=config.landmark_max_visible_misses,
            )
            visible = quality_update.predicted_visible
        landmark_quality_ms = (perf_counter() - quality_started) * 1000.0
        extraction_started = perf_counter()
        features = frontend.extract(frame.image)
        extraction_ms = (perf_counter() - extraction_started) * 1000.0
        optical_flow_started = perf_counter()
        flow_associations = []
        flow_diagnostics = OpticalFlowDiagnostics(0, 0, 0, 0, 0, 0, 0, None)
        if config.optical_flow_enabled:
            flow_associations, flow_diagnostics = track_landmarks_optical_flow(
                segment, previous_image, frame.image, previous_landmark_pixels,
                previous_pose, config.camera.matrix,
                projection_search_radius_px=config.projection_search_radius_px,
                max_forward_backward_error_px=config.optical_flow_max_forward_backward_error_px,
                max_lk_error=config.optical_flow_max_lk_error,
            )
        optical_flow_ms = (perf_counter() - optical_flow_started) * 1000.0
        result = estimate_map_pose(
            segment, features, frame.image.shape, config.camera.matrix,
            previous_pose, thresholds=thresholds,
            association_mode=config.association_mode,
            projection_search_radius_px=config.projection_search_radius_px,
            descriptor_reference_limit=config.association_descriptor_reference_limit,
            supplemental_associations=tuple(flow_associations),
        )
        map_update_started = perf_counter()
        frame_id = f"{segment.segment_id}:frame-{frame.index:06d}"
        update_segment_with_tracking(segment, frame_id, frame.index, result)
        if config.landmark_quality_enabled:
            update_landmark_quality(segment, result, visible_landmark_ids, frame.index)
        map_update_ms = (perf_counter() - map_update_started) * 1000.0
        descriptor_updates = 0
        keyframe_inserted = False
        keyframe_reason = "mapping disabled" if not config.mapping_enabled else "pose rejected"
        new_landmarks = 0
        mapping_ms = 0.0
        mapping_descriptor_matching_ms = 0.0
        triangulation_ms = 0.0
        keyframe_selection_ms = 0.0
        if result.accepted:
            assert result.pose_camera_to_segment is not None
            previous_pose = result.pose_camera_to_segment
            consecutive_failures = 0
            previous_image = frame.image
            previous_landmark_pixels = {
                result.associations[int(index)].landmark_id:
                result.associations[int(index)].pixel.copy()
                for index in result.inlier_association_indices
            }
            descriptor_updates = update_landmark_references(
                segment, result, features, frame_id,
                policy=config.descriptor_update_policy,
                max_bank_size=config.descriptor_bank_max_size,
            )
            if config.mapping_enabled:
                keyframe_started = perf_counter()
                decision = decide_keyframe(
                    segment, frame.index, result.pose_camera_to_segment,
                    result.ransac_inliers, len(segment.active_landmarks()),
                    KeyframePolicy(
                        config.keyframe_min_frame_gap,
                        config.keyframe_max_frame_gap,
                        config.keyframe_min_translation_arbitrary,
                        config.keyframe_min_rotation_degrees,
                        config.keyframe_support_ratio_trigger,
                        config.keyframe_critical_support_ratio,
                        config.keyframe_critical_min_translation_arbitrary,
                        config.keyframe_critical_min_frame_gap,
                    ),
                )
                keyframe_selection_ms = (perf_counter() - keyframe_started) * 1000.0
                keyframe_reason = decision.reason
                if decision.insert:
                    reference = max(
                        segment.keyframes.values(), key=lambda item: item.source_frame_index,
                    )
                    current = create_mapping_keyframe(
                        segment, frame_id, frame.index, result.pose_camera_to_segment,
                        features, result, decision.reason,
                    )
                    mapping_result = replenish_landmarks(
                        segment, reference, current, config.camera.matrix, frame.image.shape,
                        MappingThresholds(
                            ratio_test=config.ratio_test,
                            max_descriptor_distance=config.pnp_max_descriptor_distance,
                            min_matches=config.mapping_min_matches,
                            epipolar_threshold_px=config.mapping_epipolar_threshold_px,
                            max_reprojection_error_px=config.mapping_max_reprojection_error_px,
                            min_triangulation_angle_degrees=config.mapping_min_triangulation_angle_degrees,
                            max_depth_arbitrary=config.mapping_max_depth_arbitrary,
                            min_new_landmarks=config.mapping_min_new_landmarks,
                            min_occupied_grid_cells=config.mapping_min_occupied_grid_cells,
                            duplicate_radius_px=config.mapping_duplicate_radius_px,
                            max_landmarks=config.mapping_max_landmarks,
                            max_new_landmarks_per_keyframe=config.mapping_max_new_landmarks_per_keyframe,
                            max_new_landmarks_per_grid_cell=config.mapping_max_new_landmarks_per_grid_cell,
                            efficient_candidate_reverse=config.mapping_efficient_candidate_reverse,
                        ),
                    )
                    keyframe_inserted = True
                    mapping_ms = mapping_result.processing_ms
                    mapping_descriptor_matching_ms = mapping_result.descriptor_matching_ms
                    triangulation_ms = mapping_result.triangulation_ms
                    new_landmarks = mapping_result.stats.inserted_landmarks
                    # New geometry did not participate in the accepted pose.
                    # Seed its validated current observations only for the
                    # following frame's temporal tracking.
                    for feature_index, landmark_id in current.landmark_observations.items():
                        previous_landmark_pixels[landmark_id] = np.asarray(
                            features.keypoints[feature_index].pt, dtype=np.float64,
                        )
                    mapping_events.append({
                        "frame_offset": frame.index,
                        "keyframe_decision": asdict(decision),
                        **mapping_result.summary(),
                    })
        else:
            consecutive_failures += 1
        association = result.association_diagnostics
        feature_pixels = np.asarray([item.pt for item in features.keypoints], dtype=float)
        descriptor_pixels = [item.pixel for item in result.associations if item.source == "DESCRIPTOR"]
        fused_pixels = [item.pixel for item in result.associations]
        frame_metrics.append({
            "frame_offset": frame.index, "accepted": result.accepted,
            "tracking_state": "TRACKED" if result.accepted else "REJECTED",
            "pose_provenance": PoseProvenance.MAP_PNP.value,
            "reason": result.reason, "features": len(features.keypoints),
            "predicted_visible_landmarks": visible,
            "associations": len(result.associations),
            "association_distance_pass": association.distance_pass,
            "association_ratio_pass": association.ratio_pass,
            "association_consistency_pass": association.consistency_pass,
            "landmark_projection_ms": association.projection_ms,
            "descriptor_matching_ms": association.descriptor_matching_ms,
            "nearest_descriptor_distance_median": association.nearest_distance_median,
            "nearest_descriptor_distance_p95": association.nearest_distance_p95,
            "nearest_descriptor_ratio_median": association.nearest_ratio_median,
            "pnp_inliers": result.ransac_inliers,
            "inlier_ratio": result.inlier_ratio,
            "reprojection_median_px": result.reprojection_median_px,
            "reprojection_p95_px": result.reprojection_p95_px,
            "occupied_grid_cells": result.occupied_grid_cells,
            "feature_grid_cells": _occupied_grid_cells(feature_pixels, frame.image.shape),
            "descriptor_association_grid_cells": _occupied_grid_cells(descriptor_pixels, frame.image.shape),
            "fused_association_grid_cells": _occupied_grid_cells(fused_pixels, frame.image.shape),
            "rotation_jump_degrees": result.rotation_jump_degrees,
            "translation_jump_arbitrary": result.translation_jump_arbitrary,
            "extraction_ms": extraction_ms,
            "association_ms": result.association_ms,
            "optical_flow_ms": optical_flow_ms,
            "optical_flow_attempted": flow_diagnostics.attempted,
            "optical_flow_accepted": flow_diagnostics.accepted,
            "optical_flow_forward_valid": flow_diagnostics.forward_valid,
            "optical_flow_backward_valid": flow_diagnostics.backward_valid,
            "optical_flow_forward_backward_pass": flow_diagnostics.forward_backward_pass,
            "optical_flow_lk_error_pass": flow_diagnostics.lk_error_pass,
            "optical_flow_projection_pass": flow_diagnostics.projection_pass,
            "optical_flow_forward_backward_median_px": flow_diagnostics.forward_backward_median_px,
            "pnp_ms": result.pnp_ms,
            "validation_ms": result.validation_ms,
            "map_update_ms": map_update_ms,
            "mapping_ms": mapping_ms,
            "mapping_descriptor_matching_ms": mapping_descriptor_matching_ms,
            "triangulation_ms": triangulation_ms,
            "keyframe_selection_ms": keyframe_selection_ms,
            "landmark_quality_ms": landmark_quality_ms,
            "descriptor_references_updated": descriptor_updates,
            "keyframe_inserted": keyframe_inserted,
            "keyframe_reason": keyframe_reason,
            "new_landmarks": new_landmarks,
            "active_landmarks": len(segment.active_landmarks()),
            "quality_eligible_landmarks": quality_update.eligible if quality_update else len(segment.landmarks),
            "quality_temporarily_inactive": quality_update.temporarily_inactive if quality_update else 0,
            "quality_deactivated_for_budget": quality_update.deactivated_for_budget if quality_update else 0,
            "quality_deactivated_for_misses": quality_update.deactivated_for_misses if quality_update else 0,
            "quality_reactivated": quality_update.reactivated if quality_update else 0,
            "total_ms": (perf_counter() - frame_started) * 1000.0,
        })
        if consecutive_failures >= max_consecutive_failures:
            segment.status = MapSegmentStatus.LOST
            break

    accepted = sum(bool(item["accepted"]) for item in frame_metrics)
    totals = np.asarray([item["total_ms"] for item in frame_metrics], dtype=float)
    accepted_items = [item for item in frame_metrics if item["accepted"]]
    pnp_executed = [item["pnp_ms"] for item in frame_metrics if item["pnp_ms"] > 0]
    summary = {
        "attempted_tracking_frames": len(frame_metrics),
        "accepted_tracking_frames": accepted,
        "rejected_tracking_frames": len(frame_metrics) - accepted,
        "tracking_success_rate": accepted / max(len(frame_metrics), 1),
        "steady_state_fps": float(1000.0 / np.mean(totals)) if len(totals) else 0.0,
        "latency_mean_ms": float(np.mean(totals)) if len(totals) else 0.0,
        "latency_p95_ms": float(np.percentile(totals, 95)) if len(totals) else 0.0,
        "feature_extraction_mean_ms": float(np.mean([item["extraction_ms"] for item in frame_metrics])) if frame_metrics else 0.0,
        "association_mean_ms": float(np.mean([item["association_ms"] for item in frame_metrics])) if frame_metrics else 0.0,
        "landmark_projection_mean_ms": float(np.mean([item.get("landmark_projection_ms", 0.0) for item in frame_metrics])) if frame_metrics else 0.0,
        "descriptor_matching_mean_ms": float(np.mean([item.get("descriptor_matching_ms", 0.0) for item in frame_metrics])) if frame_metrics else 0.0,
        "optical_flow_mean_ms": float(np.mean([item["optical_flow_ms"] for item in frame_metrics])) if frame_metrics else 0.0,
        "pnp_mean_ms": float(np.mean([item["pnp_ms"] for item in frame_metrics])) if frame_metrics else 0.0,
        "validation_mean_ms": float(np.mean([item["validation_ms"] for item in frame_metrics])) if frame_metrics else 0.0,
        "map_update_mean_ms": float(np.mean([item["map_update_ms"] for item in frame_metrics])) if frame_metrics else 0.0,
        "mapping_mean_ms": float(np.mean([item["mapping_ms"] for item in frame_metrics])) if frame_metrics else 0.0,
        "mapping_descriptor_matching_mean_ms": float(np.mean([item.get("mapping_descriptor_matching_ms", 0.0) for item in frame_metrics])) if frame_metrics else 0.0,
        "triangulation_mean_ms": float(np.mean([item.get("triangulation_ms", 0.0) for item in frame_metrics])) if frame_metrics else 0.0,
        "keyframe_selection_mean_ms": float(np.mean([item.get("keyframe_selection_ms", 0.0) for item in frame_metrics])) if frame_metrics else 0.0,
        "landmark_quality_mean_ms": float(np.mean([item.get("landmark_quality_ms", 0.0) for item in frame_metrics])) if frame_metrics else 0.0,
        "pnp_executed_mean_ms": float(np.mean(pnp_executed)) if pnp_executed else 0.0,
        "associations_mean": float(np.mean([item["associations"] for item in frame_metrics])) if frame_metrics else 0.0,
        "accepted_pnp_inliers_mean": float(np.mean([item["pnp_inliers"] for item in accepted_items])) if accepted_items else 0.0,
        "first_accepted_tracking_offset": int(accepted_items[0]["frame_offset"]) if accepted_items else None,
        "last_accepted_tracking_offset": int(accepted_items[-1]["frame_offset"]) if accepted_items else None,
    }
    observation_counts = [len(item.observations) for item in segment.landmarks.values()]
    landmark_batches = []
    for created_index in sorted({item.created_frame_index for item in segment.landmarks.values()}):
        batch = [item for item in segment.landmarks.values() if item.created_frame_index == created_index]
        landmark_batches.append({
            "created_frame_offset": created_index,
            "landmarks": len(batch),
            "reobserved_after_creation": sum(len(item.observations) > 2 for item in batch),
            "pnp_inlier_observations_total": sum(item.pnp_inlier_observations for item in batch),
            "active_at_end": sum(item.status.value == "ACTIVE" for item in batch),
            "temporarily_inactive_at_end": sum(item.status.value == "TEMPORARILY_INACTIVE" for item in batch),
        })
    integrity_errors = validate_map_integrity(segment)
    report = {
        "scale": "arbitrary local-map units; not meters",
        "coordinate_convention": "solvePnP returns segment-to-camera; stored pose is its camera-to-segment inverse",
        "initialization": {
            "anchor_offset": selected.anchor.index,
            "candidate_offset": selected.candidate.index,
            "landmarks": initial_landmark_count,
            "pose_provenance": PoseProvenance.TWO_VIEW_INITIALIZATION.value,
            "initialization_ms_excluded_from_tracking_fps": initialization_ms,
        },
        "segment": {
            "segment_id": segment.segment_id,
            "status": segment.status.value,
            "trajectory_continuous": segment.trajectory_continuous,
            "keyframes": len(segment.keyframes),
            "landmarks": len(segment.landmarks),
            "new_landmarks": len(segment.landmarks) - initial_landmark_count,
            "landmark_observations_min": min(observation_counts) if observation_counts else 0,
            "landmark_observations_median": float(np.median(observation_counts)) if observation_counts else 0.0,
            "landmark_observations_max": max(observation_counts) if observation_counts else 0,
            "integrity_valid": not integrity_errors,
            "integrity_errors": list(integrity_errors),
            "landmark_batches": landmark_batches,
        },
        "summary": summary,
        "runtime_policy": {
            "association_mode": config.association_mode,
            "projection_search_radius_px": config.projection_search_radius_px,
            "association_descriptor_reference_limit": config.association_descriptor_reference_limit,
            "descriptor_update_policy": config.descriptor_update_policy,
            "descriptor_bank_max_size": config.descriptor_bank_max_size,
            "optical_flow_enabled": config.optical_flow_enabled,
            "mapping_enabled": config.mapping_enabled,
            "landmark_quality_enabled": config.landmark_quality_enabled,
            "landmark_max_active": config.landmark_max_active,
            "landmark_max_visible_misses": config.landmark_max_visible_misses,
            "mapping_max_new_landmarks_per_keyframe": config.mapping_max_new_landmarks_per_keyframe,
            "mapping_max_new_landmarks_per_grid_cell": config.mapping_max_new_landmarks_per_grid_cell,
            "mapping_efficient_candidate_reverse": config.mapping_efficient_candidate_reverse,
        },
        "mapping_events": mapping_events,
        "frames": frame_metrics,
    }
    (output / "tracking_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    accepted_poses = [item for item in segment.frame_poses if item.accepted and item.pose_camera_to_segment is not None]
    trajectory = np.asarray([
        [item.source_frame_index, *item.pose_camera_to_segment[:3, 3]] for item in accepted_poses
    ])
    np.savetxt(
        output / "map_trajectory.csv", trajectory, delimiter=",",
        header="frame_offset,segment_x_arbitrary,segment_y_arbitrary,segment_z_arbitrary", comments="",
    )
    poses = np.asarray([
        [item.source_frame_index, *item.pose_camera_to_segment.reshape(-1)] for item in accepted_poses
    ])
    np.savetxt(
        output / "map_poses_camera_to_segment.csv", poses, delimiter=",",
        header="frame_offset," + ",".join(f"t{r}{c}" for r in range(4) for c in range(4)), comments="",
    )
    landmark_rows = np.asarray([
        [index, *item.position, len(item.observations), len(item.descriptor_bank),
         item.projection_attempts, item.association_successes, item.pnp_inlier_observations,
         item.consecutive_visible_misses]
        for index, item in enumerate(segment.landmarks.values())
    ], dtype=float)
    np.savetxt(
        output / "map_landmarks.csv", landmark_rows, delimiter=",",
        header="landmark_index,segment_x_arbitrary,segment_y_arbitrary,segment_z_arbitrary,observations,descriptor_references,projection_attempts,association_successes,pnp_inlier_observations,consecutive_visible_misses",
        comments="",
    )
    _save_plots(segment, frame_metrics, output)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--anchor-offset", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=30)
    parser.add_argument("--max-tracking-frames", type=int, default=50)
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--output-dir", help="override configured output directory")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        report = run(
            args.config, anchor_offset=args.anchor_offset,
            max_candidates=args.max_candidates,
            max_tracking_frames=args.max_tracking_frames,
            max_consecutive_failures=args.max_consecutive_failures,
            output_dir_override=args.output_dir,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        LOGGER.error("Cannot track local map: %s", exc)
        return 2
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
