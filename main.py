#!/usr/bin/env python3
"""Command-line entry point for classical monocular visual odometry."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import logging
from pathlib import Path
from time import perf_counter

import numpy as np

from src.config import load_config
from src.features import ORBFeatures, matched_points
from src.frame_source import ImageSequence
from src.geometry import MotionState, PoseEstimate, estimate_relative_pose, estimate_reliable_motion
from src.metrics import Metrics, PairMetric
from src.odometry import VisualOdometry
from src.visualization import save_feature_matches, save_trajectory


LOGGER = logging.getLogger("visual_odometry")


def run(config_path: str | Path) -> tuple[VisualOdometry, Metrics]:
    config = load_config(config_path)
    source = ImageSequence(config.image_dir, config.max_frames)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    match_directory = config.output_dir / "matches"
    if config.save_match_every > 0:
        match_directory.mkdir(parents=True, exist_ok=True)

    frontend = ORBFeatures(config.max_features, config.ratio_test)
    odometry = VisualOdometry()
    metrics = Metrics()
    previous_frame = None
    previous_features = None
    pending_extraction_ms = 0.0
    recent_rotations: list[np.ndarray] = []

    LOGGER.info("Processing %d images from %s", len(source), source.directory)
    for frame in source:
        if frame.image is None:
            reason = f"unreadable frame: {frame.path}"
            LOGGER.error("TRACKING LOST: %s", reason)
            odometry.mark_failure()
            metrics.add(PairMetric(
                max(frame.index - 1, 0), 0.0, 0.0, 0.0, 0.0, 0.0,
                0, 0, False, reason, state=MotionState.TRACKING_LOST.value,
            ))
            previous_frame = previous_features = None
            pending_extraction_ms = 0.0
            continue

        extraction_start = perf_counter()
        current_features = frontend.extract(frame.image)
        current_extraction_ms = (perf_counter() - extraction_start) * 1000.0
        if previous_frame is None or previous_features is None:
            previous_frame, previous_features = frame, current_features
            pending_extraction_ms = current_extraction_ms
            continue

        pair_index = frame.index - 1
        extraction_ms = pending_extraction_ms + current_extraction_ms
        pending_extraction_ms = 0.0
        match_start = perf_counter()
        matches = frontend.match(previous_features, current_features)
        matching_ms = (perf_counter() - match_start) * 1000.0

        geometry_start = perf_counter()
        if len(matches) < config.min_matches:
            estimate = PoseEstimate(
                False, None, None, np.zeros(len(matches), dtype=bool), 0,
                f"only {len(matches)} descriptor matches (need {config.min_matches})",
                MotionState.TRACKING_LOST,
            )
        else:
            previous_points, current_points = matched_points(previous_features, current_features, matches)
            if config.geometric_reliability:
                estimate = estimate_reliable_motion(
                    previous_points, current_points, config.camera.matrix,
                    recent_rotations=recent_rotations,
                    min_inliers=config.min_inliers,
                    ransac_probability=config.ransac_probability,
                    ransac_threshold_px=config.ransac_threshold_px,
                    rotation_only_residual_degrees=config.rotation_only_residual_degrees,
                    rotation_only_min_ratio=config.rotation_only_min_ratio,
                    rotation_innovation_degrees=config.rotation_innovation_degrees,
                    suspicious_parallax_p10_degrees=config.suspicious_parallax_p10_degrees,
                )
            else:
                estimate = estimate_relative_pose(
                    previous_points, current_points, config.camera.matrix,
                    min_inliers=config.min_inliers,
                    ransac_probability=config.ransac_probability,
                    ransac_threshold_px=config.ransac_threshold_px,
                )
        geometry_ms = (perf_counter() - geometry_start) * 1000.0
        success = odometry.update(estimate)
        if success and estimate.rotation is not None:
            recent_rotations.append(estimate.rotation)
            recent_rotations = recent_rotations[-config.rotation_history_size:]
        if success:
            if estimate.state == MotionState.ROTATION_ONLY:
                LOGGER.warning(
                    "pair %06d: ROTATION_ONLY; translation not integrated; %s",
                    pair_index, estimate.reason,
                )
            elif odometry.tracking_reliable:
                LOGGER.info("pair %06d: VALID_MOTION matches=%d inliers=%d", pair_index, len(matches), estimate.inlier_count)
            else:
                LOGGER.warning(
                    "pair %06d: LOCAL POSE OK, but accumulated trajectory remains unreliable after earlier loss; "
                    "matches=%d inliers=%d",
                    pair_index, len(matches), estimate.inlier_count,
                )
        else:
            LOGGER.warning("pair %06d: %s; output unreliable; %s", pair_index, estimate.state.value, estimate.reason)

        visualization_ms = 0.0
        if config.save_match_every and pair_index % config.save_match_every == 0:
            visualization_start = perf_counter()
            save_feature_matches(
                previous_frame.image, frame.image, previous_features, current_features,
                matches, estimate.inlier_mask, match_directory / f"pair_{pair_index:06d}.jpg",
            )
            visualization_ms = (perf_counter() - visualization_start) * 1000.0
        core_total_ms = extraction_ms + matching_ms + geometry_ms
        metrics.add(PairMetric(
            pair_index, extraction_ms, matching_ms, geometry_ms, core_total_ms,
            visualization_ms, len(matches), estimate.inlier_count, success, estimate.reason,
            (estimate.rotation.reshape(-1).tolist() if estimate.rotation is not None else None),
            (estimate.translation.reshape(-1).tolist() if estimate.translation is not None else None),
            estimate.state.value,
            (asdict(estimate.diagnostics) if estimate.diagnostics is not None else None),
        ))
        previous_frame, previous_features = frame, current_features

    save_trajectory(odometry.trajectory, config.output_dir / "trajectory.png")
    np.savetxt(
        config.output_dir / "trajectory.csv",
        np.asarray(odometry.trajectory),
        delimiter=",",
        header="world_x_arbitrary,world_y_arbitrary,world_z_arbitrary",
        comments="",
    )
    np.savetxt(
        config.output_dir / "poses_camera_to_world.csv",
        np.asarray(odometry.poses_camera_to_world).reshape(-1, 16),
        delimiter=",",
        header=",".join(f"t{row}{column}" for row in range(4) for column in range(4)),
        comments="",
    )
    metrics.save(config.output_dir)
    LOGGER.info("Summary: %s", metrics.summary())
    return odometry, metrics


def main() -> int:
    parser = argparse.ArgumentParser(description="CPU classical monocular visual odometry")
    parser.add_argument("--config", required=True, help="Path to a YAML configuration file")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        run(args.config)
    except (FileNotFoundError, ValueError, OSError) as exc:
        LOGGER.error("Cannot run visual odometry: %s", exc)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
