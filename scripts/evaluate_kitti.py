#!/usr/bin/env python3
"""Validation-only KITTI trajectory comparison using explicit Sim(3) alignment."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_kitti_poses(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 12:
        raise ValueError(f"Expected KITTI N x 12 poses in {path}, got {values.shape}")
    poses = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], len(values), axis=0)
    poses[:, :3, :] = values.reshape(-1, 3, 4)
    return poses


def load_kitti_positions(path: Path) -> np.ndarray:
    return load_kitti_poses(path)[:, :3, 3]


def rotation_angle_degrees(rotation: np.ndarray) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0))))


def align_similarity(source: np.ndarray, target: np.ndarray) -> tuple[np.ndarray, float, np.ndarray, np.ndarray]:
    """Least-squares Umeyama Sim(3), mapping source points onto target points."""
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3 or len(source) < 3:
        raise ValueError("Similarity alignment requires matching N x 3 arrays with N >= 3")
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    source_centered = source - source_mean
    target_centered = target - target_mean
    covariance = target_centered.T @ source_centered / len(source)
    left, singular_values, right_transpose = np.linalg.svd(covariance)
    sign = np.ones(3)
    if np.linalg.det(left @ right_transpose) < 0:
        sign[-1] = -1
    rotation = left @ np.diag(sign) @ right_transpose
    variance = float(np.mean(np.sum(source_centered * source_centered, axis=1)))
    if variance <= 0:
        raise ValueError("Estimated trajectory has zero variance")
    scale = float(np.dot(singular_values, sign) / variance)
    translation = target_mean - scale * (rotation @ source_mean)
    aligned = (scale * (rotation @ source.T)).T + translation
    return aligned, scale, rotation, translation


def evaluate(output_dir: Path, ground_truth_path: Path, start_index: int = 0) -> dict[str, float | int | str]:
    estimated = np.loadtxt(output_dir / "trajectory.csv", delimiter=",", skiprows=1)
    estimated = np.atleast_2d(estimated)
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    successful_items = [item for item in metrics["pairs"] if item["success"]]
    successful_pairs = [item["pair_index"] for item in successful_items]
    frame_indices = np.asarray([start_index, *(start_index + index + 1 for index in successful_pairs)], dtype=int)
    if len(frame_indices) != len(estimated):
        raise ValueError("Trajectory rows do not correspond to successful pair records")
    ground_truth_poses_all = load_kitti_poses(ground_truth_path)
    if frame_indices.max() >= len(ground_truth_poses_all):
        raise ValueError("Ground truth does not cover all processed frames")
    start_inverse = np.linalg.inv(ground_truth_poses_all[start_index])
    ground_truth_poses = start_inverse @ ground_truth_poses_all[frame_indices]
    ground_truth = ground_truth_poses[:, :3, 3]
    aligned, scale, rotation, translation = align_similarity(estimated, ground_truth)
    errors = np.linalg.norm(aligned - ground_truth, axis=1)

    estimated_from_origin = estimated - estimated[0]
    ground_truth_from_origin = ground_truth - ground_truth[0]
    scale_only = float(
        np.sum(estimated_from_origin * ground_truth_from_origin)
        / np.sum(estimated_from_origin * estimated_from_origin)
    )
    scale_only_aligned = scale_only * estimated_from_origin
    scale_only_errors = np.linalg.norm(scale_only_aligned - ground_truth_from_origin, axis=1)

    relative_rotation_errors: list[float] = []
    direction_cosines: list[float] = []
    per_pair_rotation_errors: dict[int, float] = {}
    per_pair_direction_cosines: dict[int, float] = {}
    for item in successful_items:
        pair = start_index + item["pair_index"]
        relative_gt = np.linalg.inv(ground_truth_poses_all[pair + 1]) @ ground_truth_poses_all[pair]
        rotation_gt = relative_gt[:3, :3]
        rotation_est = np.asarray(item["rotation_current_from_previous"], dtype=float).reshape(3, 3)
        rotation_error = rotation_angle_degrees(rotation_est.T @ rotation_gt)
        relative_rotation_errors.append(rotation_error)
        per_pair_rotation_errors[pair] = rotation_error
        if item.get("translation_direction_current_frame") is not None:
            translation_est = np.asarray(item["translation_direction_current_frame"], dtype=float)
            displacement_est_previous = -rotation_est.T @ translation_est
            displacement_gt_previous = -rotation_gt.T @ relative_gt[:3, 3]
            denominator = np.linalg.norm(displacement_est_previous) * np.linalg.norm(displacement_gt_previous)
            direction = float(np.dot(displacement_est_previous, displacement_gt_previous) / max(denominator, 1e-12))
            direction_cosines.append(direction)
            per_pair_direction_cosines[pair] = direction

    estimated_poses_path = output_dir / "poses_camera_to_world.csv"
    if not estimated_poses_path.is_file():
        raise ValueError("poses_camera_to_world.csv is required for rotation evaluation")
    estimated_poses = np.loadtxt(estimated_poses_path, delimiter=",", skiprows=1).reshape(-1, 4, 4)
    if len(estimated_poses) != len(frame_indices):
        raise ValueError("Saved pose count does not correspond to successful pair records")
    accumulated_orientation_errors = np.asarray([
        rotation_angle_degrees(estimated_pose[:3, :3].T @ ground_truth_pose[:3, :3])
        for estimated_pose, ground_truth_pose in zip(estimated_poses, ground_truth_poses)
    ])
    # Also compare against ground truth composed only across accepted edges. This
    # separates drift in integrated rotations from the known orientation gaps
    # created by correctly rejected pairs.
    coverage_ground_truth_rotations = [np.eye(3, dtype=np.float64)]
    coverage_rotation = np.eye(3, dtype=np.float64)
    for item in successful_items:
        pair = start_index + item["pair_index"]
        relative_gt = np.linalg.inv(ground_truth_poses_all[pair + 1]) @ ground_truth_poses_all[pair]
        coverage_rotation = coverage_rotation @ relative_gt[:3, :3].T
        coverage_ground_truth_rotations.append(coverage_rotation.copy())
    coverage_orientation_errors = np.asarray([
        rotation_angle_degrees(estimated_pose[:3, :3].T @ ground_truth_rotation)
        for estimated_pose, ground_truth_rotation in zip(estimated_poses, coverage_ground_truth_rotations)
    ])
    heading_estimated = np.unwrap(np.arctan2(estimated_poses[:, 0, 2], estimated_poses[:, 2, 2]))
    heading_ground_truth = np.unwrap(np.arctan2(ground_truth_poses[:, 0, 2], ground_truth_poses[:, 2, 2]))
    heading_estimated = np.degrees(heading_estimated - heading_estimated[0])
    heading_ground_truth = np.degrees(heading_ground_truth - heading_ground_truth[0])
    direction_cosines_array = np.asarray(direction_cosines)
    direction_errors_degrees = np.degrees(np.arccos(np.clip(direction_cosines_array, -1.0, 1.0)))
    relative_rotation_errors_array = np.asarray(relative_rotation_errors)
    alignment_rotation_degrees = rotation_angle_degrees(rotation)
    states = [item.get("state", "VALID_MOTION" if item["success"] else "TRACKING_LOST") for item in metrics["pairs"]]
    false_accepted_pairs = []
    for item in successful_items:
        pair = start_index + item["pair_index"]
        if item.get("state", "VALID_MOTION") != "VALID_MOTION":
            continue
        if per_pair_rotation_errors[pair] > 2.0 or per_pair_direction_cosines.get(pair, 1.0) < 0.5:
            false_accepted_pairs.append(pair)

    report: dict[str, float | int | str] = {
        "alignment": "Umeyama 7-DoF Sim(3): rotation, translation, and one global scale",
        "matched_poses": int(len(estimated)),
        "ground_truth_start_frame": start_index,
        "ground_truth_end_frame": int(start_index + max(successful_pairs) + 1),
        "similarity_scale_gt_meters_per_vo_unit": scale,
        "similarity_rotation_degrees": alignment_rotation_degrees,
        "similarity_translation_norm_gt_meters": float(np.linalg.norm(translation)),
        "sim3_aligned_ate_rmse_gt_meters": float(np.sqrt(np.mean(errors * errors))),
        "sim3_aligned_ate_median_gt_meters": float(np.median(errors)),
        "fixed_axes_origin_scale_gt_meters_per_vo_unit": scale_only,
        "fixed_axes_origin_scale_ate_rmse_gt_meters": float(np.sqrt(np.mean(scale_only_errors**2))),
        "relative_translation_direction_cosine_mean": float(np.mean(direction_cosines_array)),
        "relative_translation_direction_cosine_median": float(np.median(direction_cosines_array)),
        "relative_translation_direction_cosine_min": float(np.min(direction_cosines_array)),
        "relative_translation_direction_opposite_count": int(np.count_nonzero(direction_cosines_array < 0)),
        "relative_translation_direction_error_mean_degrees": float(np.mean(direction_errors_degrees)),
        "relative_translation_direction_error_p95_degrees": float(np.percentile(direction_errors_degrees, 95)),
        "relative_translation_direction_error_max_degrees": float(np.max(direction_errors_degrees)),
        "relative_rotation_error_mean_degrees": float(np.mean(relative_rotation_errors_array)),
        "relative_rotation_error_median_degrees": float(np.median(relative_rotation_errors_array)),
        "relative_rotation_error_p95_degrees": float(np.percentile(relative_rotation_errors_array, 95)),
        "relative_rotation_error_max_degrees": float(np.max(relative_rotation_errors_array)),
        "accumulated_orientation_error_final_degrees": float(accumulated_orientation_errors[-1]),
        "accumulated_orientation_error_mean_degrees": float(np.mean(accumulated_orientation_errors)),
        "accumulated_orientation_error_max_degrees": float(np.max(accumulated_orientation_errors)),
        "coverage_conditioned_orientation_note": "GT rotations composed only over accepted edges; excludes known gaps",
        "coverage_conditioned_orientation_error_final_degrees": float(coverage_orientation_errors[-1]),
        "coverage_conditioned_orientation_error_mean_degrees": float(np.mean(coverage_orientation_errors)),
        "coverage_conditioned_orientation_error_max_degrees": float(np.max(coverage_orientation_errors)),
        "estimated_net_heading_change_degrees": float(heading_estimated[-1]),
        "ground_truth_net_heading_change_degrees": float(heading_ground_truth[-1]),
        "valid_motion_count": states.count("VALID_MOTION"),
        "rotation_only_count": states.count("ROTATION_ONLY"),
        "degenerate_count": states.count("DEGENERATE"),
        "tracking_lost_count": states.count("TRACKING_LOST"),
        "false_accepted_definition": "accepted VALID_MOTION with rotation error >2 deg or translation-direction cosine <0.5",
        "false_accepted_count": len(false_accepted_pairs),
        "false_accepted_global_pairs": false_accepted_pairs,
        "rotation_only_global_pairs": [
            start_index + item["pair_index"] for item in metrics["pairs"] if item.get("state") == "ROTATION_ONLY"
        ],
        "degenerate_global_pairs": [
            start_index + item["pair_index"] for item in metrics["pairs"] if item.get("state") == "DEGENERATE"
        ],
    }
    (output_dir / "comparison_metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(estimated[:, 0], estimated[:, 2], label="VO (arbitrary scale)", color="tab:blue")
    axes[0].set_title("Native monocular VO — not metric")
    axes[0].set_xlabel("world X [arbitrary scale]")
    axes[0].set_ylabel("world Z [arbitrary scale]")
    axes[1].plot(ground_truth[:, 0], ground_truth[:, 2], label="KITTI ground truth", color="black")
    axes[1].plot(aligned[:, 0], aligned[:, 2], label="VO after Sim(3)", color="tab:orange")
    axes[1].plot(
        scale_only_aligned[:, 0] + ground_truth[0, 0],
        scale_only_aligned[:, 2] + ground_truth[0, 2],
        label="VO, origin + scale only",
        color="tab:green",
        linestyle="--",
    )
    axes[1].set_title("Shape comparison after 7-DoF Sim(3) alignment")
    axes[1].set_xlabel("KITTI world X [ground-truth meters]")
    axes[1].set_ylabel("KITTI world Z [ground-truth meters]")
    for axis in axes:
        axis.axis("equal")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "trajectory_comparison.png", dpi=170)
    plt.close(figure)

    pair_frames = np.asarray([start_index + pair + 1 for pair in successful_pairs])
    all_pair_frames = np.asarray([start_index + item["pair_index"] + 1 for item in metrics["pairs"]])
    all_matches = np.asarray([item["matches"] for item in metrics["pairs"]])
    all_inliers = np.asarray([item["inliers"] for item in metrics["pairs"]])
    failures = np.asarray([not item["success"] for item in metrics["pairs"]])
    rotation_only = np.asarray([item.get("state") == "ROTATION_ONLY" for item in metrics["pairs"]])
    diagnostics, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(frame_indices, heading_ground_truth, color="black", label="KITTI ground-truth heading")
    axes[0].plot(frame_indices, heading_estimated, color="tab:blue", label="VO heading")
    axes[0].set_ylabel("heading change [degrees]")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    axes[1].plot(pair_frames, relative_rotation_errors_array, color="tab:purple", label="relative rotation error")
    axes[1].plot(frame_indices, accumulated_orientation_errors, color="tab:red", label="accumulated orientation error")
    axes[1].set_ylabel("rotation error [degrees]")
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    axes[2].plot(all_pair_frames, all_matches, color="tab:orange", alpha=0.75, label="ratio-test matches")
    axes[2].plot(all_pair_frames, all_inliers, color="tab:green", label="geometric inliers")
    if np.any(failures):
        axes[2].scatter(all_pair_frames[failures], np.zeros(np.count_nonzero(failures)), marker="x", color="red", label="tracking failure")
    if np.any(rotation_only):
        axes[2].scatter(all_pair_frames[rotation_only], all_inliers[rotation_only], marker="o", facecolors="none", edgecolors="blue", label="rotation only")
    axes[2].set_ylabel("count")
    axes[2].set_xlabel("KITTI frame index")
    axes[2].legend()
    axes[2].grid(alpha=0.3)
    diagnostics.tight_layout()
    diagnostics.savefig(output_dir / "rotation_tracking_diagnostics.png", dpi=170)
    plt.close(diagnostics)

    if any(item.get("diagnostics") for item in metrics["pairs"]):
        def diagnostic_values(name: str) -> np.ndarray:
            return np.asarray([
                np.nan if item.get("diagnostics") is None else item["diagnostics"].get(name, np.nan)
                for item in metrics["pairs"]
            ], dtype=float)

        reliability, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        axes[0].plot(all_pair_frames, diagnostic_values("essential_inlier_ratio"), label="essential cheirality ratio")
        axes[0].plot(all_pair_frames, diagnostic_values("homography_inlier_ratio"), label="homography RANSAC ratio", alpha=0.8)
        axes[0].plot(all_pair_frames, diagnostic_values("rotation_only_inlier_ratio"), label="constrained rotation ratio", alpha=0.8)
        axes[0].axhline(0.6, color="blue", linestyle="--", alpha=0.5, label="rotation-only threshold")
        axes[0].set_ylabel("inlier ratio")
        axes[0].legend(ncol=2)
        axes[1].plot(all_pair_frames, diagnostic_values("compensated_parallax_p10_degrees"), label="compensated parallax p10")
        axes[1].plot(all_pair_frames, diagnostic_values("rotation_innovation_degrees"), label="rotation innovation")
        axes[1].axhline(1.0, color="tab:blue", linestyle="--", alpha=0.5, label="parallax threshold")
        axes[1].axhline(2.0, color="tab:orange", linestyle="--", alpha=0.5, label="innovation threshold")
        axes[1].set_ylabel("degrees")
        axes[1].set_ylim(bottom=0)
        axes[1].legend(ncol=2)
        state_code = {"VALID_MOTION": 0, "ROTATION_ONLY": 1, "DEGENERATE": 2, "TRACKING_LOST": 3}
        axes[2].scatter(all_pair_frames, [state_code[state] for state in states], c=[state_code[state] for state in states], cmap="viridis", s=14)
        axes[2].set_yticks(list(state_code.values()), list(state_code.keys()))
        axes[2].set_xlabel("KITTI frame index")
        axes[2].set_ylabel("decision")
        for axis in axes:
            axis.grid(alpha=0.3)
        reliability.tight_layout()
        reliability.savefig(output_dir / "geometric_reliability_diagnostics.png", dpi=170)
        plt.close(reliability)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--start-index", type=int, default=0, help="KITTI index of the first input image")
    args = parser.parse_args()
    report = evaluate(args.output_dir, args.ground_truth, args.start_index)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
