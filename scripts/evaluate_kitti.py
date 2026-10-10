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


def load_kitti_positions(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 12:
        raise ValueError(f"Expected KITTI N x 12 poses in {path}, got {values.shape}")
    return values[:, [3, 7, 11]]


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


def evaluate(output_dir: Path, ground_truth_path: Path) -> dict[str, float | int | str]:
    estimated = np.loadtxt(output_dir / "trajectory.csv", delimiter=",", skiprows=1)
    estimated = np.atleast_2d(estimated)
    metrics = json.loads((output_dir / "metrics.json").read_text(encoding="utf-8"))
    successful_pairs = [item["pair_index"] for item in metrics["pairs"] if item["success"]]
    frame_indices = np.asarray([0, *(index + 1 for index in successful_pairs)], dtype=int)
    if len(frame_indices) != len(estimated):
        raise ValueError("Trajectory rows do not correspond to successful pair records")
    ground_truth_all = load_kitti_positions(ground_truth_path)
    if frame_indices.max() >= len(ground_truth_all):
        raise ValueError("Ground truth does not cover all processed frames")
    ground_truth = ground_truth_all[frame_indices]
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

    estimated_steps = np.diff(estimated, axis=0)
    gt_steps = np.asarray([
        ground_truth_all[pair + 1] - ground_truth_all[pair] for pair in successful_pairs
    ])
    denominators = np.linalg.norm(estimated_steps, axis=1) * np.linalg.norm(gt_steps, axis=1)
    direction_cosines = np.sum(estimated_steps * gt_steps, axis=1) / np.maximum(denominators, 1e-12)
    alignment_rotation_degrees = float(
        np.degrees(np.arccos(np.clip((np.trace(rotation) - 1.0) / 2.0, -1.0, 1.0)))
    )

    report: dict[str, float | int | str] = {
        "alignment": "Umeyama 7-DoF Sim(3): rotation, translation, and one global scale",
        "matched_poses": int(len(estimated)),
        "similarity_scale_gt_meters_per_vo_unit": scale,
        "similarity_rotation_degrees": alignment_rotation_degrees,
        "similarity_translation_norm_gt_meters": float(np.linalg.norm(translation)),
        "sim3_aligned_ate_rmse_gt_meters": float(np.sqrt(np.mean(errors * errors))),
        "sim3_aligned_ate_median_gt_meters": float(np.median(errors)),
        "fixed_axes_origin_scale_gt_meters_per_vo_unit": scale_only,
        "fixed_axes_origin_scale_ate_rmse_gt_meters": float(np.sqrt(np.mean(scale_only_errors**2))),
        "raw_step_direction_cosine_mean": float(np.mean(direction_cosines)),
        "raw_step_direction_cosine_median": float(np.median(direction_cosines)),
        "raw_step_direction_cosine_min": float(np.min(direction_cosines)),
        "raw_step_direction_opposite_count": int(np.count_nonzero(direction_cosines < 0)),
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
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    args = parser.parse_args()
    report = evaluate(args.output_dir, args.ground_truth)
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
