#!/usr/bin/env python3
"""Post-run KITTI evaluation for V2.1 local-map PnP tracking."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_kitti import align_similarity, load_kitti_poses, rotation_angle_degrees


def evaluate(output_dir: Path, ground_truth_path: Path, start_index: int) -> dict[str, object]:
    runtime = json.loads((output_dir / "tracking_metrics.json").read_text(encoding="utf-8"))
    saved = np.loadtxt(output_dir / "map_poses_camera_to_segment.csv", delimiter=",", skiprows=1)
    saved = np.atleast_2d(saved)
    offsets = saved[:, 0].astype(int)
    estimated_poses = saved[:, 1:].reshape(-1, 4, 4)
    ground_truth_all = load_kitti_poses(ground_truth_path)
    initialization_anchor = int(runtime["initialization"]["anchor_offset"])
    origin_inverse = np.linalg.inv(ground_truth_all[start_index + initialization_anchor])
    ground_truth_poses = origin_inverse @ ground_truth_all[start_index + offsets]
    estimated_positions = estimated_poses[:, :3, 3]
    ground_truth_positions = ground_truth_poses[:, :3, 3]

    initialization_candidate = int(runtime["initialization"]["candidate_offset"])
    candidate_row = int(np.flatnonzero(offsets == initialization_candidate)[0])
    initialization_scale = float(
        np.linalg.norm(ground_truth_positions[candidate_row])
        / np.linalg.norm(estimated_positions[candidate_row])
    )
    initialization_scaled = initialization_scale * estimated_positions
    initialization_errors = np.linalg.norm(initialization_scaled - ground_truth_positions, axis=1)
    denominator = float(np.sum(estimated_positions * estimated_positions))
    scale_only = float(np.sum(estimated_positions * ground_truth_positions) / denominator)
    scale_only_positions = scale_only * estimated_positions
    scale_only_errors = np.linalg.norm(scale_only_positions - ground_truth_positions, axis=1)
    sim3_positions, sim3_scale, sim3_rotation, sim3_translation = align_similarity(
        estimated_positions, ground_truth_positions
    )
    sim3_errors = np.linalg.norm(sim3_positions - ground_truth_positions, axis=1)
    trajectory_singular_values = np.linalg.svd(
        ground_truth_positions - ground_truth_positions.mean(axis=0), compute_uv=False
    )
    trajectory_linearity_ratio = float(
        trajectory_singular_values[1] / max(trajectory_singular_values[0], 1e-12)
    )

    rotation_errors = np.asarray([
        rotation_angle_degrees(estimated[:3, :3].T @ truth[:3, :3])
        for estimated, truth in zip(estimated_poses, ground_truth_poses)
    ])
    relative_rotation_errors = []
    relative_position_errors = []
    contiguous_edges = []
    for index in range(1, len(offsets)):
        if offsets[index] != offsets[index - 1] + 1:
            continue
        relative_estimated = np.linalg.inv(estimated_poses[index]) @ estimated_poses[index - 1]
        relative_truth = np.linalg.inv(ground_truth_poses[index]) @ ground_truth_poses[index - 1]
        relative_rotation_errors.append(
            rotation_angle_degrees(relative_estimated[:3, :3].T @ relative_truth[:3, :3])
        )
        estimated_step = initialization_scale * (
            estimated_positions[index] - estimated_positions[index - 1]
        )
        truth_step = ground_truth_positions[index] - ground_truth_positions[index - 1]
        relative_position_errors.append(float(np.linalg.norm(estimated_step - truth_step)))
        contiguous_edges.append([int(offsets[index - 1]), int(offsets[index])])

    tracked_mask = offsets > initialization_candidate
    report = {
        "ground_truth_usage": "post-run evaluation only; never available to map creation, association, or PnP",
        "accepted_pose_offsets": offsets.tolist(),
        "accepted_tracking_pose_count": int(np.count_nonzero(tracked_mask)),
        "initialization_baseline_scale": {
            "method": (
                f"one scale from the {start_index + initialization_anchor}->"
                f"{start_index + initialization_candidate} initialization baseline; "
                "axes and initialization-anchor origin fixed"
            ),
            "gt_meters_per_map_unit": initialization_scale,
            "tracked_position_error_mean_meters": float(np.mean(initialization_errors[tracked_mask])),
            "tracked_position_error_p95_meters": float(np.percentile(initialization_errors[tracked_mask], 95)),
            "tracked_position_error_max_meters": float(np.max(initialization_errors[tracked_mask])),
        },
        "global_scale_only_alignment": {
            "method": "least-squares scale; axes and origin fixed; evaluation only",
            "gt_meters_per_map_unit": scale_only,
            "ate_rmse_meters": float(np.sqrt(np.mean(scale_only_errors**2))),
        },
        "sim3_alignment": {
            "method": "Umeyama 7-DoF rotation, translation, and global scale; evaluation only",
            "gt_meters_per_map_unit": sim3_scale,
            "rotation_degrees": rotation_angle_degrees(sim3_rotation),
            "translation_norm_meters": float(np.linalg.norm(sim3_translation)),
            "ate_rmse_meters": float(np.sqrt(np.mean(sim3_errors**2))),
            "trajectory_second_to_first_singular_value_ratio": trajectory_linearity_ratio,
            "warning": (
                "short trajectory is nearly collinear; Sim(3) rotation and ATE are poorly conditioned"
                if trajectory_linearity_ratio < 0.05 else None
            ),
        },
        "absolute_orientation_error_degrees": {
            "tracked_mean": float(np.mean(rotation_errors[tracked_mask])),
            "tracked_p95": float(np.percentile(rotation_errors[tracked_mask], 95)),
            "tracked_max": float(np.max(rotation_errors[tracked_mask])),
        },
        "contiguous_tracking_edges": contiguous_edges,
        "relative_rotation_error_degrees": {
            "mean": float(np.mean(relative_rotation_errors)),
            "max": float(np.max(relative_rotation_errors)),
        },
        "relative_position_error_initialization_scaled_meters": {
            "mean": float(np.mean(relative_position_errors)),
            "max": float(np.max(relative_position_errors)),
        },
    }
    (output_dir / "tracking_ground_truth_evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )

    last_attempted = max(item["frame_offset"] for item in runtime["frames"])
    full_offsets = np.arange(initialization_anchor, last_attempted + 1)
    full_ground_truth = origin_inverse @ ground_truth_all[start_index + full_offsets]
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    axes[0].plot(estimated_positions[:, 0], estimated_positions[:, 2], "o-", label="map PnP (arbitrary scale)")
    axes[0].set_title("Native local-map trajectory — not metric")
    axes[0].set_xlabel("segment X [arbitrary units]")
    axes[0].set_ylabel("segment Z [arbitrary units]")
    axes[1].plot(
        full_ground_truth[:, 0, 3], full_ground_truth[:, 2, 3], "k.-",
        label="KITTI ground truth (through last attempt)",
    )
    axes[1].plot(
        initialization_scaled[:, 0], initialization_scaled[:, 2], "o-",
        label="PnP, initialization-baseline scale",
    )
    axes[1].plot(sim3_positions[:, 0], sim3_positions[:, 2], "x--", label="PnP after 7-DoF Sim(3)")
    axes[1].set_title("Evaluation alignments; rejected-frame gaps remain")
    axes[1].set_xlabel("KITTI X [meters]")
    axes[1].set_ylabel("KITTI Z [meters]")
    for axis in axes:
        axis.axis("equal")
        axis.grid(alpha=0.3)
        axis.legend()
    figure.tight_layout()
    figure.savefig(output_dir / "map_trajectory_comparison.png", dpi=180)
    plt.close(figure)
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--ground-truth", required=True, type=Path)
    parser.add_argument("--start-index", required=True, type=int)
    args = parser.parse_args()
    print(json.dumps(evaluate(args.output_dir, args.ground_truth, args.start_index), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
