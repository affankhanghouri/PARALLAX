#!/usr/bin/env python3
"""Post-run KITTI evaluation for a saved two-view initialization."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.evaluate_kitti import load_kitti_poses, rotation_angle_degrees


def evaluate(output_dir: Path, ground_truth: Path, start_index: int) -> dict[str, object]:
    runtime = json.loads((output_dir / "reconstruction_metrics.json").read_text(encoding="utf-8"))
    if not runtime.get("success"):
        poses = load_kitti_poses(ground_truth)
        assessed_ground_truth = []
        for item in runtime.get("pair_assessments", []):
            anchor = start_index + int(item["anchor_index"])
            candidate = start_index + int(item["candidate_index"])
            relative = np.linalg.inv(poses[candidate]) @ poses[anchor]
            assessed_ground_truth.append({
                "global_anchor_frame": anchor,
                "global_candidate_frame": candidate,
                "ground_truth_baseline_meters_evaluation_only": float(np.linalg.norm(relative[:3, 3])),
                "ground_truth_relative_rotation_degrees": rotation_angle_degrees(relative[:3, :3]),
            })
        report: dict[str, object] = {
            "runtime_success": False,
            "note": "No runtime-selected pair was triangulated; ground truth was not consulted for selection.",
            "assessed_pair_ground_truth": assessed_ground_truth,
        }
        (output_dir / "ground_truth_evaluation.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report
    saved = np.load(output_dir / "reconstruction.npz")
    anchor = start_index + int(runtime["anchor_offset"])
    candidate = start_index + int(runtime["candidate_offset"])
    poses = load_kitti_poses(ground_truth)
    if candidate >= len(poses):
        raise ValueError("ground truth does not cover the selected pair")
    relative_gt = np.linalg.inv(poses[candidate]) @ poses[anchor]
    rotation_est = saved["rotation_second_from_first"]
    translation_est = saved["translation_second_from_first"]
    translation_gt = relative_gt[:3, 3]
    direction_cosine = float(
        np.dot(translation_est, translation_gt)
        / max(np.linalg.norm(translation_est) * np.linalg.norm(translation_gt), 1e-12)
    )
    report = {
        "runtime_success": True,
        "ground_truth_usage": "post-run evaluation only; not available to pair selection or triangulation",
        "global_anchor_frame": anchor,
        "global_candidate_frame": candidate,
        "frame_gap": candidate - anchor,
        "ground_truth_baseline_meters_evaluation_only": float(np.linalg.norm(translation_gt)),
        "relative_rotation_error_degrees": rotation_angle_degrees(rotation_est.T @ relative_gt[:3, :3]),
        "translation_direction_cosine": direction_cosine,
        "translation_direction_error_degrees": float(np.degrees(np.arccos(np.clip(direction_cosine, -1, 1)))),
    }
    (output_dir / "ground_truth_evaluation.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
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
