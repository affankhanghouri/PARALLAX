#!/usr/bin/env python3
"""Select a calibrated frame pair and initialize a sparse two-view reconstruction."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import json
import logging
from pathlib import Path
import sys
from time import perf_counter

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import load_config
from src.features import ORBFeatures
from src.frame_source import ImageSequence
from src.initialization import PairSelectionThresholds, TwoViewPairSelector
from src.reconstruction import TriangulationThresholds, triangulate_two_view
from src.reconstruction_visualization import save_reconstruction


LOGGER = logging.getLogger("two_view_initialization")


def run(
    config_path: str | Path,
    *,
    anchor_offset: int = 0,
    max_candidates: int = 30,
) -> dict[str, object]:
    config = load_config(config_path)
    source = ImageSequence(config.image_dir, anchor_offset + max_candidates + 1)
    iterator = iter(source)
    anchor = None
    for frame in iterator:
        if frame.index == anchor_offset:
            anchor = frame
            break
    if anchor is None:
        raise ValueError(f"anchor offset {anchor_offset} is outside the input sequence")

    frontend = ORBFeatures(config.max_features, config.ratio_test)
    selector = TwoViewPairSelector(
        frontend, config.camera.matrix,
        thresholds=PairSelectionThresholds(
            min_matches=max(config.initialization_min_matches, config.min_matches),
            min_geometric_inliers=config.initialization_min_inliers,
            min_median_compensated_parallax_degrees=config.initialization_min_median_parallax_degrees,
            min_p25_compensated_parallax_degrees=config.initialization_min_p25_parallax_degrees,
            grid_rows=config.initialization_grid_rows,
            grid_columns=config.initialization_grid_columns,
            min_occupied_grid_ratio=config.initialization_min_grid_ratio,
        ),
        min_pose_inliers=config.min_inliers,
        ransac_probability=config.ransac_probability,
        ransac_threshold_px=config.ransac_threshold_px,
        rotation_only_residual_degrees=config.rotation_only_residual_degrees,
        rotation_only_min_ratio=config.rotation_only_min_ratio,
        rotation_innovation_degrees=config.rotation_innovation_degrees,
        suspicious_parallax_p10_degrees=config.suspicious_parallax_p10_degrees,
    )
    total_started = perf_counter()
    selection = selector.select(anchor, iterator)
    selection_ms = (perf_counter() - total_started) * 1000.0
    output = config.output_dir
    output.mkdir(parents=True, exist_ok=True)
    assessments = [
        {**asdict(item), "state": item.state.value}
        for item in selection.assessments
    ]
    if selection.selected is None:
        report: dict[str, object] = {
            "success": False,
            "reason": "no suitable pair found",
            "anchor_offset": anchor_offset,
            "selection_ms": selection_ms,
            "pair_assessments": assessments,
        }
        (output / "reconstruction_metrics.json").write_text(
            json.dumps(report, indent=2), encoding="utf-8"
        )
        return report

    selected = selection.selected
    triangulation_started = perf_counter()
    reconstruction = triangulate_two_view(
        selected.points_anchor, selected.points_candidate, config.camera.matrix,
        selected.estimate.rotation, selected.estimate.translation,
        TriangulationThresholds(
            max_reprojection_error_px=config.triangulation_max_reprojection_error_px,
            min_triangulation_angle_degrees=config.triangulation_min_angle_degrees,
            max_depth=config.triangulation_max_depth,
        ),
    )
    triangulation_ms = (perf_counter() - triangulation_started) * 1000.0
    visualization_started = perf_counter()
    save_reconstruction(
        reconstruction, output / "sparse_reconstruction.png",
        title=f"Selected local frames {selected.anchor.index} and {selected.candidate.index}",
    )
    visualization_ms = (perf_counter() - visualization_started) * 1000.0
    accepted = reconstruction.accepted_mask
    np.savetxt(
        output / "landmarks.csv",
        reconstruction.points_first_camera[accepted], delimiter=",",
        header="camera1_x_arbitrary,camera1_y_arbitrary,camera1_z_arbitrary", comments="",
    )
    np.savez_compressed(
        output / "reconstruction.npz",
        points_first_camera=reconstruction.points_first_camera,
        accepted_mask=accepted,
        points_anchor=selected.points_anchor,
        points_candidate=selected.points_candidate,
        anchor_feature_indices=selected.anchor_feature_indices,
        candidate_feature_indices=selected.candidate_feature_indices,
        rotation_second_from_first=reconstruction.rotation_second_from_first,
        translation_second_from_first=reconstruction.translation_second_from_first,
        camera_center_second=reconstruction.camera_center_second,
    )
    report = {
        "success": True,
        "coordinate_convention": "X_second = R_second_first X_first + t_second_first",
        "reconstruction_frame": "camera 1; camera 1 at origin; camera 2 center = -R.T t",
        "scale": "arbitrary; translation normalized to one; values are not meters",
        "anchor_offset": selected.anchor.index,
        "candidate_offset": selected.candidate.index,
        "anchor_file": selected.anchor.path.name,
        "candidate_file": selected.candidate.path.name,
        "selected_pair": {**asdict(selected.assessment), "state": selected.assessment.state.value},
        "pair_assessments": assessments,
        "triangulation": reconstruction.summary(),
        "timing_ms": {
            "pair_selection": selection_ms,
            "triangulation": triangulation_ms,
            "visualization": visualization_ms,
            "core_without_visualization": selection_ms + triangulation_ms,
            "total": (perf_counter() - total_started) * 1000.0,
        },
    }
    (output / "reconstruction_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--anchor-offset", type=int, default=0)
    parser.add_argument("--max-candidates", type=int, default=30)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
    try:
        report = run(
            args.config, anchor_offset=args.anchor_offset,
            max_candidates=args.max_candidates,
        )
    except (FileNotFoundError, ValueError, OSError) as exc:
        LOGGER.error("Cannot initialize reconstruction: %s", exc)
        return 2
    print(json.dumps(report, indent=2))
    return 0 if report["success"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
