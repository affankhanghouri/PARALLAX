import json

import numpy as np

from scripts.evaluate_map_tracking_kitti import evaluate


def test_evaluator_uses_nonzero_initialization_anchor_as_ground_truth_origin(tmp_path) -> None:
    output = tmp_path / "output"
    output.mkdir()
    runtime = {
        "initialization": {"anchor_offset": 2, "candidate_offset": 3},
        "frames": [{"frame_offset": 4}],
    }
    (output / "tracking_metrics.json").write_text(json.dumps(runtime))
    poses = []
    for offset, x in ((2, 0.0), (3, 1.0), (4, 2.0)):
        pose = np.eye(4)
        pose[0, 3] = x
        poses.append([offset, *pose.reshape(-1)])
    np.savetxt(
        output / "map_poses_camera_to_segment.csv", np.asarray(poses), delimiter=",",
        header="offset," + ",".join(f"t{i}" for i in range(16)), comments="",
    )
    ground_truth = []
    for x in (100.0, 110.0, 120.0, 130.0, 140.0):
        pose = np.eye(4)[:3]
        pose[0, 3] = x
        ground_truth.append(pose.reshape(-1))
    ground_truth_path = tmp_path / "poses.txt"
    np.savetxt(ground_truth_path, np.asarray(ground_truth))
    report = evaluate(output, ground_truth_path, 0)
    assert report["initialization_baseline_scale"]["gt_meters_per_map_unit"] == 10.0
    assert report["initialization_baseline_scale"]["tracked_position_error_max_meters"] < 1e-9
    assert report["absolute_orientation_error_degrees"]["tracked_max"] < 1e-9
    assert "2->3" in report["initialization_baseline_scale"]["method"]
