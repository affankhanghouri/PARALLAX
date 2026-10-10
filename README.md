# Real-Time Monocular Visual SLAM & 3D Spatial Tracking Engine — Version 1

This repository implements **Version 1 only: classical monocular visual odometry**. It reads a calibrated image sequence, extracts and matches ORB features, verifies matches with a RANSAC essential matrix, recovers relative rotation and translation direction, and accumulates a camera trajectory. It is CPU-only and deliberately contains no mapping, bundle adjustment, loop closure, relocalization, scale recovery, dense reconstruction, ROS, or learned models.

## Installation

Python 3.11 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
pytest -q
```

`opencv-python-headless` and Matplotlib's non-interactive backend allow execution without a display server.

## KITTI odometry setup

1. Obtain the color or grayscale odometry data set and calibration files from the official [KITTI Visual Odometry benchmark](https://www.cvlibs.net/datasets/kitti/eval_odometry.php). The project never downloads a data set automatically.
2. Extract a sequence so its images are in a directory such as `data/kitti/sequences/00/image_0/`.
3. Copy `configs/kitti_example.yaml` to a new file. The example matrix is intentionally `null`; the application will reject it until real calibration is supplied.
4. Open that sequence's `calib.txt`. For the left grayscale camera, `P0` has the form `K [I | 0]`; its first three columns form the 3×3 intrinsic matrix. For another image directory, use the corresponding projection row. Copy those nine values into `camera.matrix` as three YAML rows. Do not reuse calibration from a different sequence or resized imagery.
5. Set `input.image_dir` to the extracted image directory. Relative paths are resolved from the configuration file's directory.

Example shape (the symbols must be replaced with actual numeric calibration values):

```yaml
camera:
  matrix:
    - [fx, 0.0, cx]
    - [0.0, fy, cy]
    - [0.0, 0.0, 1.0]
input:
  image_dir: ../data/kitti/sequences/00/image_0
```

Run:

```bash
python main.py --config configs/kitti_00.yaml
```

### Small real-world validation set (sequence 04)

KITTI's current download page requires an account and the complete grayscale odometry archive is 22 GB. For a smaller validation download, the proSLAM project publishes a compact sequence-04 archive (271 frames, about 181 MB) on Google Drive. It contains rectified left/right grayscale KITTI images converted losslessly to PGM, the sequence calibration, and ground truth. The application now accepts PGM directly.

```bash
source .venv/bin/activate
python -m pip install gdown       # setup helper only; not a runtime dependency
mkdir -p data/kitti/downloads data/kitti/srrg04
gdown 0ByaBRAPfmgEqOEhEdEdLcUJUMlE -O data/kitti/downloads/kitti_04.tar.gz
tar -xzf data/kitti/downloads/kitti_04.tar.gz -C data/kitti/srrg04

mkdir -p data/kitti/sequences/04/image_0 data/kitti/poses
cp data/kitti/srrg04/04.txt.d/camera_left.image_raw_*.pgm data/kitti/sequences/04/image_0/
cp data/kitti/srrg04/calib.txt data/kitti/sequences/04/calib.txt
cp data/kitti/srrg04/04_gt.txt data/kitti/poses/04.txt
```

Before running, verify that `P0` in `data/kitti/sequences/04/calib.txt` has the same first 3×3 values as `camera.matrix` in `configs/kitti_04_validation.yaml`. That configuration selects only the left camera, limits processing to 200 frames, and saves every twentieth match visualization.

```bash
python main.py --config configs/kitti_04_validation.yaml
python scripts/evaluate_kitti.py \
  --output-dir outputs/kitti_04_200 \
  --ground-truth data/kitti/poses/04.txt
```

The evaluation script leaves native VO coordinates in arbitrary scale. It additionally reports a fixed-axis, fixed-origin global scale fit and a separately labeled Umeyama 7-DoF Sim(3) fit for shape comparison. Those aligned results are evaluation views, not metric scale recovered by the odometry pipeline.

Outputs are written to the configured output directory:

- `matches/pair_*.jpg`: consecutive images with geometrically verified inlier matches in green and counts in the title.
- `trajectory.png`: 3D camera path with initial/current locations and arbitrary-scale coordinate labels.
- `trajectory.csv`: accepted camera positions, explicitly named as arbitrary-scale coordinates.
- `metrics.json`: per-pair timings, match/inlier counts, reasons for failures, success/failure totals, core FPS, and p50/p95 core latency.

Set `output.save_match_every: 0` to disable match images, or use a larger integer to reduce visualization cost. `runtime.max_frames` can limit a diagnostic run.

## Geometry and pose convention

OpenCV `recoverPose` returns a transform satisfying

```text
X_current = R_current_previous X_previous + t_current_previous
```

The tracker stores camera-to-world transforms `T_world_camera`. It therefore updates with the inverse relative transform:

```text
T_world_camera_current = T_world_camera_previous inverse(T_current_previous)
R_world_camera_current = R_world_camera_previous R_current_previous^T
p_world_camera_current = p_world_camera_previous
                         - R_world_camera_previous R_current_previous^T t_current_previous
```

`recoverPose` determines only the direction of translation. This implementation normalizes every accepted translation to length one. Positions are consequently in **arbitrary, non-metric scale units**, and step length does not reflect actual speed or distance.

## Failure behavior

Missing descriptors, too few filtered matches, failed/degenerate essential matrix estimation, insufficient pose inliers, invalid transforms, and unreadable images are reported as tracking loss. A failed estimate never changes the pose or adds a trajectory point. Logs explicitly call subsequent output unreliable during a failed pair; this Version 1 has no relocalization. After an unreadable frame the pair chain is reset rather than matching across the missing frame.

## Performance interpretation

Core timing contains feature extraction, matching, and calibrated geometry. Visualization time is measured separately and excluded from core FPS. The first pair includes extraction of both its frames; later frames are extracted once and reused. Results depend on the CPU, image size, feature count, sequence texture, and output settings. No benchmark numbers are included without running a real data set.

## Tests

```bash
pytest -q
```

Tests cover intrinsic validation, missing/insufficient descriptors, synthetic calibrated relative-pose recovery, camera-to-world composition, failure immutability, and a short generated image-sequence smoke run. Synthetic feature recovery validates two-view geometry; it does not establish real-world trajectory accuracy.

## Known limitations

- Monocular translation has no metric scale; forcing unit steps distorts trajectory length and speed.
- Consecutive-frame tracking drifts because there is no map, bundle adjustment, loop closure, or relocalization.
- Pure rotation, little parallax, repeated texture, blur, dynamic objects, and weak texture can make essential-matrix estimation degenerate or unreliable.
- Calibration must match the camera and image resolution. Lens distortion should be removed before input; this version does not load distortion coefficients.
- A recovered essential-matrix pose can still be geometrically plausible but wrong. Inlier thresholds are safeguards, not an accuracy guarantee.

## Project structure

```text
.
├── main.py
├── configs/kitti_example.yaml
├── configs/kitti_04_validation.yaml
├── requirements.txt
├── scripts/evaluate_kitti.py
├── src/
│   ├── config.py
│   ├── features.py
│   ├── frame_source.py
│   ├── geometry.py
│   ├── metrics.py
│   ├── odometry.py
│   └── visualization.py
└── tests/
    ├── test_config.py
    ├── test_features.py
    ├── test_geometry.py
    ├── test_odometry.py
    └── test_smoke.py
```
