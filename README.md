# PARALLAX — Monocular Visual Odometry and Sparse Local Mapping

The repository contains the validated V1.3 classical monocular visual-odometry engine and the V2.0 foundation for reliable two-view sparse landmark initialization. V2.0 selects one geometrically suitable frame pair, triangulates and filters local landmarks, and stops. It is CPU-only and deliberately contains no persistent map, map-based tracking, bundle adjustment, loop closure, relocalization, metric-scale recovery, dense reconstruction, ROS, or learned models.
V2.1 additionally provides a deliberately limited in-memory local map and calibrated 3D–2D PnP tracking against only the initialization landmarks. It still performs no map expansion or optimization.
V2.2 adds opt-in conservative keyframes, existing-scale landmark replenishment,
verified descriptor references, and forward–backward temporal landmark tracks.
It remains a local arbitrary-scale map with no bundle adjustment, loop closure,
relocalization, or metric-scale recovery.

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

### Challenging-turn validation

The V1.2 validation uses only sequence-00 left-camera frames 3200–3599, selected from ground-truth heading change and extracted from the compact archive. Its configuration and evaluation command are:

```bash
python main.py --config configs/kitti_00_turns_3200_3599.yaml
python scripts/evaluate_kitti.py \
  --output-dir outputs/kitti_00_turns_3200_3599 \
  --ground-truth data/kitti/srrg00/00_gt.txt \
  --start-index 3200
```

See `VALIDATION_V1_2.md` for measured rotation, heading, failure, trajectory, memory, and performance results. Ground truth is used by the separate evaluation command only, never by odometry.

### V1.3 reliability comparison

V1.3 keeps the V1.2 path available and enables reliability checks only when
`geometry.reliability_checks: true`. Reproduce both sides on the same images:

```bash
python main.py --config configs/kitti_00_turns_v13_baseline.yaml
python main.py --config configs/kitti_00_turns_v13_reliable.yaml
python scripts/evaluate_kitti.py --output-dir outputs/kitti_00_turns_v13_reliable \
  --ground-truth data/kitti/srrg00/00_gt.txt --start-index 3200
```

The four per-pair states are `VALID_MOTION`, `ROTATION_ONLY`, `DEGENERATE`, and
`TRACKING_LOST`. Rotation-only updates integrate the supported orientation and
hold position; they never add a fictional unit translation. See
`VALIDATION_V1_3.md` for threshold rationale, controlled measurements, plots,
and limitations.

### V2.0 two-view initialization

The initializer scans candidates from an anchor and selects the first pair that
passes motion-state, correspondence, compensated-parallax, and image-coverage
checks. It does not select every Nth frame and never reads ground truth.

```bash
python scripts/initialize_two_view.py \
  --config configs/kitti_00_v20_initialization.yaml \
  --anchor-offset 0 --max-candidates 30

# Separate post-run evaluation only:
python scripts/evaluate_two_view_kitti.py \
  --output-dir outputs/kitti_00_v20_initialization \
  --ground-truth data/kitti/srrg00/00_gt.txt --start-index 3200
```

The reconstruction frame is camera 1. Camera 1 is `[I|0]`, camera 2 is
`[R|t]`, and camera 2's center is `-R^T t`. Translation is normalized to one,
so all landmark coordinates and depths are arbitrary reconstruction units, not
meters. See `VALIDATION_V2.md` for thresholds, synthetic tests, KITTI results,
performance, and the proposed V2.1 data model.

### V2.1 local-map tracking

```bash
python scripts/track_local_map.py \
  --config configs/kitti_00_v21_tracking.yaml \
  --anchor-offset 0 --max-candidates 30 \
  --max-tracking-frames 50 --max-consecutive-failures 3

# Ground truth remains a separate post-run evaluation:
python scripts/evaluate_map_tracking_kitti.py \
  --output-dir outputs/kitti_00_v21_tracking \
  --ground-truth data/kitti/srrg00/00_gt.txt --start-index 3200
```

OpenCV PnP returns the segment/world-to-camera transformation. PARALLAX
inverts it before storing a camera-to-segment pose. Rejected frames store no
pose; after repeated map-tracking failures the segment is explicitly `LOST`.
See `VALIDATION_V2_1.md` for the short-sequence result and limitations.

### V2.2 local-map expansion

The V2.1 behavior remains reproducible with
`configs/kitti_00_v22_baseline.yaml`. Controlled association-only, mapping-only,
and combined configurations are also provided:

```bash
python scripts/track_local_map.py \
  --config configs/kitti_00_v22_combined.yaml \
  --anchor-offset 0 --max-candidates 30 \
  --max-tracking-frames 100 --max-consecutive-failures 3

# Ground truth is still post-run evaluation only:
python scripts/evaluate_map_tracking_kitti.py \
  --output-dir outputs/kitti_00_v22_combined \
  --ground-truth data/kitti/srrg00/00_gt.txt --start-index 3200
```

New landmarks are triangulated from already accepted map poses and directly in
the owning segment frame; later pairs are never renormalized to independent
unit baselines. See `VALIDATION_V2_2.md` for the A/B/C/D ablation, rejected
alternatives, accuracy, performance, and limitations.

### V2.3 quality-managed local map

The retained V2.3 configuration adds reversible landmark inactivity, verified
quality histories, a bounded spatially selected active set, detailed spatial
diagnostics, and an exact mapping-match optimization. Geometric PnP acceptance
thresholds are unchanged.

```bash
/usr/bin/time -v .venv/bin/python scripts/track_local_map.py \
  --config configs/kitti_00_v23_combined.yaml \
  --anchor-offset 0 --max-candidates 30 --max-tracking-frames 95

.venv/bin/python scripts/evaluate_map_tracking_kitti.py \
  --output-dir outputs/kitti_00_v23_combined \
  --ground-truth data/kitti/srrg00/00_gt.txt --start-index 3200
```

The output includes `spatial_coverage.png` and per-stage projection, descriptor
matching, quality-maintenance, mapping-match, and triangulation timings. The
native trajectory remains arbitrary scale; KITTI alignment is offline
evaluation only. See `VALIDATION_V2_3.md` for the A–E ablation, independent
3500 interval, rejected approaches, and measured limitations.

Outputs are written to the configured output directory:

- `matches/pair_*.jpg`: consecutive images with geometrically verified inlier matches in green and counts in the title.
- `trajectory.png`: 3D camera path with initial/current locations and arbitrary-scale coordinate labels.
- `trajectory.csv`: accepted camera positions, explicitly named as arbitrary-scale coordinates.
- `poses_camera_to_world.csv`: accepted 4×4 camera-to-world poses for validation.
- `metrics.json`: per-pair timings, state, geometric diagnostics, match/inlier counts, reasons, totals, core FPS, and p50/p95 core latency.

Two-view output additionally includes `reconstruction_metrics.json`, filtered
`landmarks.csv`, complete `reconstruction.npz`, and
`sparse_reconstruction.png`.

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

`recoverPose` determines only the direction of translation. For `VALID_MOTION`, this implementation normalizes translation to length one. For `ROTATION_ONLY`, position is held. Positions are consequently in **arbitrary, non-metric scale units**, and step length does not reflect actual speed or distance.

## Failure behavior

Missing descriptors or unreadable input are `TRACKING_LOST`; questionable or unobservable two-view geometry is `DEGENERATE`. Neither changes the pose. `ROTATION_ONLY` changes orientation but not position, and marks the global position trajectory unreliable because translation is missing. Reliability remains sticky: a later valid local pair cannot repair a missing transform without relocalization. After an unreadable frame the pair chain is reset rather than matching across the missing frame.

## Performance interpretation

Core timing contains feature extraction, matching, and calibrated geometry. Visualization time is measured separately and excluded from core FPS. The first pair includes extraction of both its frames; later frames are extracted once and reused. Results depend on the CPU, image size, feature count, sequence texture, and output settings. No benchmark numbers are included without running a real data set.

## Tests

```bash
pytest -q
```

Tests cover intrinsic validation, missing/insufficient descriptors, synthetic calibrated relative-pose recovery, pure rotation, tiny translation, normal 3D translation, planar geometry, rotation-only accumulation, failure immutability, and a short generated image-sequence smoke run. Synthetic feature recovery validates two-view geometry; it does not establish real-world trajectory accuracy.

## Known limitations

- Monocular translation has no metric scale; forcing unit steps distorts trajectory length and speed.
- V1 consecutive-frame tracking and the V2.2 unoptimized local map both accumulate drift; there is no bundle adjustment, loop closure, or relocalization.
- Pure rotation, little parallax, repeated texture, blur, dynamic objects, and weak texture can make essential-matrix estimation degenerate or unreliable.
- Calibration must match the camera and image resolution. Lens distortion should be removed before input; this version does not load distortion coefficients.
- A recovered essential-matrix pose can still be geometrically plausible but wrong. Inlier thresholds are safeguards, not an accuracy guarantee.
- A low two-view reprojection error does not establish accurate depth. V2.0 also requires positive depth, triangulation angle, bounded unit-baseline depth, and a suitable input pair.

## Project structure

```text
.
├── main.py
├── configs/kitti_example.yaml
├── configs/kitti_04_validation.yaml
├── configs/kitti_00_v20_initialization.yaml
├── configs/kitti_00_v21_tracking.yaml
├── configs/kitti_00_v22_baseline.yaml
├── configs/kitti_00_v22_association.yaml
├── configs/kitti_00_v22_mapping.yaml
├── configs/kitti_00_v22_combined.yaml
├── configs/kitti_00_v23_quality.yaml
├── configs/kitti_00_v23_efficient_association.yaml
├── configs/kitti_00_v23_spatial.yaml
├── configs/kitti_00_v23_combined.yaml
├── requirements.txt
├── scripts/evaluate_kitti.py
├── scripts/evaluate_two_view_kitti.py
├── scripts/initialize_two_view.py
├── scripts/track_local_map.py
├── scripts/evaluate_map_tracking_kitti.py
├── src/
│   ├── config.py
│   ├── features.py
│   ├── frame_source.py
│   ├── geometry.py
│   ├── metrics.py
│   ├── odometry.py
│   ├── initialization.py
│   ├── reconstruction.py
│   ├── reconstruction_visualization.py
│   ├── local_map.py
│   ├── local_mapping.py
│   ├── landmark_quality.py
│   ├── map_tracking.py
│   └── visualization.py
└── tests/
    ├── test_config.py
    ├── test_features.py
    ├── test_geometry.py
    ├── test_initialization.py
    ├── test_local_mapping.py
    ├── test_map_tracking.py
    ├── test_odometry.py
    ├── test_reconstruction.py
    └── test_smoke.py
```
