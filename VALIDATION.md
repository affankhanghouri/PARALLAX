# V1.1 real-world validation record

Validation was run on 11 October 2026 using the first 200 left-camera frames of KITTI odometry sequence 04. The compact proSLAM/SRRG mirror supplied 271 rectified grayscale frames, `calib.txt`, and KITTI ground truth. Images were 1226×370. The configured matrix was copied from `P0`:

```text
fx = fy = 707.0912, cx = 601.8873, cy = 183.1104
```

Only `camera_left` frames were staged in `data/kitti/sequences/04/image_0`; right-camera images were not supplied to the application.

## Commands

```bash
pytest -q
mkdir -p outputs/kitti_04_200 /tmp/parallax-mpl
env MPLCONFIGDIR=/tmp/parallax-mpl /usr/bin/time -v \
  -o outputs/kitti_04_200/resource_usage.txt \
  .venv/bin/python main.py --config configs/kitti_04_validation.yaml
env MPLCONFIGDIR=/tmp/parallax-mpl .venv/bin/python scripts/evaluate_kitti.py \
  --output-dir outputs/kitti_04_200 \
  --ground-truth data/kitti/poses/04.txt
```

## Measured results

- Frames: 200; consecutive pairs: 199
- Pose estimates: 199 successful, 0 failed
- Core throughput: 9.807 FPS
- Core latency: 101.965 ms mean, 98.380 ms p50, 121.399 ms p95
- Mean stage times: 21.006 ms extraction, 64.326 ms matching, 16.633 ms geometry
- Visualization: 3.289 ms mean per pair when averaged over all pairs (only every twentieth pair was rendered)
- Ratio-test matches: 816 min, 1,217 median, 1,218.543 mean, 1,558 max
- Geometric inliers: 235 min, 549 median, 564.633 mean, 1,083 max
- Mean inlier ratio: 46.25%; minimum inlier ratio: 18.49%
- Peak resident memory (`/usr/bin/time -v`): 118,192 KiB (115.42 MiB)
- End-to-end wall time: 24.03 seconds. Core FPS intentionally excludes startup, final plotting, and periodic match rendering.

## Trajectory comparison

Native VO is arbitrary-scale and advances one unit for every accepted pair. KITTI ground truth is metric; raw VO positions were never interpreted as meters.

Two evaluation alignments are reported:

1. Fixed axes and origin, one fitted global scale: 1.3850 KITTI ground-truth meters per VO unit; ATE RMSE 1.9704 ground-truth meters.
2. Umeyama 7-DoF Sim(3) (one global scale, rotation, translation): scale 1.3942 ground-truth meters per VO unit; ATE RMSE 1.4783 and median 1.1072 ground-truth meters.

The first 200 sequence-04 frames are an almost straight drive. Actual ground-truth frame displacement varied from 1.3108 to 1.5508 meters (mean 1.4070), so the unit-step approximation happens to fit this segment comparatively well; that does not establish metric scale recovery.

Raw, pre-alignment step-direction cosine against ground truth was 0.99834 mean, 0.99936 median, and 0.97745 minimum, with zero opposite-direction steps. The native trajectory moves forward along the expected camera/world Z direction, bends consistently with the subtle ground-truth path, and exhibits modest vertical/lateral drift. No sign inversion, incorrect transform composition, severe direction error, or tracking loss was observed in this segment. Sequence 04's near-straight geometry makes it a useful convention check but not a comprehensive turning-motion validation.

## Changes justified by validation

- Added `.pgm` to the existing directory loader because the compact real KITTI mirror uses PGM. A regression test verifies PGM discovery and loading.
- Added a sequence-04 validation configuration containing the exact `P0` intrinsics and left-image path.
- Added a validation-only comparison utility with explicit scale-only and Sim(3) labeling, plus a numerical regression test for the alignment.

No mapping, landmarks, bundle adjustment, loop closure, relocalization, or metric-scale estimation was added.

