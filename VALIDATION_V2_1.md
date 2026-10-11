# PARALLAX V2.1 persistent local landmarks and map tracking

Validation date: 11 October 2026. V2.1 tests whether the 60 fixed V2.0 landmarks can localize subsequent frames. It adds no landmark creation after initialization, fusion, bundle adjustment, loop closure, global relocalization, or metric scale.

## Architecture

`MapSegment` owns a segment ID, arbitrary-scale coordinate system, status, continuity flag, two initialization keyframes, fixed landmarks, and accepted/rejected frame-pose records. Each `Keyframe` stores its segment identity, camera-to-segment pose, features, landmark observations, provenance, and quality status. Each `Landmark` stores a persistent segment-scoped ID, 3D position, fixed reference descriptor bank, quality fields, status, and observation history. Each `Observation` records frame/landmark IDs, feature index, pixel, reprojection error, and inlier state.

V2.0 initializes the segment exactly as before. Camera 1 has `T_segment_camera = I`; camera 2 is the inverse of `X_second = R X_first + t`. All landmarks and PnP translations remain in the same arbitrary unit-baseline scale.

For subsequent frames:

1. ORB features are extracted.
2. Active landmark reference descriptors are matched to frame descriptors with the existing 0.75 ratio test, Hamming distance ≤64, reverse mutual consistency, and explicit one-to-one landmark/feature uniqueness.
3. OpenCV `solvePnPRansac` estimates the segment-to-camera transform with EPNP, 100 iterations, 3 px threshold, and 0.999 confidence. Inliers are refined with iterative PnP.
4. The result is checked for finite SO(3), at least 6/8 inliers/correspondences, at least 50% inliers, median/P95 reprojection ≤2/4 px, and at least four occupied cells in a 4×6 image grid.
5. Gross temporal bounds of 15° and 2 arbitrary units reject numerical jumps. They are sanity limits, not a constant-velocity model.
6. The world/segment-to-camera PnP transform is inverted before storing the camera-to-segment pose.

A rejected frame stores no pose. One gap marks trajectory continuity false, but a future absolute map PnP could still be accepted without chaining across the gap. Three consecutive failures mark the segment `LOST` and stop this validation run.

The descriptor bank supports future multiple references, but V2.1 uses the second initialization keyframe descriptor because it is closest in appearance to subsequent frames. An experiment adding both initialization descriptors increased raw associations but reduced spatial consistency and caused earlier rejection; it was not retained.

## KITTI sequence 00 result

Runtime used only images and KITTI P0 calibration. It reproduced initialization at frames 3200→3205 and created 60 persistent landmarks. Ground truth was loaded afterward by a separate evaluator.

| Global frame | Predicted in-view landmarks | Unique associations | PnP inliers | Inlier ratio | Median / P95 reprojection | Grid cells | Decision |
|---|---:|---:|---:|---:|---:|---:|---|
| 3206 | 60 | 31 | 23 | 74.2% | 1.382 / 2.437 px | 4 | accepted |
| 3207 | 60 | 18 | 13 | 72.2% | 0.956 / 2.105 px | 5 | accepted |
| 3208 | 60 | 19 | 12 | 63.2% | 1.308 / 2.572 px | 4 | accepted |
| 3209 | 60 | 15 | 11 | 73.3% | 0.885 / 2.442 px | 4 | accepted |
| 3210 | 60 | 6 | 0 | — | — | 0 | insufficient correspondences |
| 3211 | 60 | 6 | 0 | — | — | 0 | insufficient correspondences |
| 3212 | 60 | 8 | 7 | 87.5% | 1.739 / 2.641 px | 3 | rejected: spatial concentration |

The map therefore supported four consecutive tracked frames, 3206–3209, or about 0.4 seconds at KITTI's 10 Hz rate. Three later failures marked the segment `LOST`. The last rejection is intentional: a high 7/8 inlier ratio does not compensate for observations occupying only three grid cells.

Landmark observations remained bounded because there is no map growth: minimum 2, median 3, maximum 5 observations per landmark. Projection from the last accepted pose predicted all 60 landmarks inside the image, but descriptor associations fell from 31 to six. The dominant failure is appearance/association persistence and the small fixed landmark set, not points simply leaving the image.

## Ground-truth evaluation

The accepted pose offsets are 0, 5, 6, 7, 8, and 9. Four are PnP tracking results.

- Absolute tracked orientation error: 0.669° mean, 1.006° P95, 1.080° maximum.
- Relative rotation error over contiguous tracked edges: 0.685° mean, 1.245° maximum.
- Using one evaluation-only scale from the 3200→3205 ground-truth initialization baseline, tracked position error is 0.366 m mean, 0.637 m P95, and 0.696 m maximum.
- Relative position error under that same initialization scale is 0.360 m mean and 0.670 m maximum.
- Fixed axes/origin least-squares scale-only ATE RMSE is 0.314 m at 5.951 ground-truth meters per map unit.
- Umeyama 7-DoF Sim(3) ATE RMSE is 0.228 m, but the trajectory's second/first singular-value ratio is only 0.00348. It is nearly collinear, so the 81.17° Sim(3) rotation and its low ATE are poorly conditioned and must not be treated as strong evidence.

All meters above come from post-run alignment/evaluation. Runtime map coordinates remain arbitrary scale.

## Reproduction

The runtime command does not receive the ground-truth path. The evaluator is a
separate process run only after tracking has written its estimates.

```bash
mkdir -p outputs/kitti_00_v21_tracking
env MPLCONFIGDIR=/tmp/parallax-mpl /usr/bin/time -v \
  -o outputs/kitti_00_v21_tracking/resource_usage.txt \
  .venv/bin/python scripts/track_local_map.py \
  --config configs/kitti_00_v21_tracking.yaml \
  --anchor-offset 0 --max-candidates 30 \
  --max-tracking-frames 50 --max-consecutive-failures 3

env MPLCONFIGDIR=/tmp/parallax-mpl \
  .venv/bin/python scripts/evaluate_map_tracking_kitti.py \
  --output-dir outputs/kitti_00_v21_tracking \
  --ground-truth data/kitti/srrg00/00_gt.txt --start-index 3200

.venv/bin/pytest -q
```

`tracking_metrics.json` reports `TWO_VIEW_INITIALIZATION` or `MAP_PNP` pose
provenance and an explicit `TRACKED`/`REJECTED` state for every attempted
tracking frame. Rejection records contain no pose.

## Performance

The final provenance-reporting run took 1,581.205 ms for one-time initialization,
which is excluded from tracking FPS.

- Steady-state tracking: 19.389 FPS.
- Mean/P95 total tracking latency: 51.576/60.993 ms.
- Mean feature extraction: 37.851 ms.
- Mean landmark association: 8.502 ms.
- Mean PnP contribution across all attempted frames: 3.985 ms.
- Mean PnP time when executed: 5.578 ms.
- Mean pose validation: 0.705 ms.
- Mean map update: 0.077 ms.
- End-to-end process wall time including initialization, plotting, and imports: 4.78 s.
- Peak RSS: 135,904 KiB (132.72 MiB).

Two immediately preceding runs of the identical geometry produced 22.740 and
31.280 FPS (52.375 and 35.151 ms P95), showing that this short seven-frame
timing sample is sensitive to host load. All three runs exceed 10 FPS with
quality checks enabled. The more important limitation is four-frame map
lifetime, not CPU throughput.

## Tests

`.venv/bin/pytest -q` passes 37 tests: all 30 V2.0/V1.x tests plus seven new map/PnP tests. Added coverage includes known-pose PnP, correct camera-to-world inversion, unique/ambiguous associations, insufficient correspondences, outlier correspondences, spatial-degeneracy rejection, interruption without a fictional pose (including rejection provenance), and initialization identity/observation consistency.

## Failure analysis and V2.2 recommendation

The initialized landmarks are geometrically useful: accepted PnP poses have low rotation error, bounded reprojection error, and consistent arbitrary-scale translation. They are not persistent enough for meaningful sequence tracking. The fixed set rapidly loses descriptor support, and it has no mechanism to refresh viewpoints, add newly triangulated landmarks, or optimize landmark/pose uncertainty.

V2.2 should add conservative keyframe insertion and multi-keyframe landmark creation, descriptor/reference-view management, observation-based landmark health, and local bundle adjustment. New landmarks should only join the current segment through accepted map poses and validated multi-view geometry. Broken/lost segments must remain separate unless an explicit future relocalization establishes a transform; no trajectory should be connected across loss by assumption.

Do not prioritize loop closure or global mapping yet. First demonstrate that local keyframes and landmark replenishment extend map support without increasing false PnP acceptance.
