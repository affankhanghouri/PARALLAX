# PARALLAX V2.3 Validation

## Scope and reproducibility

V2.3 is limited to quality-aware management and performance of the existing
arbitrary-scale local map. It does not add bundle adjustment, loop closure,
relocalization, metric-scale recovery, or a new SLAM architecture. Ground truth
is used only by the offline evaluator.

## Checkpoint 1 — retained V2.2 baseline and diagnosis

Command:

```bash
/usr/bin/time -v .venv/bin/python scripts/track_local_map.py \
  --config configs/kitti_00_v22_combined.yaml \
  --anchor-offset 0 --max-candidates 30 --max-tracking-frames 50
.venv/bin/python scripts/evaluate_map_tracking_kitti.py \
  --output-dir outputs/kitti_00_v22_combined \
  --ground-truth data/kitti/srrg00/00_gt.txt --start-index 3200
pytest -q
```

The retained baseline reproduced 14 accepted tracking frames (3206–3219), then
three rejected attempts. The first rejection, frame 3220, had 18 fused 3D–2D
associations and 10 PnP inliers but occupied only three of the 4x6 validation
grid cells. This is a correct rejection, not a tracking success. The segment is
therefore marked `LOST` and `trajectory_continuous=false`; it is not silently
reconnected.

The map grew from 60 to 338 landmarks in five batches (+86, +55, +37, +62,
+38). The measured core rate was 8.395 FPS, mean latency 119.116 ms, P95
239.397 ms, and peak RSS 154.60 MiB. Mean stage costs were feature extraction
23.372 ms, association 41.193 ms, optical flow 6.632 ms, PnP 4.177 ms,
validation 0.572 ms, map update 0.112 ms, and mapping 41.512 ms. Association and
mapping therefore dominate and both increase with unbounded map growth.

Against evaluation-only KITTI poses, the arbitrary-scale trajectory had
0.39549 m scale-only aligned ATE RMSE and 0.14533 m Sim(3)-aligned ATE RMSE.
The latter is poorly conditioned over this short path (estimated/GT path ratio
0.003589), so it must not be interpreted as independent metric accuracy.
Absolute rotation error was 0.77108 degrees mean, 1.23761 degrees P95, and
1.27876 degrees maximum. The test suite passed: **43 passed**.

### Diagnosis

- The failure is spatial-support collapse, not too few raw matches.
- New landmark batches are large and every active landmark remains eligible for
  projection and matching, making cost grow with map size.
- Raw observation count does not distinguish repeatedly useful PnP landmarks
  from visible landmarks that repeatedly fail association or pose validation.
- V2.3 will preserve every landmark identity and observation while allowing
  reversible inactive status; it will not delete evidence or use ground truth
  in runtime decisions.

Further checkpoints below are filled from controlled A–E runs on the same
frames and thresholds.

## Checkpoint 2 — quality and active-set infrastructure

Landmarks now retain creation/last-observation offsets, projection attempts,
association successes, verified PnP-inlier observations, consecutive
predicted-visible misses, and an exponential moving average of reprojection
error. `TEMPORARILY_INACTIVE` was added separately from irreversible
`INVALID`. No observation or identity is deleted.

The active set is recomputed from the previous accepted pose. Out-of-view
landmarks are inactive but reconsidered every frame. Predicted-visible points
are selected round-robin over the 4x6 image grid and ranked by verified runtime
evidence plus triangulation quality. The retained thresholds are 220 active
points and four consecutive predicted-visible misses. Four means a point must
fail across multiple frames rather than one noisy association; it was selected
before ground-truth evaluation and is covered by transition tests.

## Checkpoint 3 — association and mapping performance

Projection and descriptor matching are timed separately. The existing
projection-neighborhood index and one-to-one assignment remain intact. A
one-reference-per-landmark association option was benchmarked, but is not in
the retained configuration because of the accuracy loss in experiment C.

The defensible optimization is exact: after forward mapping-descriptor ratio
and distance filtering, reverse mutual-nearest matching is performed only for
the surviving current descriptors. Matching every unmapped current descriptor
produced the same accepted matches but wasted work. This option is explicit in
configuration, so the V2.2 matching implementation remains reproducible.
Duplicate landmark checks during mapping were also vectorized without changing
their radius or logic.

## Checkpoint 4 — spatial distribution and map growth

The runner now measures occupied 4x6 cells separately for ORB detections,
descriptor associations, fused descriptor/flow associations, and final PnP
inliers. It writes `spatial_coverage.png`.

Spatial insertion caps were implemented and tested. Experiment D limited each
keyframe to 48 points and each grid cell to four. It reduced the map to 139
points and met the speed target, but tracking ended after 11 accepted frames.
The cap discarded useful support and is therefore not enabled in E. The code
and regression test remain available for future scene-dependent policy work.

## Checkpoint 5 — deterministic A–E ablation

OpenCV RANSAC was previously unseeded: identical invocations could change PnP,
keyframes, and map growth. The benchmark runner now calls `cv2.setRNGSeed(0)`.
Two repeated runs produced byte-identical pose CSV files. This explains why the
deterministic A replay below reaches 18 frames whereas the historically
retained V2.2 measurement reached 14; the latter remains the project baseline
quoted at the top of this report. All A–E rows below use the same deterministic
runner, images, calibration, initialization, and geometric thresholds.

| Run | Change | Accepted / attempted | Last accepted | Landmarks | FPS | Mean / P95 ms | Peak MiB | Abs. rotation mean / P95 / max | Scale-only ATE RMSE | Sim(3) ATE RMSE |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A | V2.2 behavior | 18 / 21 | 3223 | 311 | 8.70 | 114.96 / 215.04 | 160.09 | 1.190 / 2.197 / 3.773 deg | 0.452 m | 0.216 m |
| B | quality active set | 14 / 17 | 3219 | 277 | 10.40 | 96.19 / 200.78 | 156.98 | 0.839 / 1.236 / 1.383 deg | 0.322 m | 0.105 m |
| C | one descriptor reference + efficient mapping match | 16 / 19 | 3221 | 297 | 10.72 | 93.31 / 156.92 | 159.97 | 1.585 / 4.061 / 4.727 deg | 0.599 m | 0.273 m |
| D | spatial insertion budget | 11 / 14 | 3216 | 139 | 10.51 | 95.17 / 191.79 | 159.46 | 0.726 / 1.172 / 1.334 deg | 0.341 m | 0.075 m |
| E | quality + exact mapping-match optimization | 14 / 17 | 3219 | 277 | **12.02** | **83.20 / 148.78** | 157.94 | 0.839 / 1.236 / 1.383 deg | **0.322 m** | 0.105 m |

Experiment C is rejected despite two more poses and good speed: its absolute
rotation mean is 89% worse than E, its maximum is 3.4 times worse, and its
scale-only ATE is 86% worse. Experiment D is also rejected because its compact
map loses three accepted poses. E therefore intentionally does not combine
either harmful change. Relative rotation error in E is 0.139 degrees mean and
0.307 degrees maximum.

The E stage means are feature extraction 22.49 ms, complete association 29.36
ms (projection 6.96 ms and descriptor matching 17.87 ms), optical flow 6.32 ms,
PnP 3.23 ms, validation 0.54 ms, keyframe selection 0.10 ms, quality maintenance
0.84 ms, and mapping 16.96 ms averaged over all frames. Mapping descriptor
matching accounts for 15.61 ms and triangulation itself only 0.07 ms. These
overlap inside the reported end-to-end total only as explicitly named stages;
no mapping-heavy frame is excluded from FPS.

## Checkpoint 6 — failure diagnosis and landmark survival

E's first rejection is frame 3220: median reprojection error 2.333 px exceeds
the unchanged 2.0 px limit. ORB still occupies 18/24 cells, while descriptor
associations occupy only two, fused associations five, and PnP inliers four.
The detector and scene therefore offer broad image evidence, but the current
map descriptors do not yield broad, accurate reobservations. Frames 3221 and
3222 then have only three unique correspondences; no pose is integrated and
the segment becomes `LOST` with trajectory continuity false.

Landmark-batch survival is recorded in `tracking_metrics.json`. E creates 60
initial points and batches of 86, 58, 41, and 32. Respectively 37, 71, 46, 29,
and 16 are later reobserved as PnP inliers. This confirms that later batches do
contribute, while also showing substantial churn. Inactivity is not deletion:
the complete 277-point history passes map-integrity validation.

## Checkpoint 7 — ground-truth evaluation and scale treatment

The native map and trajectory are arbitrary scale. No runtime component reads
ground truth. E's initialization-baseline alignment uses one evaluation-only
scale from 3200->3205 with origin and axes fixed: position error is 0.451 m
mean, 0.754 m P95, and 0.812 m maximum. The global scale-only ATE is 0.322 m.
The 7-DoF Umeyama Sim(3) ATE is 0.105 m, but the trajectory singular-value
ratio is 0.00359, so its rotation and ATE are explicitly marked poorly
conditioned and are not evidence of metric-scale recovery.

The evaluator itself had a real bug for nonzero initialization anchors: it
used the extract's first frame as the GT origin and hard-coded the 3200->3205
label. It now uses the actual initialization anchor, and a regression test
covers that coordinate convention. This fix is offline evaluation only.

## Checkpoint 8 — independent interval

Without tuning thresholds on it, frames 3500->3532 were tested by initializing
at offsets 300->302 in the same 400-frame KITTI extract. Both attempted anchors
3300 and 3400 correctly failed the unchanged initializer, so no map was forced.
The 3500 interval initialized 300 landmarks and accepted all 30 attempted
tracking frames through 3532; the segment remained continuous and active.

The map reached its configured 600-point cap. Performance was 9.28 FPS, 107.79
ms mean, 172.56 ms P95, and 158.71 MiB peak RSS. Absolute orientation error was
1.362 degrees mean, 2.083 degrees P95, and 2.171 degrees maximum; relative
rotation error was 0.176 degrees mean and 0.556 degrees maximum. Evaluation-only
scale-only ATE was 0.701 m. Sim(3) ATE was 0.082 m but again poorly conditioned
(singular-value ratio 0.0366). The below-target FPS with 600 landmarks makes
tracking association—not triangulation—the next scaling bottleneck.

Reproduction:

```bash
/usr/bin/time -v .venv/bin/python scripts/track_local_map.py \
  --config configs/kitti_00_v23_combined.yaml \
  --anchor-offset 300 --max-candidates 50 --max-tracking-frames 30 \
  --output-dir outputs/kitti_00_v23_independent_3500
.venv/bin/python scripts/evaluate_map_tracking_kitti.py \
  --output-dir outputs/kitti_00_v23_independent_3500 \
  --ground-truth data/kitti/srrg00/00_gt.txt --start-index 3200
```

## Tests, artifacts, and decision

`pytest -q` reports **46 passed**. New coverage includes quality evidence,
reversible active/inactive transitions, spatially bounded insertion, map
integrity, and nonzero-anchor evaluation coordinates. Existing tests continue
to cover unique assignment, projection gates, rejected-pose immutability,
segment coordinates, and map bounds.

Primary artifacts:

- `outputs/kitti_00_v23_combined/local_map_tracking.png`
- `outputs/kitti_00_v23_combined/map_trajectory_comparison.png`
- `outputs/kitti_00_v23_combined/spatial_coverage.png`
- `outputs/kitti_00_v23_combined/association_diagnostics.png`
- `outputs/kitti_00_v23_combined/tracking_diagnostics.png`
- `outputs/kitti_00_v23_combined/tracking_metrics.json`
- `outputs/kitti_00_v23_combined/tracking_ground_truth_evaluation.json`

V2.3 meets the primary CPU objectives without lowering geometric thresholds and
preserves the established 14-frame coverage, but it does not extend it. The
independent run shows that the foundation can remain geometrically coherent for
30 frames when support is strong. The limiting issue is view-invariant,
spatially broad landmark reobservation and association as the map grows. V2.3
is suitable as a carefully bounded foundation for local bundle-adjustment
research, but bundle adjustment alone will not solve descriptor-support loss;
that next milestone should retain explicit failure semantics and benchmark the
600-landmark association cost. No V2.4 work is included here.
