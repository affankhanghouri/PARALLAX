# PARALLAX V2.2 reliable local-map expansion validation

Validation date: 11 October 2026. V2.2 adds conservative keyframes, landmark
replenishment in the existing arbitrary-scale segment, bounded verified
descriptor references, and temporal optical-flow observations. It does not add
bundle adjustment, loop closure, relocalization, metric scale, or disconnected
map merging.

## Engineering result

V2.2 extends KITTI 00 tracking after the 3200→3205 initialization from four to
14 consecutive accepted frames (3206–3219). Five mapping events add 278
landmarks to the original 60. Frame 3220 is rejected even with 10 PnP inliers
because they occupy only three image-grid cells; frames 3221–3222 then have only
three and four unique correspondences. No pose or geometry is created for those
frames, the segment becomes `LOST`, and continuity remains false.

This is a meaningful local-mapping improvement, not long-term SLAM. The retained
configuration tracks for about 1.4 seconds after initialization and averages
8.40 FPS. It remains limited by recognition/coverage loss and unoptimized map
growth.

## V2.1 failure diagnosis

All 60 initial landmarks project into the image through frame 3212. Descriptor
evidence nevertheless deteriorates:

| Frame | nearest distance ≤64 | ratio pass | mutual/unique | median nearest Hamming |
|---:|---:|---:|---:|---:|
| 3206 | 55 | 34 | 31 | 33.0 |
| 3207 | 53 | 22 | 18 | 44.0 |
| 3208 | 49 | 24 | 19 | 48.0 |
| 3209 | 45 | 15 | 15 | 56.5 |
| 3210 | 37 | 7 | 6 | 58.5 |
| 3211 | 38 | 7 | 6 | 59.0 |
| 3212 | 36 | 8 | 8 | 60.5 |

At frame 3210, distance thresholding alone retains 37 landmarks, while the
ratio test retains seven. The failure is mainly descriptor drift and ambiguity,
not image exit, a descriptor-indexing error, or an excessively low absolute
distance threshold. A 35 px projection gate improved intermediate matches but
fixed initialization descriptors still failed at 3210.

## Runtime architecture and dependency order

The original global/fixed V2.1 mode remains available as Experiment A. V2.2
features are configuration-controlled.

1. ORB extracts the current image features.
2. Existing segment landmarks are projected using the last accepted
   `T_segment_camera`; descriptor search is restricted to a configurable image
   radius and remains ratio-, distance-, and uniqueness-filtered.
3. Forward–backward Lucas–Kanade tracks supplement descriptor associations only
   for landmark observations validated in the previous accepted pose.
4. The unchanged PnP RANSAC, reprojection, numerical, motion-sanity, inlier-ratio,
   and spatial-coverage checks accept or reject the camera pose.
5. Only after pose acceptance may verified inlier descriptors enter the bounded
   reference bank.
6. Conservative keyframe selection then considers frame gap, camera-center
   translation, rotation, and declining landmark support.
7. Unmapped ORB observations are matched between accepted keyframes, checked by
   the known-pose epipolar geometry, and triangulated directly into the owning
   segment frame.
8. New landmarks pass finite homogeneous, positive-depth, reprojection,
   triangulation-angle, bounded-depth, duplicate, batch-size, and grid-coverage
   checks before insertion. They can first contribute to the following frame.

The optical-flow observation uses `feature_index: null` and explicit
`OPTICAL_FLOW` provenance rather than pretending it is an ORB keypoint. Newly
triangulated current-frame observations seed the following optical-flow step;
they never justify their own creation pose.

## Coordinates and scale

Every stored pose is `T_segment_camera`. OpenCV PnP returns
`T_camera_segment`, which is inverted. For mapping keyframes 1 and 2:

```text
T_camera2_camera1 = inverse(T_segment_camera2) T_segment_camera1
P1 = K inverse(T_segment_camera1)[0:3, :]
P2 = K inverse(T_segment_camera2)[0:3, :]
```

Triangulated homogeneous points are therefore already in the original segment
frame. No later keyframe pair is normalized to a new unit baseline. Ground truth
is unavailable to every runtime module.

## Retained thresholds

- Projected descriptor radius: 50 px. A 75 px trial did not extend lifetime and
  increased association cost and ambiguity.
- Descriptor evidence: Lowe ratio 0.75 and Hamming distance ≤64.
- Descriptor bank: at most three references, added only from accepted PnP
  inliers. The second initialization descriptor remains the initial reference.
- Optical flow: 21×21 window, four pyramid levels, forward–backward error ≤1.5
  px, LK error ≤20, and the same 50 px projection sanity gate.
- Normal keyframe baseline: ≥0.45 arbitrary units, at least two frames apart.
  Under ≤10% landmark support, ≥0.30 units is allowed, still at least two frames
  apart. The lower baseline is protected by per-landmark angle filtering.
- Mapping: at least 20 unmapped descriptor matches; epipolar error ≤1.5 px;
  reprojection ≤2 px in both views; triangulation angle ≥1°; positive depths ≤100
  arbitrary units; at least 12 accepted new landmarks in six 4×6 grid cells;
  duplicate observation radius 3 px; map capacity 1,000.
- Pose validation remains the V2.1 8/6 correspondence/inlier minimum, 50%
  inlier ratio, 2/4 px median/P95 reprojection, and four occupied grid cells.

## Controlled ablations

All experiments use identical KITTI 00 images, calibration, initialization, and
post-run evaluator. “Accepted” counts only subsequent PnP poses.

| Experiment | Association / mapping | Accepted | Last accepted | Final landmarks | First rejection |
|---|---|---:|---:|---:|---|
| A | V2.1 global, fixed 60 | 4/7 | 3209 | 60 | 3210: 6 correspondences |
| B | projected + verified references + flow, fixed map | 8/11 | 3213 | 60 | 3214: 7 correspondences |
| C | projected fixed references + replenishment, no flow | 9/12 | 3214 | 207 | 3215: 44.4% inliers and 3 cells |
| D | combined retained system | 14/17 | 3219 | 338 | 3220: 3 grid cells |

Combined mapping events:

| Keyframe | New landmarks | Total afterward |
|---:|---:|---:|
| 3208 | 86 | 146 |
| 3211 | 55 | 201 |
| 3214 | 37 | 238 |
| 3216 | 62 | 300 |
| 3218 | 38 | 338 |

The final rejected pose at 3220 had 18 fused associations and 10 RANSAC
inliers, but accepting three-cell support would weaken the established geometry
policy. It remains rejected.

## Ground-truth evaluation

Ground truth is loaded only by `evaluate_map_tracking_kitti.py` after runtime.

| Metric | A baseline | B association | C mapping | D combined |
|---|---:|---:|---:|---:|
| Absolute rotation mean | 0.669° | 0.598° | 1.321° | 0.771° |
| Absolute rotation P95 | 1.006° | 0.770° | 2.533° | 1.238° |
| Absolute rotation max | 1.080° | 0.824° | 2.794° | 1.279° |
| Relative rotation mean | 0.685° | 0.213° | 0.622° | 0.163° |
| Relative rotation max | 1.245° | 0.738° | 1.229° | 0.359° |
| Initialization-scale position mean | 0.366 m | 0.294 m | 0.667 m | 0.480 m |
| Initialization-scale position P95 | 0.637 m | 0.441 m | 1.017 m | 0.937 m |
| Scale-only ATE RMSE | 0.314 m | 0.279 m | 0.570 m | 0.395 m |
| Sim(3) ATE RMSE | 0.228 m | 0.099 m | 0.336 m | 0.145 m |

The initialization-baseline alignment uses one evaluation-only scale of
5.84066 ground-truth meters per map unit with axes and origin fixed. Scale-only
ATE fits one global scale, also with axes/origin fixed. Sim(3) is an explicitly
labeled Umeyama 7-DoF evaluation alignment. Every trajectory is nearly
collinear (combined second/first singular-value ratio 0.00359), so Sim(3)
rotation and ATE are poorly conditioned. Runtime coordinates remain arbitrary
scale and are never meters.

Combined relative position error under the initialization scale is 0.080 m mean
and 0.187 m maximum. Its absolute position error is 0.480 m mean, 0.937 m P95,
and 1.039 m maximum. The gap between relative and accumulated error shows drift
from using unoptimized PnP poses to create later landmarks.

## Performance and memory

| Experiment | FPS | mean / P95 latency | Peak RSS |
|---|---:|---:|---:|
| A | 28.31 | 35.32 / 42.79 ms | 151.48 MiB |
| B | 19.40 | 51.55 / 63.42 ms | 153.17 MiB |
| C | 11.70 | 85.48 / 185.77 ms | 152.91 MiB |
| D | 8.40 | 119.12 / 239.40 ms | 154.60 MiB |

Combined mean stage timings are 23.37 ms feature extraction, 41.19 ms projected
descriptor association, 6.63 ms optical flow, 4.18 ms PnP, 0.57 ms pose
validation, 0.11 ms observation update, and 41.51 ms local mapping amortized over
all attempted frames. Keyframe mapping causes the high P95. V2.2 trades speed
for coverage and no longer sustains the V2.1 10 FPS target in the retained
combined mode.

## Rejected alternatives

- Replacing the descriptor with every accepted observation alone caused early
  spatially concentrated PnP rejection.
- Multiple unverified initialization descriptors were not restored.
- A global 0.30-unit keyframe baseline grew the map too quickly and shortened
  accepted tracking.
- Tightening mapping to 1 px epipolar error and 1.5° angle starved later
  replenishment.
- A 75 px descriptor radius added cost without extending lifetime.
- Allowing one-frame emergency keyframes reached 24 accepted poses but grew to
  896 landmarks, fell to about 5.2 FPS, and increased initialization-scale
  position error to 1.23 m mean and 3.95 m maximum. It was rejected as an
  accuracy/complexity tradeoff, not selected for its higher success count.

## Tests and invariants

`pytest -q` passes 43 tests. New regression coverage verifies projected
association uniqueness, accepted-pose-only bounded descriptor updates,
forward–backward optical-flow landmark identity, existing-scale known-pose
triangulation, duplicate observation prevention, conservative keyframe policy,
and prohibition of keyframes from rejected poses.

The retained implementation maintains segment ownership, persistent IDs,
valid observation provenance, one coordinate scale, and failure immutability.
Rejected frames cannot create keyframes or landmarks. The final KITTI map's
runtime integrity audit reports `integrity_valid: true` with no violations.

## Reproduction

```bash
for experiment in baseline association mapping combined; do
  mkdir -p "outputs/kitti_00_v22_${experiment}"
  env MPLCONFIGDIR=/tmp/parallax-mpl /usr/bin/time -v \
    -o "outputs/kitti_00_v22_${experiment}/resource_usage.txt" \
    .venv/bin/python scripts/track_local_map.py \
    --config "configs/kitti_00_v22_${experiment}.yaml" \
    --anchor-offset 0 --max-candidates 30 \
    --max-tracking-frames 100 --max-consecutive-failures 3

  env MPLCONFIGDIR=/tmp/parallax-mpl \
    .venv/bin/python scripts/evaluate_map_tracking_kitti.py \
    --output-dir "outputs/kitti_00_v22_${experiment}" \
    --ground-truth data/kitti/srrg00/00_gt.txt --start-index 3200
done

.venv/bin/pytest -q
```

Runtime tracking commands receive no ground-truth path.

## Recommendation

V2.2 demonstrates that accepted map poses can safely replenish landmarks in one
consistent local scale, and that verified temporal observations materially
improve landmark persistence. It is not yet a reliable long-duration tracker.
The next bottlenecks are map-quality/uncertainty management, scalable projected
association, and accumulated pose/landmark drift. Any next milestone should
first bound map growth and optimize a local window before attempting loop
closure or global map claims.
