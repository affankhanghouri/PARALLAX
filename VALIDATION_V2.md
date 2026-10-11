# PARALLAX V2.0 two-view sparse reconstruction validation

Validation date: 11 October 2026. V2.0 initializes landmarks from one selected calibrated image pair. It does not maintain a persistent map or add PnP, bundle adjustment, loop closure, relocalization, dense reconstruction, or metric scale.

## Reused V1.3 architecture and coordinates

V2.0 reuses `ORBFeatures`, Lowe-ratio Hamming matching, the calibrated essential/recoverPose pipeline, explicit V1.3 motion states, camera intrinsics from `AppConfig`, and the established transform convention

```text
X_second = R_second_first X_first + t_second_first.
```

The reconstruction frame is camera 1. Projection matrices are `P1 = K[I|0]` and `P2 = K[R|t]`; camera centers in the first-camera frame are `C1 = 0` and `C2 = -R^T t`. Monocular `t` is normalized to one. Landmark coordinates and depths are therefore arbitrary reconstruction units and are never claimed as meters.

Temporal rotation innovation is not used to select a separated pair because anchor-to-candidate rotations at different gaps are not consecutive motion increments. The rest of `estimate_reliable_motion`, including essential cheirality, constrained-rotation detection, numerical checks, and explicit states, is reused unchanged. V1.3 thresholds were not modified.

## Pair selection

Starting from an anchor, candidates are tested sequentially and the first pair satisfying all image-only conditions is selected. This is not an every-Nth-frame policy.

- At least 50 Lowe-ratio matches and 50 geometric inliers provide a useful correspondence sample.
- State must be `VALID_MOTION`; `ROTATION_ONLY`, `DEGENERATE`, and `TRACKING_LOST` are never triangulated.
- Median rotation-compensated parallax must be at least 1.0°, and its 25th percentile at least 0.5°. At KITTI's 718.856 px focal length, one pixel is about 0.080°, making these approximately 12.5× and 6.25× the one-pixel angular scale.
- Inliers must occupy at least 8 of a 4×6 grid in both images. The minimum coverage of the two views is used, preventing a localized feature cluster from passing.

The thresholds are explicit in `configs/kitti_00_v20_initialization.yaml`. Ground truth is not accepted by the selector API.

## Landmark filters

Essential/cheirality inliers are triangulated with OpenCV in the camera-1 coordinate frame. Filters are applied sequentially:

1. Finite input and homogeneous coordinates with non-negligible `w`.
2. Positive depth in both cameras.
3. Reprojection error no greater than 2 px in either image. This is twice the 1 px essential RANSAC threshold and allows ORB localization noise without accepting large inconsistencies.
4. Triangulation angle at least 1°. This guards depth conditioning; low reprojection error alone can occur at nearly parallel rays.
5. Depth no greater than 100 arbitrary unit-baseline units, an explicit numerical/uncertainty bound.

Every stage count is saved. The point cloud contains only landmarks passing all filters.

## Real KITTI positive example

Input was sequence 00 left-camera imagery starting at global frame 3200, using the matching P0 intrinsic matrix. With anchor offset 0 and up to 30 candidates, the runtime selector rejected candidates 3201–3204 for insufficient median or lower-quartile parallax and selected 3200→3205.

Pair evidence:

- 165 Lowe-ratio matches; median Hamming distance 34.
- 104 essential/cheirality inliers.
- `VALID_MOTION`.
- 1.526° median and 0.573° p25 compensated parallax.
- 9/24 occupied grid cells in the weaker view.

Triangulation filtering:

| Stage | Remaining points |
|---|---:|
| Candidates | 104 |
| Finite homogeneous | 104 |
| Positive in both cameras | 104 |
| Reprojection-consistent | 104 |
| Angle ≥1° | 60 |
| Depth ≤100 arbitrary units | 60 |

Accepted ratio was 57.69%; positive-depth ratio was 100%. For accepted landmarks, first/second-view reprojection error medians were 0.223/0.278 px and P95 values were 0.574/0.698 px. Triangulation angle was 3.187° median, 5.027° P95, and 5.891° maximum. First-camera depth was 5.985 arbitrary units median and 14.955 maximum.

The separate evaluation script, run only after reconstruction, reported a 5.841 m ground-truth baseline, 0.349° relative rotation error, translation-direction cosine 0.99970, and 1.403° direction error. The ground-truth baseline is not the scale of the reconstructed point cloud.

## Real KITTI negative example

The same runtime selector rejected global pair 3527→3528:

- 1,721 matches and 1,203 constrained-rotation inliers.
- State `ROTATION_ONLY`.
- Median/p25 compensated parallax only 0.113°/0.084°.
- No triangulation or point-cloud output was produced.

Post-run ground truth confirms a 0.228 m translation with 1.060° rotation. The negative result demonstrates why match or inlier count alone cannot authorize triangulation.

## Performance

Positive-run internal timing was 985.261 ms for scanning five candidates, 0.696 ms for triangulation, 602.154 ms for visualization, and 985.957 ms for selection plus triangulation. End-to-end wall time was 2.72 s and peak RSS 116,400 KiB (113.67 MiB). The rejected negative pair took 146.854 ms internally, 1.69 s end to end including process/import overhead, and 100,824 KiB (98.46 MiB) peak RSS.

This is initialization latency, not per-frame VO throughput. Feature extraction, matching, and full V1.3 reliability diagnostics dominate; triangulation itself is below 1 ms for 104 candidates.

## Automated tests

`pytest -q` passes 30 tests: all 22 V1.3 tests plus eight reconstruction tests. Added synthetic cases cover a normal non-planar scene with known scale relationship, very small baseline, pure rotation, noisy observations, planar geometry with observable baseline, incorrect correspondences, non-finite input, and maximum-depth filtering.

The small-baseline test is especially important: all points have essentially zero reprojection error, yet all are rejected by the triangulation-angle check. This explicitly prevents low reprojection error from being treated as proof of depth accuracy.

## Failure modes and limits

- The recovered scene has one arbitrary scale tied to the selected pair's unit translation; it is not metric.
- Incorrect but self-consistent relative pose can still produce low reprojection error. Pair reliability and multi-view validation remain necessary.
- Dynamic objects and repeated texture can create coherent false structure.
- The current coverage grid measures distribution, not semantic scene rigidity.
- Thresholds are justified geometrically and validated on synthetic cases plus one positive and one negative KITTI example, but require broader camera/frame-rate validation.
- A rejected or missing pose chain is not repaired. This initializer creates a new local segment only; it must not be silently attached across a V1.3 reliability gap.

## Proposed V2.1 persistent-map data model

The next design can introduce, without changing the current local geometry:

- `Keyframe`: immutable keyframe ID, source frame/time, calibration ID, camera-to-segment pose, feature descriptors, and segment ID.
- `Landmark`: landmark ID, 3D position in its segment, quality/uncertainty fields, validity state, and a collection of observation IDs.
- `LandmarkObservation`: keyframe ID, landmark ID, feature index, measured pixel, pyramid level, and reprojection status.
- `PoseState`: camera-to-segment transform, tracking state, covariance/quality summary, and provenance.
- `MapSegment`: segment ID, local arbitrary-scale frame, keyframes, landmarks, validity, and the reason it began or ended.

Every landmark must support observations from multiple keyframes. If tracking becomes `DEGENERATE` or `TRACKING_LOST`, the current segment must close. A later initialization creates a different segment with no assumed transform or shared scale. Segments may be connected only by an explicit future relocalization/alignment result; never by continuing accumulation across the missing edge.

V2.0 does not implement these persistent containers or behaviors.
