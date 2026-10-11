# PARALLAX V1.3 geometric reliability validation

Validation date: 11 October 2026. This work remains Version 1 monocular visual odometry. It adds no map, bundle adjustment, loop closure, relocalization, learned model, or metric-scale source.

## Runtime decisions

The original essential-matrix estimator is retained behind `geometry.reliability_checks: false`. Reliability mode records essential cheirality support, rotation-compensated bearing residuals, homography support, a constrained bearing-rotation fit, and rotation innovation against the component-wise median Rodrigues vector of the last five accepted rotations.

The thresholds in `configs/kitti_00_turns_v13_reliable.yaml` are:

- Rotation-only angular residual: 0.2°, approximately 2.5 pixels at KITTI's 718.856 px focal length. This permits ORB localization error while remaining much smaller than the suspicious residuals.
- Rotation-only consensus: at least 60% of all ratio-test correspondences. Homography support alone never triggers this state; the fitted SO(3) rotation must satisfy the angular residual on the original bearings.
- Questionable rotation innovation: greater than 2° relative to the recent five-pair median, equivalent to a 20°/s one-frame innovation at KITTI's 10 Hz rate.
- Suspicious compensated parallax: the lowest-decile residual must also exceed 1°. Requiring both this and temporal innovation avoids rejecting ordinary smooth rotation based on one signal.

`recoverPose` continues to provide the essential cheirality test. Finite values, SO(3) determinant, and nonzero translation are checked as before. A RANSAC homography initializes the constrained rotation model and is logged as a diagnostic, but cannot by itself establish pure rotation because a plane under translation also supports a homography.

States have operational meaning:

- `VALID_MOTION`: integrate rotation and arbitrary-unit translation direction.
- `ROTATION_ONLY`: integrate the constrained rotation and hold position; no unit translation is invented.
- `DEGENERATE`: do not integrate the questionable transform.
- `TRACKING_LOST`: descriptors/input are insufficient to continue the local pair.

Any rotation-only or missing motion makes global position reliability sticky. Later locally valid pairs do not claim to repair the gap.

## Failure-case investigation

Pair 3471 had 882 descriptor matches and 632 essential/cheirality inliers (71.7%), yet its V1.2 relative rotation error was 7.698° and translation-direction cosine was 0.144. Its essential rotation leaves 5.450° p10 and 7.367° median compensated bearing residual, has 7.622° temporal innovation, and differs from the auxiliary rotation hypothesis by 7.330°. V1.3 labels it `DEGENERATE`. The high inlier count only says that one epipolar model has consensus; it does not establish a well-conditioned, correct decomposition.

Pair 3527→3528 had 1,721 matches but only one essential cheirality inlier during a 0.228 m, 1.060° ground-truth step. A constrained rotation explains 1,203/1,721 correspondences (69.9%) with 0.113° median angular residual. V1.3 labels it `ROTATION_ONLY`; its recovered rotation has 0.154° evaluation error and translation is deliberately absent.

The other six new degeneracy rejections are global pairs 3282, 3285, 3290, 3395, 3551, and 3575. Together with 3471, these are exactly the seven baseline accepted pairs meeting the declared false-accept rule below on this segment. This clean separation is promising but is validation on one real segment, not proof that the thresholds generalize.

## Controlled KITTI comparison

Both configurations processed the same 400 left-camera images, sequence 00 frames 3200–3599, with the same calibration, 2,500 ORB cap, matcher, essential estimator, and visualization disabled. Ground truth was read only by `scripts/evaluate_kitti.py` after both VO runs.

| Measurement | V1.2 baseline | V1.3 reliability |
|---|---:|---:|
| Pairs | 399 | 399 |
| `VALID_MOTION` | 398 | 385 |
| `ROTATION_ONLY` | 0 | 7 |
| `DEGENERATE` | 1 | 7 |
| `TRACKING_LOST` | 0 | 0 |
| Full translation-direction coverage | 99.75% | 96.49% |
| Orientation update coverage | 99.75% | 98.25% |
| Relative rotation mean / P95 / max | 0.295° / 0.537° / 7.698° | 0.217° / 0.476° / 1.571° |
| Translation-direction error mean / P95 / max | 6.794° / 17.423° / 81.866° | 5.594° / 13.903° / 55.056° |
| False accepted poses | 7 | 0 |
| Core FPS | 9.917 | 7.864 |
| Core latency P95 | 116.402 ms | 170.397 ms |
| Geometry mean | 17.505 ms | 40.883 ms |
| Peak RSS | 107.64 MiB | 110.58 MiB |

For this report, a false accepted pose is an accepted `VALID_MOTION` edge with ground-truth relative rotation error greater than 2° or translation-direction cosine below 0.5 (direction error above 60°). This definition is evaluation-only and is not passed to the runtime estimator.

The full-timeline accumulated orientation error is baseline 3.247° final / 12.058° maximum and V1.3 17.140° final / 17.471° maximum. The V1.3 number includes seven deliberately missing rotations, so it measures both drift and coverage gaps. A separately labeled coverage-conditioned comparison composes ground-truth rotation only over accepted edges: baseline 3.309° final / 12.058° maximum, V1.3 4.519° final / 6.127° maximum. The latter isolates the drift of rotations actually integrated but must not be mistaken for a complete-frame trajectory.

The secondary full-match homography/rotation diagnostic is the performance cost. A 300-match shortcut was tested and rejected because spatially biased subsets changed decisions, missed 3527, and introduced questionable rotation-only labels. Accuracy was kept ahead of throughput.

## Trajectory comparison and scale

Native output remains arbitrary scale. No VO translation is interpreted as meters.

| Evaluation-only alignment | V1.2 baseline | V1.3 reliability |
|---|---:|---:|
| Fixed axes/origin + one global scale ATE RMSE | 23.345 m | 10.418 m |
| Umeyama 7-DoF Sim(3) ATE RMSE | 6.586 m | 5.733 m |

The lower V1.3 trajectory errors are not a metric-scale result and are not sufficient evidence of a universally better trajectory: seven translations are held or rejected, and the Sim(3) fit can absorb a global rotation, translation, and scale. The trajectory remains unsuitable for metric distance claims.

Generated evidence in each output directory includes `trajectory_comparison.png`, `rotation_tracking_diagnostics.png`, `comparison_metrics.json`, `metrics.json`, poses, and resource usage. Reliability mode additionally produces `geometric_reliability_diagnostics.png`.

## Synthetic and regression tests

`pytest -q` passes 22 tests. Added cases cover exact pure rotation, very small translation, normal translation with depth variation, planar translation with essentially perfect homography support, rotation-only accumulation without translation, and sticky reliability. The planar case is not called rotation-only merely because a homography fits.

## Limitations and recommendation

The checks materially improve accepted-edge precision on this segment, but their temporal assumptions are KITTI-like (10 Hz, smooth vehicle motion) and need validation on other cameras, frame rates, dynamic scenes, and abrupt legitimate rotations. A full homography fit costs about 23 ms/pair here. Degenerate edges still create irrecoverable gaps because V1 has no relocalization. Unit translation still distorts all valid-motion trajectory lengths, and rotation-only updates intentionally omit real but unobservable displacement.

Recommendation: do not call the global pose chain reliable, and do not yet begin integrating sparse landmarks into a persistent map. V1.3 can identify edges that are safer to triangulate, but mapping would need an explicit keyframe/baseline policy, triangulation uncertainty and positive-depth/reprojection filtering, and a policy for pose gaps. Those belong to the next design review; none were implemented here.
