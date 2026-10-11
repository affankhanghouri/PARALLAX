# PARALLAX V1.2 challenging-motion validation

Validation date: 11 October 2026. The core ORB / essential-matrix / `recoverPose` algorithm was not changed.

## Dataset and interval selection

KITTI odometry sequence 00 was obtained from the compact proSLAM/SRRG mirror. Only left rectified grayscale camera frames 3200–3599 were extracted from the compressed archive. `P0` from the included `calib.txt` supplied the camera matrix (`fx = fy = 718.856`, `cx = 607.1928`, `cy = 185.2157`). Ground truth was used only for interval selection and post-run evaluation.

The 400-frame interval was selected quantitatively from ground truth:

- Path length: 298.689 m
- Endpoint displacement: 162.354 m
- Net heading change: 180.908°
- Heading range: 183.549°
- Cumulative absolute heading change: 377.739°

This segment contains multiple strong turns and direction reversals in heading. It is substantially more challenging than sequence 04's nearly straight validation segment.

## Baseline execution

Configuration: `configs/kitti_00_turns_3200_3599.yaml`; 2,500 ORB features; match image saved every 40 pairs.

- Frames / pairs: 400 / 399
- Successful / failed pairs: 398 / 1 (99.75% / 0.25%)
- Core speed: 9.583 FPS
- Core latency: 104.348 ms mean, 101.201 ms p50, 127.738 ms p95
- Mean stages: 22.411 ms extraction, 63.936 ms matching, 18.001 ms geometry
- Peak resident memory: 119,780 KiB (116.97 MiB)
- End-to-end wall time: 44.60 seconds
- Ratio-test matches: 366 minimum, 1,008 median, 1,066.579 mean, 1,772 maximum
- Geometric inliers: 1 minimum, 543 median, 584.529 mean, 1,389 maximum
- Mean inlier ratio: 58.29%; p5: 20.01%

The one failed estimate was global pair 3527→3528: 1,721 descriptor matches but only one cheirality-consistent geometric inlier. Ground-truth displacement was only 0.228 m while rotation was 1.060°, within a longer low-translation turn. The failure was correctly rejected and did not update the pose.

## Geometry evaluation

For each successful pair, KITTI relative motion was computed as

```text
T_current_previous_gt = inverse(T_world_camera_current_gt) T_world_camera_previous_gt
```

and its rotation was compared with the `recoverPose` rotation using the SO(3) geodesic angle. Translation direction was compared in the previous-camera frame using camera-center displacement `-R^T t`. Accumulated orientation compared estimated camera-to-start rotation against ground truth normalized to frame 3200.

- Relative rotation error: 0.295° mean, 0.180° median, 0.537° p95, 7.698° maximum
- Accumulated orientation error: 5.074° mean, 12.058° maximum, 3.247° at the final frame
- Estimated / ground-truth net heading change: 180.274° / 180.815°
- Relative translation-direction cosine: 0.97898 mean, 0.99701 median, 0.14149 minimum
- Opposite translation directions: zero

Most rotations are good, including the large turns. Several isolated essential-matrix decompositions are badly wrong despite many RANSAC/recoverPose inliers. For example:

- Pair 3471: 632/882 inliers, but 7.698° relative rotation error and direction cosine 0.144. Ground-truth rotation was only 0.172°.
- Pair 3282: 1,021/1,230 inliers, but 6.787° relative rotation error and direction cosine 0.494.
- Pair 3290: 932/1,210 inliers, but 4.891° relative rotation error and direction cosine 0.141.

These are not simple descriptor-count failures. During low translation, turning, approximately planar structure, or dominant rotation, an essential matrix can be poorly conditioned while still supporting a large consensus. A fixed minimum-inlier threshold cannot detect this degeneracy. The isolated errors create visible step changes in accumulated orientation drift.

## Trajectory and scale

Native VO remains arbitrary-scale and was never interpreted as metric output.

- Fixed axes/origin plus one global scale: scale 0.78360 ground-truth meters per VO unit; ATE RMSE 23.345 ground-truth meters.
- Umeyama 7-DoF Sim(3): scale 0.72743 ground-truth meters per VO unit; ATE RMSE 6.586 and median 5.642 ground-truth meters.

The large gap between scale-only and Sim(3) error exposes orientation/direction drift that a free rotation partly hides. Sim(3) required a 7.794° rotation and 14.453 m translation. The trajectory's overall turning topology is recognizable, but it does not maintain the original coordinate axes accurately.

Ground-truth step length ranged from 0.221 to 1.251 m (0.749 m mean). Assigning every accepted step length one therefore overweights slow, rotation-heavy sections and underweights fast sections. This is the largest systematic trajectory-shape distortion in this segment, compounded by isolated pose outliers.

## Performance experiment

A separate 2,000-feature run reached 12.700 FPS, 96.270 ms p95, and 107.06 MiB peak RAM. Sim(3) ATE improved slightly to 6.250 m, but final orientation drift worsened from 3.247° to 11.874°, estimated net heading became 192.393° instead of 180.815°, and one translation direction became opposite. The experiment is retained for evidence but rejected as the default: it is faster, not safely equivalent.

## Bug found and fixed

After a failed pair, the tracker previously set `tracking_reliable` back to true on the next locally valid pair. Without relocalization the missing transform cannot be recovered, so this silently overstated accumulated-trajectory reliability. Reliability is now sticky after loss, and later logs explicitly say that the local pose succeeded while the accumulated trajectory remains unreliable. A regression test covers this behavior.

The implementation also now saves accepted 4×4 camera-to-world poses and per-pair relative rotations/translations for evaluation. These are instrumentation changes, not changes to pose estimation.

## Readiness assessment

The transform convention and basic two-view geometry remain correct through turns. Median and p95 relative rotation errors are strong, and net heading change is accurate. Pose accumulation is useful as a V1 learning/validation baseline but is not robust enough to be called reliable after degeneracy or tracking loss. The main blockers exposed for a future version are observability/degeneracy checks, scale handling, outlier-resistant motion integration, and recovery after loss. Those features were not implemented here because they belong beyond V1 validation.

