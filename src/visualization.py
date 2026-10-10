"""Headless-safe feature and trajectory plots."""

from __future__ import annotations

from pathlib import Path

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .features import Features


def save_feature_matches(
    previous_image: np.ndarray,
    current_image: np.ndarray,
    previous_features: Features,
    current_features: Features,
    matches: list[cv2.DMatch],
    inlier_mask: np.ndarray,
    output_path: Path,
) -> None:
    mask = inlier_mask.astype(bool) if len(inlier_mask) == len(matches) else np.zeros(len(matches), dtype=bool)
    left = cv2.cvtColor(previous_image, cv2.COLOR_GRAY2BGR) if previous_image.ndim == 2 else previous_image
    right = cv2.cvtColor(current_image, cv2.COLOR_GRAY2BGR) if current_image.ndim == 2 else current_image
    height = max(left.shape[0], right.shape[0])
    canvas = np.zeros((height, left.shape[1] + right.shape[1], 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] :] = right
    for index, match in enumerate(matches):
        start = tuple(int(round(value)) for value in previous_features.keypoints[match.queryIdx].pt)
        current_point = current_features.keypoints[match.trainIdx].pt
        end = (int(round(current_point[0])) + left.shape[1], int(round(current_point[1])))
        color = (0, 220, 0) if mask[index] else (0, 120, 255)
        cv2.line(canvas, start, end, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, start, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, end, 2, color, -1, cv2.LINE_AA)
    title = f"ratio-test matches: {len(matches)} | geometric inliers: {int(mask.sum())} (green)"
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(canvas, title, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(output_path), canvas):
        raise OSError(f"Could not write visualization: {output_path}")


def save_trajectory(positions: list[np.ndarray], output_path: Path) -> None:
    points = np.asarray(positions, dtype=float)
    figure = plt.figure(figsize=(8, 7))
    axis = figure.add_subplot(111, projection="3d")
    axis.plot(points[:, 0], points[:, 1], points[:, 2], color="tab:blue", linewidth=1.5, label="estimated path")
    axis.scatter(*points[0], color="tab:green", s=60, label="initial camera")
    axis.scatter(*points[-1], color="tab:red", s=60, label="current camera")
    axis.set_xlabel("world X [arbitrary scale]")
    axis.set_ylabel("world Y [arbitrary scale]")
    axis.set_zlabel("world Z [arbitrary scale]")
    axis.set_title("Monocular visual odometry trajectory (non-metric)")
    axis.legend()
    axis.set_box_aspect((1, 1, 1))
    figure.tight_layout()
    figure.savefig(output_path, dpi=150)
    plt.close(figure)
