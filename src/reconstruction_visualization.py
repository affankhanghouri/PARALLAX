"""Headless Matplotlib visualization for one two-view reconstruction."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .reconstruction import TwoViewReconstruction


def _draw_camera_axes(
    axis,
    center: np.ndarray,
    rotation_camera_to_first: np.ndarray,
    label: str,
    scale: float = 0.3,
) -> None:
    colors = ("red", "green", "blue")
    for index, color in enumerate(colors):
        direction = rotation_camera_to_first[:, index] * scale
        axis.quiver(*center, *direction, color=color, linewidth=1.5)
    axis.scatter(*center, marker="^", s=70, color="black")
    axis.text(*center, label)


def save_reconstruction(
    reconstruction: TwoViewReconstruction,
    path: str | Path,
    *,
    title: str = "Two-view sparse reconstruction",
) -> None:
    accepted = reconstruction.points_first_camera[reconstruction.accepted_mask]
    figure = plt.figure(figsize=(11, 8))
    axis = figure.add_subplot(111, projection="3d")
    if len(accepted):
        colors = accepted[:, 2]
        axis.scatter(
            accepted[:, 0], accepted[:, 1], accepted[:, 2],
            c=colors, cmap="viridis", s=5, alpha=0.75,
            label=f"accepted landmarks ({len(accepted)})",
        )
    _draw_camera_axes(axis, reconstruction.camera_center_first, np.eye(3), "camera 1")
    _draw_camera_axes(
        axis, reconstruction.camera_center_second,
        reconstruction.rotation_second_from_first.T, "camera 2",
    )
    axis.plot(
        [reconstruction.camera_center_first[0], reconstruction.camera_center_second[0]],
        [reconstruction.camera_center_first[1], reconstruction.camera_center_second[1]],
        [reconstruction.camera_center_first[2], reconstruction.camera_center_second[2]],
        color="black", linestyle="--", label="unit baseline",
    )
    stats = reconstruction.stats
    axis.text2D(
        0.02, 0.96,
        f"candidates: {stats.candidates}\naccepted: {stats.accepted}\nrejected: {stats.candidates - stats.accepted}",
        transform=axis.transAxes, va="top",
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "gray"},
    )
    axis.set_title(f"{title}\nCoordinates are arbitrary reconstruction units — not meters")
    axis.set_xlabel("camera-1 X [arbitrary units]")
    axis.set_ylabel("camera-1 Y [arbitrary units]")
    axis.set_zlabel("camera-1 Z [arbitrary units]")
    if len(accepted):
        combined = np.vstack([
            accepted,
            reconstruction.camera_center_first,
            reconstruction.camera_center_second,
        ])
        lower = np.percentile(combined, 2, axis=0)
        upper = np.percentile(combined, 98, axis=0)
        midpoint = (lower + upper) / 2.0
        radius = max(float(np.max(upper - lower)) / 2.0, 0.5)
        axis.set_xlim(midpoint[0] - radius, midpoint[0] + radius)
        axis.set_ylim(midpoint[1] - radius, midpoint[1] + radius)
        axis.set_zlim(midpoint[2] - radius, midpoint[2] + radius)
    axis.legend(loc="upper right")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)
