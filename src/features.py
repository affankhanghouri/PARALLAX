"""ORB feature extraction and descriptor matching."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class Features:
    keypoints: tuple[cv2.KeyPoint, ...]
    descriptors: np.ndarray | None


class ORBFeatures:
    def __init__(self, max_features: int = 2500, ratio_test: float = 0.75) -> None:
        self.detector = cv2.ORB_create(nfeatures=max_features)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        self.ratio_test = ratio_test

    def extract(self, image: np.ndarray) -> Features:
        if image is None or image.size == 0:
            return Features((), None)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        keypoints, descriptors = self.detector.detectAndCompute(gray, None)
        return Features(tuple(keypoints or ()), descriptors)

    def match(self, previous: Features, current: Features) -> list[cv2.DMatch]:
        """Apply a two-nearest-neighbour Lowe ratio test using Hamming distance."""
        if previous.descriptors is None or current.descriptors is None:
            return []
        if len(previous.descriptors) < 2 or len(current.descriptors) < 2:
            return []
        pairs = self.matcher.knnMatch(previous.descriptors, current.descriptors, k=2)
        good = [first for pair in pairs if len(pair) == 2 for first, second in [pair] if first.distance < self.ratio_test * second.distance]
        return sorted(good, key=lambda match: match.distance)


def matched_points(
    previous: Features, current: Features, matches: list[cv2.DMatch]
) -> tuple[np.ndarray, np.ndarray]:
    points_previous = np.float64([previous.keypoints[m.queryIdx].pt for m in matches])
    points_current = np.float64([current.keypoints[m.trainIdx].pt for m in matches])
    return points_previous, points_current

