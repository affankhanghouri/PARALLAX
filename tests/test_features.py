import cv2
import numpy as np

from src.features import Features, ORBFeatures


def test_matching_handles_missing_descriptors() -> None:
    frontend = ORBFeatures()
    empty = Features((), None)
    assert frontend.match(empty, empty) == []


def test_matching_handles_too_few_descriptors() -> None:
    frontend = ORBFeatures()
    keypoint = (cv2.KeyPoint(1.0, 1.0, 3.0),)
    one = Features(keypoint, np.zeros((1, 32), dtype=np.uint8))
    assert frontend.match(one, one) == []

