from pathlib import Path

import cv2
import numpy as np
import yaml

from main import run
from src.frame_source import ImageSequence


def test_pgm_kitti_mirror_images_are_supported(tmp_path: Path) -> None:
    image = np.zeros((20, 30), dtype=np.uint8)
    assert cv2.imwrite(str(tmp_path / "000000.pgm"), image)
    assert cv2.imwrite(str(tmp_path / "000001.pgm"), image)
    source = ImageSequence(tmp_path)
    assert len(source) == 2
    assert all(frame.image is not None for frame in source)


def test_short_image_sequence_smoke(tmp_path: Path) -> None:
    images = tmp_path / "images"
    images.mkdir()
    rng = np.random.default_rng(7)
    base = np.zeros((360, 480, 3), dtype=np.uint8)
    for point in rng.integers([20, 20], [460, 340], size=(250, 2)):
        cv2.circle(base, tuple(point), 2, (255, 255, 255), -1)
    for index, shift in enumerate((0, 4, 8)):
        transform = np.float32([[1, 0, shift], [0, 1, 0]])
        frame = cv2.warpAffine(base, transform, (480, 360))
        assert cv2.imwrite(str(images / f"{index:06d}.png"), frame)

    config = {
        "camera": {"matrix": [[400, 0, 240], [0, 400, 180], [0, 0, 1]]},
        "input": {"image_dir": str(images)},
        "features": {"max_features": 1000, "ratio_test": 0.8, "min_matches": 10},
        "geometry": {"min_inliers": 8, "ransac_threshold_px": 1.0},
        "output": {"directory": str(tmp_path / "output"), "save_match_every": 1},
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")

    odometry, metrics = run(config_path)

    assert len(metrics.pairs) == 2
    assert odometry.successes + odometry.failures == 2
    assert (tmp_path / "output" / "metrics.json").is_file()
    assert (tmp_path / "output" / "trajectory.png").is_file()
    assert len(list((tmp_path / "output" / "matches").glob("*.jpg"))) == 2
