"""Image-directory frame source."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".pgm", ".tif", ".tiff"}


@dataclass(frozen=True)
class Frame:
    index: int
    path: Path
    image: np.ndarray | None


class ImageSequence:
    """Naturally ordered images loaded one at a time to limit memory usage."""

    def __init__(self, directory: str | Path, max_frames: int | None = None) -> None:
        self.directory = Path(directory)
        if not self.directory.is_dir():
            raise FileNotFoundError(f"Image directory not found: {self.directory}")
        self.paths = sorted(
            (p for p in self.directory.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS),
            key=lambda p: p.name,
        )
        if max_frames is not None:
            self.paths = self.paths[:max_frames]
        if len(self.paths) < 2:
            raise ValueError(f"Need at least two images in {self.directory}; found {len(self.paths)}")

    def __len__(self) -> int:
        return len(self.paths)

    def __iter__(self):
        for index, path in enumerate(self.paths):
            yield Frame(index=index, path=path, image=cv2.imread(str(path), cv2.IMREAD_COLOR))
