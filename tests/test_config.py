import numpy as np
import pytest

from src.config import validate_camera_matrix


def test_valid_camera_matrix() -> None:
    matrix = validate_camera_matrix([[500, 0, 320], [0, 510, 240], [0, 0, 1]])
    assert matrix.dtype == np.float64
    assert matrix.shape == (3, 3)


@pytest.mark.parametrize(
    "matrix",
    [
        None,
        np.eye(4),
        [[0, 0, 320], [0, 500, 240], [0, 0, 1]],
        [[500, 0, 320], [0, -1, 240], [0, 0, 1]],
        [[500, 0, 320], [0, 500, 240], [0, 1, 1]],
        [[500, 0, np.nan], [0, 500, 240], [0, 0, 1]],
    ],
)
def test_invalid_camera_matrix(matrix) -> None:
    with pytest.raises(ValueError):
        validate_camera_matrix(matrix)

