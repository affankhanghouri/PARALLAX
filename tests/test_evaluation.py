import numpy as np

from scripts.evaluate_kitti import align_similarity


def test_similarity_alignment_recovers_known_transform() -> None:
    source = np.array([[0, 0, 0], [1, 0, 0], [1, 0, 2], [-1, 1, 3]], dtype=float)
    rotation = np.array([[0, 0, 1], [0, 1, 0], [-1, 0, 0]], dtype=float)
    target = (2.5 * (rotation @ source.T)).T + np.array([4.0, -2.0, 7.0])
    aligned, scale, recovered_rotation, translation = align_similarity(source, target)
    assert np.allclose(aligned, target)
    assert np.isclose(scale, 2.5)
    assert np.allclose(recovered_rotation, rotation)
    assert np.allclose(translation, [4.0, -2.0, 7.0])

