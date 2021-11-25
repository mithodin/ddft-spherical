import numpy as np

from cutoff import Cutoff


def test_cutoff() -> None:
    cutoff = Cutoff(0.5)
    arr = -np.random.rand(128)
    assert np.all(cutoff.cutoff(arr) == 0.5)

    arr = np.array([1, 2, 3, 0, 0, 2, -1], dtype=np.double)
    assert np.all(cutoff.cutoff(arr) == np.array([1, 2, 3, 0.5, 0.5, 2, 0.5]))
