import numpy as np

class Cutoff:
    def __init__(self, cutoff: float):
        self._cutoff = cutoff

    def cutoff(self, array: np.array):
        array[array < self._cutoff] = self._cutoff
        return array


def test_cutoff():
    cutoff = Cutoff(0.5)
    arr = -np.random.rand(128)
    assert np.all(cutoff.cutoff(arr) == 0.5)

    arr = np.array([1, 2, 3, 0, 0, 2, -1], dtype=np.float)
    assert np.all(cutoff.cutoff(arr) == np.array([1, 2, 3, 0.5, 0.5, 2, 0.5]))


