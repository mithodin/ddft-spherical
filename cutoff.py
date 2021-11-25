import numpy as np


class Cutoff:
    def __init__(self, cutoff: float):
        self._cutoff = cutoff

    def cutoff(self, array: np.array):
        array[array < self._cutoff] = self._cutoff
        return array
