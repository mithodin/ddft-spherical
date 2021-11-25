import numpy as np


class Cutoff:
    def __init__(self: 'Cutoff', cutoff: float) -> None:
        self._cutoff = cutoff

    def cutoff(self: 'Cutoff', array: np.ndarray) -> np.ndarray:
        array[array < self._cutoff] = self._cutoff
        return array
