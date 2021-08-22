import numpy as np

from analysis import Analysis


class Fexc:
    """
    Base class for all excess functionals
    """
    def __init__(self, analysis: Analysis):
        self._ana = analysis

    def fexc(self, rho: np.array) -> float:
        return 0.0

    def d_fexc_d_rho(self, rho: (np.array, np.array)) -> (np.array, np.array):
        return np.zeros(self._ana.n), np.zeros(self._ana.n)
