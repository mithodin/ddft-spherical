import numpy as np

from analysis import Analysis
from fexc.fexc import Fexc


class PartiallyLinearized(Fexc):
    """
    This is an implementation of the partially linearised functional, where the force on the distinct density
    is calculated using the full functional, while the force on the self density is calculated in the limit
    rho_self -> 0
    """
    def __init__(self, analysis: Analysis, base: Fexc):
        super(PartiallyLinearized, self).__init__(analysis)
        self._base_functional = base
        self._zeros = np.zeros(analysis.n, dtype=np.float64)

    def fexc(self, rho: (np.array, np.array)) -> float:
        # In the case of this functional, Fexc[rho_s, rho_d] is ill-defined as separate functionals are used for the
        # self and distinct density components
        return float("nan")

    def d_fexc_d_rho(self, rho: (np.array, np.array)) -> (np.array, np.array):
        # lim rho_self -> 0
        grad_self = self._base_functional.d_fexc_d_rho((self._zeros, rho[1]))
        grad_dist = self._base_functional.d_fexc_d_rho(rho)
        return grad_self[0], grad_dist[1]

    def fexc_and_d_fexc(self, rho: (np.array, np.array)) -> (float, np.array, np.array):
        return self.fexc(rho), *self.d_fexc_d_rho(rho)
