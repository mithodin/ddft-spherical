import numpy as np

from analysis import Analysis
from fexc.fexc import Fexc


class Quenched(Fexc):
    """
    This is an implementation of the quenched functional, F_q[rho_s, rho_d] = F_b[rho_s + rho_d] - F_b[rho_s]
    with an arbitrary one-component base functional F_b
    """

    def __init__(self, analysis: Analysis, base: Fexc):
        super(Quenched, self).__init__(analysis)
        self._base_functional = base
        self._zeros = np.zeros(analysis.n, dtype=np.float64)

    def fexc(self, rho: (np.array, np.array)) -> float:
        return self._base_functional.fexc(rho) - self._base_functional.fexc((rho[0], self._zeros))

    def d_fexc_d_rho(self, rho: (np.array, np.array)) -> (np.array, np.array):
        grad_full = self._base_functional.d_fexc_d_rho(rho)
        grad_self = self._base_functional.d_fexc_d_rho((rho[0], self._zeros))
        return grad_full[0] - grad_self[0], grad_full[1]

    def fexc_and_d_fexc(self, rho: (np.array, np.array)) -> (float, np.array, np.array):
        return self.fexc(rho), *self.d_fexc_d_rho(rho)
