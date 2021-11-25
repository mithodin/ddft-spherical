import numpy as np

from typing import Tuple
from analysis import Analysis


class Fexc:
    """
    Base class for all excess functionals
    """
    def __init__(self: 'Fexc', analysis: Analysis) -> None:
        self._ana = analysis

    def fexc(self: 'Fexc', rho: Tuple[np.ndarray, np.ndarray]) -> float:
        return 0.0

    def d_fexc_d_rho(self: 'Fexc', rho: Tuple[np.ndarray, np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        return np.zeros(self._ana.n), np.zeros(self._ana.n)

    def fexc_and_d_fexc(self: 'Fexc', rho: Tuple[np.ndarray, np.ndarray]) -> Tuple[float, np.ndarray, np.ndarray]:
        return self.fexc(rho), *self.d_fexc_d_rho(rho)
