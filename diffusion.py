import numpy as np

from analysis import Analysis


class Diffusion:
    def __init__(self: 'Diffusion', analysis: Analysis, dt: float) -> None:
        self._ana = analysis
        self._dt = dt

    def step(self: 'Diffusion', rho: np.ndarray) -> np.ndarray:
        j = -self._ana.gradient(rho)
        return rho - self._ana.divergence(j)*self._dt
