from analysis import Analysis
import numpy as np

class Diffusion:
    def __init__(self, analysis: Analysis, dt: float):
        self._ana = analysis
        self.dt = dt

    def step(self, rho):
        j = -self._ana.gradient(rho)
        weights = self._ana.weights
        dr = self._ana.dr
        n = self._ana.n
        jk2 = j*np.arange(n)**2
        drho = (np.roll(jk2, 1) - np.roll(jk2, -1))/weights
        drho[0] = -jk2[1]/weights[0]
        drho[-1] = (jk2[-2] - jk2[-1]/(n-1)**2*n**2)/weights[-1]
        return rho + 4*np.pi*dr**2*drho*self.dt
