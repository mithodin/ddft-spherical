from analysis import Analysis
from fexc import Fexc
from weighted_density import WeightedDensity, WD
import numpy as np


class Rosenfeld(Fexc):
    def __init__(self, analysis: Analysis, wd: WeightedDensity):
        super(Rosenfeld, self).__init__(analysis)
        self._wd = wd
        self._R = 0.5

    def fexc(self, rho: (np.array, np.array)) -> float:
        rho_tot = rho[0] + rho[1]
        n2, n3, n2v = self._wd.calc_densities([WD.N2, WD.N3, WD.N2V], rho_tot)
        n1v = n2v / (4*np.pi*self._R)
        n1 = n2 / (4*np.pi*self._R)
        n0 = n1 / self._R
        phi = -n0*np.log(1-n3)+(n1*n2-n1v*n2v)/(1-n3) + n2**3*(1-np.abs(n2v/n2))**3/(24*np.pi*(1-n3)**2)
        return self._ana.integrate(phi)

    def d_fexc_d_rho(self, rho: (np.array, np.array)) -> (np.array, np.array):
        rho_tot = rho[0] + rho[1]
        return np.zeros(self._ana.n), np.zeros(self._ana.n)
