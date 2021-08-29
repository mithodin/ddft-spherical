from analysis import Analysis
from fexc import Fexc
from weighted_density import WeightedDensity, WD
import numpy as np
import sympy as sy
from pytest import approx


class Rosenfeld(Fexc):
    def __init__(self, analysis: Analysis, wd: WeightedDensity):
        super(Rosenfeld, self).__init__(analysis)
        self._wd = wd
        self._R = 0.5
        self._calc_functional_expressions()

    def _calc_functional_expressions(self):
        n2, n3, n2v = sy.symbols("n2 n3 n2v", real=True)
        R = sy.symbols("R", real=True)
        n1 = n2 / (4*sy.pi*R)
        n1v = n2v / (4*sy.pi*R)
        n0 = n2 / (4*sy.pi*R**2)
        phi = -n0*sy.log(1-n3)+(n1*n2-n1v*n2v)/(1-n3) + n2**3*(1-sy.Abs(n2v/n2))**3/(24*sy.pi*(1-n3)**2)
        self._phi = sy.lambdify([n2, n3, n2v, R], phi)
        self._dphi = {
            WD.N2: sy.lambdify([n2, n3, n2v, R], phi.diff(n2)),
            WD.N3: sy.lambdify([n2, n3, n2v, R], phi.diff(n3)),
            WD.N2V: sy.lambdify([n2, n3, n2v, R], phi.diff(n2v)),
        }

    def fexc(self, rho: (np.array, np.array)) -> float:
        rho_tot = rho[0] + rho[1]
        n2, n3, n2v = self._wd.calc_densities([WD.N2, WD.N3, WD.N2V], rho_tot)
        phi = self._phi(n2, n3, n2v, self._R)
        return self._ana.integrate(phi)

    def d_fexc_d_rho(self, rho: (np.array, np.array)) -> (np.array, np.array):
        rho_tot = rho[0] + rho[1]
        return np.zeros(self._ana.n), np.zeros(self._ana.n)


def test_rf_expression():
    dr = 2**-3
    n = 32
    ana = Analysis(dr, n)
    rf = Rosenfeld(ana, None)

    n2 = np.array([0.5, 0.1, 0.2])
    n3 = np.array([0.1, 0.3, 0.8])
    n2v = np.array([0.0, 0.3, 0.5])
    n0 = n2 / np.pi
    n1 = n2 / (2*np.pi)
    n1v = n2v / (2*np.pi)

    assert rf._phi(n2, n3, n2v, 0.5) == approx(-n0*np.log(1-n3)+(n1*n2-n1v*n2v)/(1-n3) + n2**3*(1-np.abs(n2v/n2))**3/(24*np.pi*(1-n3)**2))
