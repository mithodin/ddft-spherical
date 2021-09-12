from analysis import Analysis
from fexc.fexc import Fexc
from fexc.weighted_density import WeightedDensity, WD
import numpy as np
import sympy as sy


class Rosenfeld(Fexc):
    def __init__(self, analysis: Analysis, wd: WeightedDensity):
        super(Rosenfeld, self).__init__(analysis)
        self._wd = wd
        self._calc_functional_expressions()

    def _calc_functional_expressions(self):
        n2, n3, n2v = sy.symbols("n2 n3 n2v", real=True)
        n1 = n2 / (2*sy.pi)
        n1v = n2v / (2*sy.pi)
        n0 = n2 / sy.pi
        phi = sy.simplify(-n0*sy.log(1-n3)+(n1*n2-n1v*n2v)/(1-n3) + (n2-n2v**2/n2)**3/(24*sy.pi*(1-n3)**2))
        self._phi = sy.lambdify([n2, n3, n2v], phi)
        self._dphi = {
            WD.PSI2: sy.lambdify([n2, n3, n2v], sy.simplify(phi.diff(n2))),
            WD.PSI3: sy.lambdify([n2, n3, n2v], sy.simplify(phi.diff(n3))),
            WD.PSI2V: sy.lambdify([n2, n3, n2v], sy.simplify(phi.diff(n2v))),
        }

    def fexc(self, rho: (np.array, np.array), wd: (np.array, np.array, np.array) = None) -> float:
        try:
            n2, n3, n2v = wd
        except TypeError:
            rho_tot = rho[0] + rho[1]
            n2, n3, n2v = self._wd.calc_densities([WD.N2, WD.N3, WD.N2V], rho_tot)
        phi = self._phi(n2, n3, n2v)
        return self._ana.integrate(phi)

    def d_fexc_d_rho(self, rho: (np.array, np.array), wd: (np.array, np.array, np.array) = None) -> (np.array, np.array):
        try:
            n2, n3, n2v = wd
        except TypeError:
            rho_tot = rho[0] + rho[1]
            n2, n3, n2v = self._wd.calc_densities([WD.N2, WD.N3, WD.N2V], rho_tot)
        np.savetxt('./wd.dat', np.hstack((n2.reshape(-1, 1), n3.reshape(-1, 1), n2v.reshape(-1, 1))))

        dphi = {wd: self._dphi[wd](n2, n3, n2v) for wd in [WD.PSI2, WD.PSI3, WD.PSI2V]}
        np.savetxt('./dphi.dat', np.hstack((dphi[WD.PSI2].reshape(-1, 1), dphi[WD.PSI3].reshape(-1, 1), dphi[WD.PSI2V].reshape(-1, 1))))

        psi2, psi3, psi2v = (self._wd.calc_density(wd, dphi[wd]) for wd in [WD.PSI2, WD.PSI3, WD.PSI2V])
        s = psi2 + psi3 + psi2v
        return s, s

    def fexc_and_d_fexc(self, rho: (np.array, np.array)) -> (float, np.array, np.array):
        rho_tot = rho[0] + rho[1]
        wd = self._wd.calc_densities([WD.N2, WD.N3, WD.N2V], rho_tot)
        return self.fexc(rho, wd), *self.d_fexc_d_rho(rho, wd)


