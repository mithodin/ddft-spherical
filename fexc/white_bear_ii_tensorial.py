import sympy as sy
import numpy as np

from analysis import Analysis
from fexc.fexc import Fexc
from fexc.weighted_density import WeightedDensity, WD


class WhiteBearIITensorial(Fexc):
    """
    Implementation of the White Bear Mk. II functional with tensorial modification,
    see e.g. Roth, R. „Fundamental measure theory for hard-sphere mixtures: a review“
    J. Phys. Cond. Mat. 22, 0631002 (2010) for an introduction and review
    """

    def __init__(self, analysis: Analysis, wd: WeightedDensity):
        super(WhiteBearIITensorial, self).__init__(analysis)
        self._wd = wd
        self._calc_functional_expressions()

    def _calc_functional_expressions(self):
        n2, n3, n2v, n11 = sy.symbols("n2 n3 n2v n11", real=True)
        n1 = n2 / (2*sy.pi)
        n1v = n2v / (2*sy.pi)
        n0 = n2 / sy.pi
        phi2 = (2*n3-n3**2+2*(1-n3)*sy.log(1-n3))/n3
        phi3 = (2*n3-3*n3**2+2*n3**3+2*(1-n3)**2*sy.log(1-n3))/n3**2
        phi = -n0*sy.log(1-n3)+(n1*n2-n1v*n2v)*(1+phi2/3)/(1-n3) \
            + (n2**3-3*n2*n2v**2+9*(3*n11**3-n11*n2v**2))*(1-phi3/3)/(24*sy.pi*(1-n3)**2)
        self._phi = sy.lambdify([n2, n3, n2v, n11], phi)
        self._dphi = {
            WD.PSI2: sy.lambdify([n2, n3, n2v, n11], sy.simplify(phi.diff(n2))),
            WD.PSI3: sy.lambdify([n2, n3, n2v, n11], sy.simplify(phi.diff(n3))),
            WD.PSI2V: sy.lambdify([n2, n3, n2v, n11], sy.simplify(phi.diff(n2v))),
            WD.PSI11: sy.lambdify([n2, n3, n2v, n11], sy.simplify(phi.diff(n11)))
        }

    def fexc(self, rho: (np.array, np.array), wd: (np.array, np.array, np.array, np.array) = None) -> float:
        try:
            n2, n3, n2v, n11 = wd
        except TypeError:
            rho_tot = rho[0] + rho[1]
            n2, n3, n2v, n11 = self._wd.calc_densities([WD.N2, WD.N3, WD.N2V, WD.N11], rho_tot)
        phi = self._phi(n2, n3, n2v, n11)
        return self._ana.integrate(phi)

    def d_fexc_d_rho(self, rho: (np.array, np.array), wd: (np.array, np.array, np.array, np.array) = None)\
            -> (np.array, np.array):
        try:
            n2, n3, n2v, n11 = wd
        except TypeError:
            rho_tot = rho[0] + rho[1]
            n2, n3, n2v, n11 = self._wd.calc_densities([WD.N2, WD.N3, WD.N2V, WD.N11], rho_tot)

        dphi = {wd: self._dphi[wd](n2, n3, n2v, n11) for wd in [WD.PSI2, WD.PSI3, WD.PSI2V, WD.PSI11]}

        psi2, psi3, psi2v, psi11 = \
            (self._wd.calc_density(wd, dphi[wd]) for wd in [WD.PSI2, WD.PSI3, WD.PSI2V, WD.PSI11])
        s = psi2 + psi3 + psi2v + psi11
        return s, s

    def fexc_and_d_fexc(self, rho: (np.array, np.array)) -> (float, np.array, np.array):
        rho_tot = rho[0] + rho[1]
        wd = self._wd.calc_densities([WD.N2, WD.N3, WD.N2V, WD.N11], rho_tot)
        return self.fexc(rho, wd), *self.d_fexc_d_rho(rho, wd)
