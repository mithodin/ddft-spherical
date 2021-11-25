import numpy as np
import sympy as sy

from typing import Tuple, cast
from analysis import Analysis
from fexc.fexc import Fexc
from fexc.weighted_density import WeightedDensity, WD


class RosenfeldQ3(Fexc):
    """
    Implementation of the classic Rosenfeld functional with q3 correction,
    see e.g. Roth, R. „Fundamental measure theory for hard-sphere mixtures: a review“
    J. Phys. Cond. Mat. 22, 0631002 (2010) for an introduction and review
    """

    def __init__(self: 'RosenfeldQ3', analysis: Analysis, wd: WeightedDensity) -> None:
        super(RosenfeldQ3, self).__init__(analysis)
        self._wd = wd
        self.__calc_functional_expressions()

    def __calc_functional_expressions(self: 'RosenfeldQ3') -> None:
        n2, n3, n2v = sy.symbols("n2 n3 n2v", real=True)
        n1 = n2 / (2*sy.pi)
        n1v = n2v / (2*sy.pi)
        n0 = n2 / sy.pi
        phi = -n0*sy.log(1-n3)+(n1*n2-n1v*n2v)/(1-n3) + (n2-n2v**2/n2)**3/(24*sy.pi*(1-n3)**2)
        self._phi = sy.lambdify([n2, n3, n2v], phi)
        self._dphi = {
            WD.PSI2: sy.lambdify([n2, n3, n2v], sy.simplify(phi.diff(n2))),
            WD.PSI3: sy.lambdify([n2, n3, n2v], sy.simplify(phi.diff(n3))),
            WD.PSI2V: sy.lambdify([n2, n3, n2v], sy.simplify(phi.diff(n2v))),
        }

    def fexc(self: 'RosenfeldQ3', rho: Tuple[np.ndarray, np.ndarray],
             wd: Tuple[np.ndarray, np.ndarray, np.ndarray] = None) -> float:
        n2, n3, n2v = self.__get_weighted_densities(rho, wd)
        phi = self._phi(n2, n3, n2v)
        return self._ana.integrate(phi)

    def __get_weighted_densities(self: 'RosenfeldQ3', rho: Tuple[np.ndarray, np.ndarray],
                                 wd: Tuple[np.ndarray, np.ndarray, np.ndarray] = None)\
            -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        try:
            n2, n3, n2v = cast(Tuple[np.ndarray, np.ndarray, np.ndarray], wd)
        except TypeError:
            rho_tot = rho[0] + rho[1]
            n2, n3, n2v = self._wd.calc_densities([WD.N2, WD.N3, WD.N2V], rho_tot)
        return n2, n3, n2v

    def d_fexc_d_rho(self: 'RosenfeldQ3', rho: Tuple[np.ndarray, np.ndarray],
                     wd: Tuple[np.ndarray, np.ndarray, np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        n2, n3, n2v = self.__get_weighted_densities(rho, wd)
        dphi = {wd: self._dphi[wd](n2, n3, n2v) for wd in [WD.PSI2, WD.PSI3, WD.PSI2V]}

        psi2, psi3, psi2v = (self._wd.calc_density(wd, dphi[wd]) for wd in [WD.PSI2, WD.PSI3, WD.PSI2V])
        s = psi2 + psi3 + psi2v
        return s, s

    def fexc_and_d_fexc(self: 'RosenfeldQ3', rho: Tuple[np.ndarray, np.ndarray]) \
            -> Tuple[float, np.ndarray, np.ndarray]:
        rho_tot = rho[0] + rho[1]
        wd = cast(Tuple[np.ndarray, np.ndarray, np.ndarray],
                  tuple(self._wd.calc_densities([WD.N2, WD.N3, WD.N2V], rho_tot)))
        grad_self, grad_dist = self.d_fexc_d_rho(rho, wd)
        return self.fexc(rho, wd), grad_self, grad_dist
