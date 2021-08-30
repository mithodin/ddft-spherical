from analysis import Analysis
from fexc import Fexc
from calculate_weights import WeightCalculator
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
            WD.PSI2: sy.lambdify([n2, n3, n2v, R], sy.simplify(phi.diff(n2))),
            WD.PSI3: sy.lambdify([n2, n3, n2v, R], sy.simplify(phi.diff(n3))),
            WD.PSI2V: sy.lambdify([n2, n3, n2v, R], sy.simplify(phi.diff(n2v))),
        }

    def fexc(self, rho: (np.array, np.array), wd: (np.array, np.array, np.array) = None) -> float:
        try:
            n2, n3, n2v = wd
        except TypeError:
            rho_tot = rho[0] + rho[1]
            n2, n3, n2v = self._wd.calc_densities([WD.N2, WD.N3, WD.N2V], rho_tot)
        phi = self._phi(n2, n3, n2v, self._R)
        return self._ana.integrate(phi)

    def d_fexc_d_rho(self, rho: (np.array, np.array), wd: (np.array, np.array, np.array) = None) -> (np.array, np.array):
        try:
            n2, n3, n2v = wd
        except TypeError:
            rho_tot = rho[0] + rho[1]
            n2, n3, n2v = self._wd.calc_densities([WD.N2, WD.N3, WD.N2V], rho_tot)

        dphi = {wd: self._dphi[wd](n2, n3, n2v, self._R) for wd in [WD.PSI2, WD.PSI3, WD.PSI2V]}

        psi2, psi3, psi2v = (self._wd.calc_density(wd, dphi[wd]) for wd in [WD.PSI2, WD.PSI3, WD.PSI2V])
        s = psi2 + psi3 + psi2v
        return s, s

    def fexc_and_d_fexc(self, rho: (np.array, np.array)) -> (float, np.array, np.array):
        rho_tot = rho[0] + rho[1]
        wd = self._wd.calc_densities([WD.N2, WD.N3, WD.N2V], rho_tot)
        return self.fexc(rho, wd), *self.d_fexc_d_rho(rho, wd)


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

def test_fexc():
    dr = 2**-3
    n = 32
    ana = Analysis(dr, n)
    wc = WeightCalculator()
    wd = WeightedDensity(ana, wc)
    rf = Rosenfeld(ana, wd)

    rho0 = 0.1
    rho = np.ones(n)*rho0
    zero = np.zeros(n)

    fexc = rf.fexc((rho, zero))

    n2 = rho0*np.pi
    n3 = rho0*np.pi/6
    n2v = 0.0
    n0 = n2 / np.pi
    n1 = n2 / (2*np.pi)
    n1v = n2v / (2*np.pi)
    phi = -n0*np.log(1-n3)+(n1*n2-n1v*n2v)/(1-n3) + n2**3*(1-np.abs(n2v/n2))**3/(24*np.pi*(1-n3)**2)

    assert fexc == approx(4./3.*np.pi*((n-1)*dr)**3*phi)


def test_grad():
    dr = 2**-5
    n = 128
    ana = Analysis(dr, n)
    wc = WeightCalculator()
    # expect a runtime of about 3 min with n = 128
    wd = WeightedDensity(ana, wc)
    rf = Rosenfeld(ana, wd)

    sigma0 = 2.0
    gauss = np.exp(-(np.arange(n)*dr/sigma0)**2/2)/sigma0/np.sqrt(2*np.pi)
    rho: np.array = gauss
    zero = np.zeros(n)

    analytic_gradient, _ = rf.d_fexc_d_rho((rho, zero))

    numeric_gradient = np.zeros(n)
    delta = 2**-10
    for i in range(n):
        rho_plus = rho.copy()
        rho_plus[i] += delta/ana.weights[i] #technically, we move the whole spherical shell, so the numerical gradient differs from the actual functional derivative
        rho_minus = rho.copy()
        rho_minus[i] -= delta/ana.weights[i]
        numeric_gradient[i] = (rf.fexc((rho_plus, zero)) - rf.fexc((rho_minus, zero)))/(2*delta)

    np.savetxt('test.dat', np.hstack((analytic_gradient.reshape(-1, 1), numeric_gradient.reshape(-1, 1))))
    assert analytic_gradient[16:-64] == approx(numeric_gradient[16:-64],rel=10**-2) # near the origin, things are complicated, and near the edge, we extrapolate, so the weights are off in the numerical gradient

