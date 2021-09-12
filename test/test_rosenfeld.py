import numpy as np
import sympy
from pytest import approx

from analysis import Analysis
from fexc.calculate_weights import WeightCalculator
from fexc.rosenfeld import Rosenfeld
from fexc.weighted_density import WeightedDensity
from weightend_density_analytic import calculate_analytic


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

    assert rf._phi(n2, n3, n2v) == approx(-n0*np.log(1-n3)+(n1*n2-n1v*n2v)/(1-n3) + (n2-n2v**2/n2)**3/(24*np.pi*(1-n3)**2))


def test_rosenfeld_analytic():
    R = sympy.Rational(1, 2)
    r, rp = sympy.symbols('r rp', real=True)

    rho = sympy.Rational(2, 10) + sympy.sin(sympy.pi*rp)/1000
    rho_eval = sympy.lambdify([rp], rho)

    n11, n2, n2v, n3, psi11, psi2v = calculate_analytic(R, r, rp, rho)
    n1 = lambda r: n2(r)/(4*np.pi*np.float128(R))
    n0 = lambda r: n2(r)/(4*np.pi*np.float128(R**2))
    n1v = lambda r: n2v(r)/(4*np.pi*np.float128(R))

    dr = 2 ** -6
    n = 256
    r = np.array(np.arange(n)*dr, dtype=np.longdouble)
    rho_discrete = rho_eval(r)

    ana = Analysis(dr, n)
    wc = WeightCalculator()
    wd = WeightedDensity(ana, wc)
    rf = Rosenfeld(ana, wd)

    phi_numeric = rf._phi(n2(r), n3(r), n2v(r))
    rosenfeld_numeric = rf.fexc((rho_discrete, np.zeros(n)))

    phi_analytic = -n0(r)*np.log(1-n3(r))+(n1(r)*n2(r)-n1v(r)*n2v(r))/(1-n3(r))+(n2(r)-n2v(r)**2/n2(r))**3/(24*np.pi*(1-n3(r))**2)
    rosenfeld_analytic = ana.integrate(phi_analytic)

    assert phi_numeric == approx(phi_analytic)
    assert rosenfeld_numeric == approx(rosenfeld_analytic, rel=1e-2)


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
    phi = -n0*np.log(1-n3)+(n1*n2-n1v*n2v)/(1-n3) + (n2-n2v**2/n2)**3/(24*np.pi*(1-n3)**2)

    assert fexc == approx(4./3.*np.pi*((n-1)*dr)**3*phi)


def test_grad():
    dr = 2**-5
    n = 128
    ana = Analysis(dr, n)
    wc = WeightCalculator()
    # expect a runtime of about 30 sec with n = 128
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
        rho_plus[i] += delta/ana.weights[i]  # technically, we move the whole spherical shell, so the numerical gradient differs from the actual functional derivative
        rho_minus = rho.copy()
        rho_minus[i] -= delta/ana.weights[i]
        numeric_gradient[i] = (rf.fexc((rho_plus, zero)) - rf.fexc((rho_minus, zero)))/(2*delta)

    np.savetxt('test.dat', np.hstack((analytic_gradient.reshape(-1, 1), numeric_gradient.reshape(-1, 1))))
    assert analytic_gradient[16:-64] == approx(numeric_gradient[16:-64], rel=10**-2)  # near the origin, things are complicated, and near the edge, we extrapolate, so the weights are off in the numerical gradient
