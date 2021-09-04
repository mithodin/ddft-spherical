import numpy as np
from pytest import approx

from analysis import Analysis
from fexc.calculate_weights import WeightCalculator
from fexc.weighted_density import WeightedDensity, WD


def test_weighted_densities():
    dr = 2**-3
    n = 32
    ana = Analysis(dr, n)
    wc = WeightCalculator()

    # expect this to take a while
    wd = WeightedDensity(ana, wc)
    rho = np.ones(n)

    n3 = wd.calc_density(WD.N3, rho)
    # the edge is tricky, need to extrapolate
    assert n3 == approx(np.ones(n)*np.pi/6)

    n2 = wd.calc_density(WD.N2, rho)
    assert n2 == approx(np.ones(n)*np.pi)

    n2v = wd.calc_density(WD.N2V, rho)
    assert n2v == approx(np.zeros(n), abs=1e-10)

    n11 = wd.calc_density(WD.N11, rho)
    assert n11 == approx(np.zeros(n), abs=1e-10)

    psi2 = wd.calc_density(WD.PSI2, rho)
    assert psi2 == approx(np.ones(n)*np.pi)

    psi3 = wd.calc_density(WD.PSI3, rho)
    assert psi3 == approx(np.ones(n)*np.pi/6)

    r = np.arange(n)*dr
    r[0] = 1  # it's not, but it avoids the division warning
    R = 0.5
    psi2v_ana = np.pi*(-(R - r)**2*abs(R - r) + (R + r)**3 + 3*(R + r)*(R**2 - r**2) - 3*(R**2 - r**2)*abs(R - r))/(3*r)
    psi2v_ana[0] = np.pi
    psi2v = wd.calc_density(WD.PSI2V, rho)
    assert psi2v[:-8] == approx(psi2v_ana[:-8])


def test_extrapolate():
    dr = 2**-3
    n = 32
    ana = Analysis(dr, n)
    wc = WeightCalculator()
    wd = WeightedDensity(ana, wc)

    f = np.arange(n) * dr
    f_extrapolated = wd._extrapolate(f.copy(), (8, 17), (17, 32))
    assert f_extrapolated == approx(f)

    f = np.arange(n) * dr + 10.0
    f_extrapolated = wd._extrapolate(f.copy(), (8, 17), (17, 32))
    assert f_extrapolated == approx(f)

    f = (np.arange(n) * dr)**2
    f_extrapolated = wd._extrapolate(f.copy(), (8, 17), (17, 32))
    assert f_extrapolated == approx(f)
