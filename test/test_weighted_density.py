import numpy as np
import sympy
from pytest import approx

from analysis import Analysis
from fexc.calculate_weights import WeightCalculator
from fexc.weighted_density import WeightedDensity, WD


def test_weighted_densities_simple():
    dr = 2 ** -3
    n = 32
    ana = Analysis(dr, n)
    wc = WeightCalculator()

    # expect this to take a while
    wd = WeightedDensity(ana, wc)
    rho_bulk = 0.1
    rho = rho_bulk * np.ones(n)

    n3 = wd.calc_density(WD.N3, rho)
    # the edge is tricky, need to extrapolate
    assert n3 == approx(rho_bulk * np.ones(n) * np.pi / 6)

    n2 = wd.calc_density(WD.N2, rho)
    assert n2 == approx(rho_bulk * np.ones(n) * np.pi)

    n2v = wd.calc_density(WD.N2V, rho)
    assert n2v == approx(np.zeros(n), abs=1e-13)

    n11 = wd.calc_density(WD.N11, rho)
    assert n11 == approx(np.zeros(n), abs=1e-13)

    psi2 = wd.calc_density(WD.PSI2, rho)
    assert psi2 == approx(rho_bulk * np.ones(n) * np.pi)

    psi3 = wd.calc_density(WD.PSI3, rho)
    assert psi3 == approx(rho_bulk * np.ones(n) * np.pi / 6)

    r = np.arange(n) * dr
    r[0] = 1  # it's not, but it avoids the division warning
    R = 0.5
    psi2v_ana = np.pi * (-(R - r) ** 2 * abs(R - r) + (R + r) ** 3 + 3 * (R + r) * (R ** 2 - r ** 2) - 3 * (
                R ** 2 - r ** 2) * abs(R - r)) / (3 * r)
    psi2v_ana[0] = np.pi
    psi2v = wd.calc_density(WD.PSI2V, rho)
    assert psi2v[:-8] == approx(rho_bulk * psi2v_ana[:-8])


def test_weighted_densities():
    R = sympy.Rational(1, 2)
    r, rp = sympy.symbols('r rp', real=True)

    rho = sympy.Rational(2, 10) + sympy.sin(sympy.pi*rp)/1000
    rho_eval = sympy.lambdify([rp], rho)

    n3 = sympy.lambdify([r], sympy.Piecewise(
        (4 * sympy.pi * sympy.integrate(rp ** 2 * rho, (rp, 0, R)), sympy.Eq(r, 0)),
        (4 * sympy.pi * sympy.integrate(rp ** 2 * rho, (rp, 0, R - r)) + sympy.pi / r * sympy.integrate(
            rp * rho * (R ** 2 - (r - rp) ** 2), (rp, R - r, R + r)), sympy.LessThan(r, R)),
        (sympy.pi / r * sympy.integrate(rp * rho * (R ** 2 - (r - rp) ** 2), (rp, r - R, R + r)), True)))

    n2 = sympy.lambdify([r], sympy.Piecewise(
        (4*sympy.pi*R**2*rho.replace(rp,R), sympy.Eq(r, 0)),
        (2*sympy.pi*R/r*sympy.integrate(rp*rho, (rp,sympy.Abs(r-R),r+R)), True)
    ))

    n2v = sympy.lambdify([r], sympy.Piecewise(
        (0, sympy.Eq(r, 0)),
        (sympy.pi/r*sympy.integrate(rp*rho*(2*(r-rp)+(R**2-(r-rp)**2)/r),(rp,sympy.Abs(r-R),r+R)), sympy.StrictGreaterThan(r, 0))
    ))

    n11 = sympy.lambdify([r], sympy.Piecewise(
        (0, sympy.Eq(r, 0)),
        (sympy.pi/(4*r**3*R)*sympy.integrate(rp*rho*(4*r**2*rp**2-(r**2+rp**2-R**2)**2),(rp,sympy.Abs(r-R),r+R)) - 2*sympy.pi*R/r*sympy.integrate(rp*rho, (rp,sympy.Abs(r-R),r+R))/3, sympy.StrictGreaterThan(r, 0))
    ))

    psi2v = sympy.lambdify([r], sympy.Piecewise(
        (4*sympy.pi*R**2*rho.replace(rp,R), sympy.Eq(r, 0)),
        (sympy.pi/r*sympy.integrate(rho*((R**2-(r-rp)**2)-2*rp*(r-rp)), (rp, sympy.Abs(r-R), r+R)), True)
    ))

    psi11 = sympy.lambdify([r], sympy.Piecewise(
        (-4*sympy.pi/3*R**2*rho.replace(rp,R), sympy.Eq(r, 0)),
        (sympy.pi/R*sympy.integrate(rho*((4*rp*R**2-rp**3)/(4*R)-2*R*rp/3), (rp, sympy.Abs(r-R), r+R)), sympy.Eq(r, R)),
        (sympy.pi/r*sympy.integrate(rho*((4*rp**2*r**2-(rp**2+r**2-R**2)**2)/(4*rp*R)-2*R*rp/3), (rp, sympy.Abs(r-R), r+R)), True)
    ), modules=[{'Si': np.vectorize(lambda x: np.float64(sympy.N(sympy.Si(x))))}, 'scipy', 'numpy'])

    dr = 2 ** -6
    n = 256
    r = np.array(np.arange(n)*dr, dtype=np.longdouble)
    ana = Analysis(dr, n)
    wc = WeightCalculator()
    wd = WeightedDensity(ana, wc)

    rho_discrete = np.array([rho_eval(i * dr) for i in range(n)])

    n3_discrete, n2_discrete, n2v_discrete, n11_discrete, \
    psi3_discrete, psi2_discrete, psi2v_discrete, psi11_discrete \
        = wd.calc_densities([WD.N3, WD.N2, WD.N2V, WD.N11, WD.PSI3, WD.PSI2, WD.PSI2V, WD.PSI11], rho_discrete)

    n3_ana = n3(r)
    n2_ana = n2(r)
    n2v_ana = n2v(r)
    n11_ana = n11(r)
    psi2_ana = n2_ana
    psi3_ana = n3_ana
    psi2v_ana = psi2v(r)
    psi11_ana = psi11(r)

    mask = int(0.5/dr)
    np.savetxt("n3.dat", np.hstack((r.reshape(-1,1), n3_discrete.reshape(-1,1), n3_ana.reshape(-1,1))))
    np.savetxt("n2.dat", np.hstack((r.reshape(-1,1), n2_discrete.reshape(-1,1), n2_ana.reshape(-1,1))))
    np.savetxt("n2v.dat", np.hstack((r.reshape(-1,1), n2v_discrete.reshape(-1,1), n2v_ana.reshape(-1,1))))
    np.savetxt("n11.dat", np.hstack((r.reshape(-1,1), n11_discrete.reshape(-1,1), n11_ana.reshape(-1,1))))
    np.savetxt("psi3.dat", np.hstack((r.reshape(-1,1), psi3_discrete.reshape(-1,1), psi3_ana.reshape(-1,1))))
    np.savetxt("psi2.dat", np.hstack((r.reshape(-1,1), psi2_discrete.reshape(-1,1), psi2_ana.reshape(-1,1))))
    np.savetxt("psi2v.dat", np.hstack((r.reshape(-1,1), psi2v_discrete.reshape(-1,1), psi2v_ana.reshape(-1,1))))
    np.savetxt("psi11.dat", np.hstack((r.reshape(-1,1), psi11_discrete.reshape(-1,1), psi11_ana.reshape(-1,1))))

    assert n3_discrete[:-mask] == approx(n3_ana[:-mask], rel=1e-5)
    assert n2_discrete[:-mask] == approx(n2_ana[:-mask], rel=1e-5)
    assert n2v_discrete[mask // 2:-mask] == approx(n2v_ana[mask // 2:-mask], rel=2e-3)
    assert n11_discrete[mask // 2:-mask] == approx(n11_ana[mask // 2:-mask], rel=2e-3)
    assert psi3_discrete[:-mask] == approx(psi3_ana[:-mask], rel=1e-5)
    assert psi2_discrete[:-mask] == approx(psi2_ana[:-mask], rel=1e-5)
    assert psi2v_discrete[mask // 2:-mask] == approx(psi2v_ana[mask // 2:-mask], rel=2e-3)
    assert psi11_discrete[mask // 2:-mask] == approx(psi11_ana[mask // 2:-mask], rel=2e-3)


def test_extrapolate():
    dr = 2 ** -3
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
