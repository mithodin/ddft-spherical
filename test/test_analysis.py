import numpy as np

from pytest import approx
from analysis import Analysis


def test_weights() -> None:
    dr = 2**-3
    n = 512
    ana = Analysis(dr, n)
    assert ana.weights[1] == approx(14/3*np.pi*dr**3)
    assert ana.weights[0] == approx(np.pi*dr**3/3)
    assert ana.weights[n-1] == approx(np.pi*dr**3/3*(6*(n-1)**2-4*(n-1)+1))


def test_integral() -> None:
    dr = 2**-3
    n = 512
    ana = Analysis(dr, n)
    f = np.zeros(n)+1.0  # 1
    assert ana.integrate(f) == approx(4./3.*np.pi*((n-1)*dr)**3)

    f = np.arange(n)*dr  # r
    assert ana.integrate(f) == approx(np.pi*((n-1)*dr)**4)


def test_gradient() -> None:
    dr = 2**-3
    n = 512
    ana = Analysis(dr, n)
    f = np.ones(n)  # 1
    assert ana.gradient(f) == approx(np.zeros(n))

    f = np.arange(n)*dr  # r
    assert ana.gradient(f) == approx(np.zeros(n) + 1.0)

    f = f**2/2.  # r**2/2
    assert ana.gradient(f) == approx(np.arange(n)*dr)


def test_delta() -> None:
    dr = 2**-7
    n = 64
    ana = Analysis(dr, n)
    delta = ana.delta()
    assert delta[1:] == approx(np.zeros(n-1))
    assert ana.integrate(delta) == approx(1.0)


def test_divergence() -> None:
    dr = 2**-7
    n = 4096

    r = np.arange(n)*dr
    j = r/3.

    ana = Analysis(dr, n)

    # edges are special, disregard them
    assert ana.divergence(j)[:-1] == approx(np.ones(n-1))

    print(ana._div_op)


def test_extrapolate() -> None:
    dr = 2 ** -3
    n = 32
    ana = Analysis(dr, n)

    f = np.arange(n) * dr
    f_extrapolated = ana.extrapolate(f.copy(), (8, 17), (17, 32))
    assert f_extrapolated == approx(f)

    f = np.arange(n) * dr + 10.0
    f_extrapolated = ana.extrapolate(f.copy(), (8, 17), (17, 32))
    assert f_extrapolated == approx(f)
