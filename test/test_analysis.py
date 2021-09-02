import numpy as np
from pytest import approx

from analysis import Analysis


def test_weights():
    dr = 2**-3
    n = 512
    ana = Analysis(dr, n)
    assert ana.weights[1] == approx(14/3*np.pi*dr**3)
    assert ana.weights[0] == approx(np.pi*dr**3/3)
    assert ana.weights[n-1] == approx(np.pi*dr**3/3*(6*(n-1)**2-4*(n-1)+1))


def test_integral():
    dr = 2**-3
    n = 512
    ana = Analysis(dr, n)
    f = np.zeros(n)+1.0  # 1
    assert ana.integrate(f) == approx(4./3.*np.pi*((n-1)*dr)**3)

    f = np.arange(n)*dr  # r
    assert ana.integrate(f) == approx(np.pi*((n-1)*dr)**4)


def test_gradient():
    dr = 2**-3
    n = 512
    ana = Analysis(dr, n)
    f = np.ones(n)  # 1
    assert ana.gradient(f) == approx(np.zeros(n))

    f = np.arange(n)*dr  # r
    assert ana.gradient(f) == approx(np.zeros(n) + 1.0)

    f = f**2/2.  # r**2/2
    assert ana.gradient(f) == approx(np.arange(n)*dr)


def test_delta():
    dr = 2**-7
    n = 64
    ana = Analysis(dr, n)
    delta = ana.delta()
    assert delta[1:] == approx(np.zeros(n-1))
    assert ana.integrate(delta) == approx(1.0)


def test_divergence():
    dr = 2**-7
    n = 4096

    r = np.arange(n)*dr
    j = r/3.

    ana = Analysis(dr, n)

    # edges are special, disregard them
    assert ana.divergence(j)[15:-5] == approx((np.zeros(n-20) + 1.0), rel=10**-3)
