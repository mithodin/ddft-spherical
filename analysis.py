import numpy as np
from pytest import approx


class Analysis:
    def __init__(self, dr: float, n: int):
        self.dr = dr
        self.n = n
        self.__init_weights()

    def __init_weights(self):
        self.weights = (6 * np.arange(self.n) ** 2 + 1) * 2.0
        self.weights[0] = 1
        self.weights[-1] = (6 * (self.n - 1) ** 2 - 4 * (self.n - 1) + 1)
        self.weights *= self.dr ** 3 * np.pi / 3

    def integrate(self, f: np.array):
        return np.sum(f * self.weights)

    def gradient(self, f: np.array):
        res = (np.roll(f, -1) - np.roll(f, 1)) / 2.
        res[0] = (-3 * f[0] + 4 * f[1] - f[2]) / 2.
        res[-1] = (f[-3] - 4 * f[-2] + 3 * f[-1]) / 2.
        return res/self.dr

    def delta(self):
        res = np.zeros(self.n)
        res[0] = 1.0/self.weights[0]
        return res

    def divergence(self, f):
        weights = self.weights
        dr = self.dr
        n = self.n
        fk2 = f*np.arange(n)**2
        div = (np.roll(fk2, 1) - np.roll(fk2, -1))/weights
        div[0] = -fk2[1]/weights[0]
        div[-1] = (fk2[-2] - fk2[-1]/(n-1)**2*n**2)/weights[-1]
        # there is a factor of /2 here that I do not entirely understand
        return 2*np.pi*dr**2*div


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
    f = np.zeros(n) + 1.0  # 1
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


def test_continuity():
    dr = 2**-7
    n = 4096

    r = np.arange(n)*dr
    j = r/3.

    ana = Analysis(dr, n)

    # edges are special, disregard them
    assert ana.divergence(j)[15:-5] == approx(-(np.zeros(n-20) + 1.0), rel=10**-3)
