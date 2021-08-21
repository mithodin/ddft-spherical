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
