import sympy as s
import numpy as np
from pytest import approx
from analysis import Analysis


class WeightCalculator:
    def __init__(self):
        self.r = s.symbols('r', real=True)
        self.r0, self.r1 = s.symbols('r0 r1', real=True)
        self.k = s.symbols('k', integer=True)
        self.dr = s.symbols('dr', real=True, positive=True)
        self.f = s.Function('f')
        self.base_function = s.Piecewise(
            (0, self.r < (self.k-1)*self.dr),
            (1+(self.r-self.k*self.dr)/self.dr, self.r < self.k*self.dr),
            (1-(self.r-self.k*self.dr)/self.dr, self.r < (self.k+1)*self.dr),
            (0, self.r >= (self.k+1)*self.dr))
        self._int_cache = dict()

    def get_weights(self, prefactor, n: int, i0: int, i1: int, dr: float):
        int = self._get_cached_integral(prefactor)
        w0 = s.lambdify([self.dr, self.k], int.replace(self.r, (self.k+1)*self.dr)-int.replace(self.r, self.k*self.dr))
        wm = s.lambdify([self.dr, self.k], int.replace(self.r, self.k*self.dr)-int.replace(self.r, (self.k-1)*self.dr))
        w = s.lambdify([self.dr, self.k], int.replace(self.r, (self.k+1)*self.dr)-int.replace(self.r, (self.k-1)*self.dr))
        return np.array(
            [0 for _ in range(0, i0)]
            + [w0(dr, i0)]
            + [w(dr, i) for i in range(i0+1, i1)]
            + [wm(dr, i1)]
            + [0 for _ in range(i1+1, n)],
            dtype=np.double
        )

    def _get_cached_integral(self, prefactor):
        int = 0
        for term in s.expand(prefactor).lseries(self.r):
            coeff, order = term.leadterm(self.r)
            try:
                nint = self._int_cache[order]
            except KeyError:
                nint = (self.r**order * self.base_function).integrate(self.r)
                self._int_cache[order] = nint
            int += coeff * nint
        return int


def test_volume_integral():
    wc = WeightCalculator()
    n = 512
    dr = 2**-7
    factor = wc.r**2*4*s.pi
    ana = Analysis(dr, n)

    weights = wc.get_weights(factor, n, 0, n-1, dr)
    assert weights == approx(ana.weights)


def test_partial_integral():
    wc = WeightCalculator()
    n = 128
    dr = 2**-7
    factor = 0

    w = wc.get_weights(factor, n, 10, 20, dr)
    assert w.size == 128
    w = wc.get_weights(factor, n, 1, 20, dr)
    assert w.size == 128
    w = wc.get_weights(factor, n, 0, 127, dr)
    assert w.size == 128
