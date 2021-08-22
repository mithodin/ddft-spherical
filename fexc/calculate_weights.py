import sympy as s
import numpy as np
from pytest import approx

from analysis import Analysis


class WeightCalculator:
    def __init__(self):
        self.r = s.symbols('r', real=True)
        self.k = s.symbols('k', integer=True)
        self.dr = s.symbols('dr', real=True, positive=True)
        self.f = s.Function('f')
        self.base_function = s.Piecewise(
            (0, self.r < (self.k-1)*self.dr),
            (1+(self.r-self.k*self.dr)/self.dr, self.r < self.k*self.dr),
            (1-(self.r-self.k*self.dr)/self.dr, self.r < (self.k+1)*self.dr),
            (0, self.r >= (self.k+1)*self.dr))

    def get_weights(self, prefactor, n: int, i0: int, i1: int, dr: float):
        w0 = s.integrate(
            prefactor*self.base_function,
            (self.r, self.k*self.dr, (self.k+1)*self.dr)
        ).replace(self.k, i0)
        wm = s.integrate(
            prefactor*self.base_function,
            (self.r, (self.k-1)*self.dr, self.k*self.dr)
        ).replace(self.k, i1)
        w = s.integrate(
            prefactor*self.base_function,
            (self.r, (self.k-1)*self.dr, (self.k+1)*self.dr)
        )
        return np.array(
            [0 for _ in range(0, i0)]
            + [s.N(w0.replace(self.dr, dr))]
            + [s.N(w.replace(self.k, i).replace(self.dr, dr)) for i in range(i0+1, i1)]
            + [s.N(wm.replace(self.dr, dr))]
            + [0 for _ in range(i1+1, n)]
        )


def test_volume_integral():
    wc = WeightCalculator()
    n = 128
    dr = 2**-7
    factor = wc.r**2*4*s.pi
    ana = Analysis(dr, n)

    weights = np.array([s.N(w.replace(wc.dr, dr)) for w in wc.get_weights(factor, n, 0, n-1, dr)], dtype=float)
    assert weights == approx(ana.weights)
