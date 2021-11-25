import sympy as s
import numpy as np

from typing import Callable, Tuple, Dict, cast


def get_coefficient_calculator(w_func: Callable[[float, int], float], coeff: float, dr: float)\
        -> Callable[[int], float]:
    return lambda k: coeff*w_func(dr, k)


class WeightCalculator:
    def __init__(self: 'WeightCalculator') -> None:
        self.r = s.symbols('r', real=True)
        self.r0, self.r1 = s.symbols('r0 r1', real=True)
        self.k = s.symbols('k', integer=True)
        self.dr = s.symbols('dr', real=True, positive=True)
        self.f = s.Function('f')
        self.base_function = s.Piecewise(
            (0, self.r < self.k*self.dr - self.dr),
            (1+(self.r-self.k*self.dr)/self.dr, self.r < self.k*self.dr),
            (1-(self.r-self.k*self.dr)/self.dr, self.r < self.k*self.dr + self.dr),
            (0, self.r >= self.k*self.dr + self.dr))
        self._int_cache: Dict[int, Dict[str, Callable[[float, int], float]]] = dict()

    def get_weights(self: 'WeightCalculator', prefactor: s.Expr, n: int, i0: int, i1: int, dr: float)\
            -> np.ndarray:
        w0, wm, w = self._get_cached_integral(prefactor, dr)
        return np.hstack((
            np.zeros(i0, dtype=np.double),
            w0(np.longdouble(i0)),
            np.fromiter((w(np.longdouble(i)) for i in range(i0+1, i1)), dtype=np.double),
            wm(np.longdouble(i1)),
            np.zeros(n-i1-1, dtype=np.double)
        ))

    def _get_cached_integral(self: 'WeightCalculator', prefactor: s.Expr, dr: float)\
            -> Tuple[Callable[[int], float], Callable[[int], float], Callable[[int], float]]:
        w0 = []
        wm = []
        w = []
        for term in s.expand(prefactor).lseries(self.r):
            coeff, order = term.leadterm(self.r)
            coeff = np.double(coeff)
            try:
                weights = self._int_cache[order]
            except KeyError:
                nint = (self.r**order * self.base_function).integrate(self.r)
                w0c = s.lambdify([self.dr, self.k],
                                 nint.replace(self.r, self.k*self.dr + self.dr)-nint.replace(self.r, self.k*self.dr))
                wmc = s.lambdify([self.dr, self.k],
                                 nint.replace(self.r, self.k*self.dr)-nint.replace(self.r, self.k*self.dr - self.dr))
                wc = s.lambdify([self.dr, self.k],
                                nint.replace(self.r, self.k*self.dr + self.dr)
                                - nint.replace(self.r, self.k*self.dr - self.dr))
                weights = {
                    'w0': cast(Callable[[float, int], float], w0c),
                    'wm': cast(Callable[[float, int], float], wmc),
                    'w': cast(Callable[[float, int], float], wc)
                }
                self._int_cache[order] = weights
            w0.append(get_coefficient_calculator(weights['w0'], coeff, dr))
            wm.append(get_coefficient_calculator(weights['wm'], coeff, dr))
            w.append(get_coefficient_calculator(weights['w'], coeff, dr))
        return lambda k: sum([f(k) for f in w0]), lambda k: sum([f(k) for f in wm]), lambda k: sum([f(k) for f in w])
