import numpy as np
import sympy

from sympy import Symbol, Number
from typing import Tuple, Callable


def calculate_analytic(R: Number, r: Symbol, rp: Symbol, rho: Number) \
        -> Tuple[Callable[[float], float], Callable[[float], float], Callable[[float], float], Callable[[float], float],
                 Callable[[float], float], Callable[[float], float]]:
    n3 = sympy.lambdify([r], sympy.Piecewise(
        (4 * sympy.pi * sympy.integrate(rp ** 2 * rho, (rp, 0, R)), sympy.Eq(r, 0)),
        (4 * sympy.pi * sympy.integrate(rp ** 2 * rho, (rp, 0, R - r)) + sympy.pi / r * sympy.integrate(
            rp * rho * (R ** 2 - (r - rp) ** 2), (rp, R - r, R + r)), sympy.LessThan(r, R)),
        (sympy.pi / r * sympy.integrate(rp * rho * (R ** 2 - (r - rp) ** 2), (rp, r - R, R + r)), True)))
    n2 = sympy.lambdify([r], sympy.Piecewise(
        (4 * sympy.pi * R ** 2 * rho.replace(rp, R), sympy.Eq(r, 0)),
        (2 * sympy.pi * R / r * sympy.integrate(rp * rho, (rp, sympy.Abs(r - R), r + R)), True)
    ))
    n2v = sympy.lambdify([r], sympy.Piecewise(
        (0, sympy.Eq(r, 0)),
        (sympy.pi / r * sympy.integrate(rp * rho * (2 * (r - rp) + (R ** 2 - (r - rp) ** 2) / r),
                                        (rp, sympy.Abs(r - R), r + R)), sympy.StrictGreaterThan(r, 0))
    ))
    n11 = sympy.lambdify([r], sympy.Piecewise(
        (0, sympy.Eq(r, 0)),
        (sympy.pi / (4 * r ** 3 * R) * sympy.integrate(
            rp * rho * (4 * r ** 2 * rp ** 2 - (r ** 2 + rp ** 2 - R ** 2) ** 2),
            (rp, sympy.Abs(r - R), r + R)) - 2 * sympy.pi * R / r * sympy.integrate(rp * rho,
                                                                                    (rp, sympy.Abs(r - R), r + R)) / 3,
         sympy.StrictGreaterThan(r, 0))
    ))
    psi2v = sympy.lambdify([r], sympy.Piecewise(
        (4 * sympy.pi * R ** 2 * rho.replace(rp, R), sympy.Eq(r, 0)),
        (sympy.pi / r * sympy.integrate(rho * ((R ** 2 - (r - rp) ** 2) - 2 * rp * (r - rp)),
                                        (rp, sympy.Abs(r - R), r + R)), True)
    ))
    psi11 = sympy.lambdify([r], sympy.Piecewise(
        (-4 * sympy.pi / 3 * R ** 2 * rho.replace(rp, R), sympy.Eq(r, 0)),
        (sympy.pi / R * sympy.integrate(rho * ((4 * rp * R ** 2 - rp ** 3) / (4 * R) - 2 * R * rp / 3),
                                        (rp, sympy.Abs(r - R), r + R)), sympy.Eq(r, R)),
        (sympy.pi / r * sympy.integrate(
            rho * ((4 * rp ** 2 * r ** 2 - (rp ** 2 + r ** 2 - R ** 2) ** 2) / (4 * rp * R) - 2 * R * rp / 3),
            (rp, sympy.Abs(r - R), r + R)), True)
    ), modules=[{'Si': np.vectorize(lambda x: np.float64(sympy.N(sympy.Si(x))))}, 'scipy', 'numpy'])
    return n11, n2, n2v, n3, psi11, psi2v
