from typing import cast

import sympy as s
from pytest import approx
from sympy import Expr

from analysis import Analysis
from fexc import WeightCalculator


def test_volume_integral() -> None:
    wc = WeightCalculator()
    n = 512
    dr = 2**-7
    factor = wc.r**2*4*s.pi
    ana = Analysis(dr, n)

    weights = wc.get_weights(factor, n, 0, n-1, dr)
    assert len(weights) == n
    assert weights == approx(ana.weights)


def test_partial_integral() -> None:
    wc = WeightCalculator()
    n = 128
    dr = 2**-7
    factor = cast(Expr, 0)

    w = wc.get_weights(factor, n, 10, 20, dr)
    assert w.size == 128
    w = wc.get_weights(factor, n, 1, 20, dr)
    assert w.size == 128
    w = wc.get_weights(factor, n, 0, 127, dr)
    assert w.size == 128
