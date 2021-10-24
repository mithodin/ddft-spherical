import sys

from analysis import Analysis
from fexc.fexc import Fexc


def load_functional(base: str, variant: str, analysis: Analysis) -> Fexc:
    f_base = Fexc(analysis)
    if base == "rosenfeld_q3":
        from fexc.calculate_weights import WeightCalculator
        from fexc.weighted_density import WeightedDensity
        from fexc.rosenfeld_q3 import RosenfeldQ3
        wc = WeightCalculator()
        wd = WeightedDensity(analysis, wc)
        f_base = RosenfeldQ3(analysis, wd)
    elif base == "white_bear_ii_tensorial":
        from fexc.calculate_weights import WeightCalculator
        from fexc.weighted_density import WeightedDensity
        from fexc.white_bear_ii_tensorial import WhiteBearIITensorial
        wc = WeightCalculator()
        wd = WeightedDensity(analysis, wc)
        f_base = WhiteBearIITensorial(analysis, wd)
    elif base is not None:
        print("Warning: specified base functional '{}' not found, using zero excess functional instead".format(base), sys.stderr)

    f_exc = f_base
    if variant == "partially_linearised":
        from fexc.partially_linearised import PartiallyLinearised
        f_exc = PartiallyLinearised(analysis, f_base)
    elif variant == "quenched":
        from fexc.quenched import Quenched
        f_exc = Quenched(analysis, f_base)
    elif variant is not None:
        print("Warning: specified variant '{}' not found, using 'full' variant instead".format(variant), sys.stderr)
    return f_exc
