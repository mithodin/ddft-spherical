from analysis import Analysis
from util import log
from .fexc import Fexc


def load_functional(base: str, variant: str, analysis: Analysis) -> Fexc:
    f_base = Fexc(analysis)
    if base == "rosenfeld_q3":
        from .calculate_weights import WeightCalculator
        from .weighted_density import WeightedDensity
        from .rosenfeld_q3 import RosenfeldQ3
        wc = WeightCalculator()
        wd = WeightedDensity(analysis, wc)
        f_base = RosenfeldQ3(analysis, wd)
    elif base == "white_bear_ii_tensorial":
        from .calculate_weights import WeightCalculator
        from .weighted_density import WeightedDensity
        from .white_bear_ii_tensorial import WhiteBearIITensorial
        wc = WeightCalculator()
        wd = WeightedDensity(analysis, wc)
        f_base = WhiteBearIITensorial(analysis, wd)
    elif base is not None:
        log("Warning: specified base functional '{}' not found, using zero excess functional instead".format(base))

    f_exc = f_base
    if variant == "partially_linearised":
        from .partially_linearised import PartiallyLinearised
        f_exc = PartiallyLinearised(analysis, f_base)
    elif variant == "quenched":
        from .quenched import Quenched
        f_exc = Quenched(analysis, f_base)
    elif variant is not None:
        log("Warning: specified variant '{}' not found, using 'full' variant instead".format(variant))
    return f_exc
