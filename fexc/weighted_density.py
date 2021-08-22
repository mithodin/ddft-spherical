import numpy as np
from enum import Enum, unique, auto
from scipy import sparse
from pytest import approx
from analysis import Analysis
from calculate_weights import WeightCalculator


@unique
class WD(Enum):
    N0 = auto()
    N1 = auto()
    N2 = auto()
    N1V = auto()
    N2V = auto()
    N3 = auto()
    N11 = auto()


class WeightedDensity:
    def __init__(self, analysis: Analysis, wc: WeightCalculator, size_sphere: float = 1.0):
        self._ana = analysis
        self._size_sphere = size_sphere
        # measured in bins
        self._radius_sphere = int(size_sphere / analysis.dr) // 2 # hope you chose values that work
        self._coefficients = dict()
        self._wc = wc
        self._calc_n3_coeff()

    def _calc_n3_coeff(self):
        wn3 = np.zeros((self._ana.n, self._ana.n))
        wn3[0, :] = self._wc.get_weights(4*np.pi*self._wc.r**2, self._ana.n, 0, self._radius_sphere, self._ana.dr)
        for i in range(1, self._radius_sphere):
            r = i*self._ana.dr
            rp = self._wc.r
            R = self._size_sphere/2
            wn3[i, :] = self._wc.get_weights(
                    4*np.pi*self._wc.r**2,
                    self._ana.n,
                    0, self._radius_sphere - i,
                    self._ana.dr)\
                + self._wc.get_weights(
                    np.pi/r*rp*(R**2-(r-rp)**2),
                    self._ana.n,
                    self._radius_sphere - i, self._radius_sphere + i,
                    self._ana.dr)
        for i in range(self._radius_sphere, self._ana.n - self._radius_sphere):
            r = i*self._ana.dr
            rp = self._wc.r
            R = self._size_sphere/2
            wn3[i, :] = self._wc.get_weights(
                np.pi/r*rp*(R**2-(r-rp)**2),
                self._ana.n,
                i - self._radius_sphere, i + self._radius_sphere,
                self._ana.dr)
        self._coefficients[WD.N3] = sparse.csr_matrix(wn3)

    def calc_density(self, which: WD, rho: (np.array, np.array)):
        return self._coefficients[which].dot(rho[0]), self._coefficients[which].dot(rho[1])


def test_n3():
    dr = 2**-4
    n = 32
    ana = Analysis(dr, n)
    wc = WeightCalculator()

    wd = WeightedDensity(ana, wc)
    print("initialized")
    rho = (np.ones(n), np.zeros(n))

    (n3, zeros) = wd.calc_density(WD.N3, rho)
    print("calculated")
    assert np.all(zeros == np.zeros(n))
    # the edge is tricky, need to extrapolate
    assert n3[:-8] == approx(np.ones(n-8)*np.pi/6)
