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
    PSI0 = auto()
    PSI1 = auto()
    PSI2 = auto()
    PSI1V = auto()
    PSI2V = auto()
    PSI3 = auto()
    PSI11 = auto()


class WeightedDensity:
    def __init__(self, analysis: Analysis, wc: WeightCalculator, size_sphere: float = 1.0):
        self._ana = analysis
        self._size_sphere = size_sphere
        # measured in bins
        self._radius_sphere = int(size_sphere / analysis.dr) // 2  # hope you chose values that work
        self._coefficients = dict()
        self._wc = wc
        self._calc_n3_coeff()
        self._calc_n2_coeff()
        self._calc_n2v_coeff()
        self._calc_n11_coeff()
        self._r = np.arange(self._ana.n)*self._ana.dr

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
        # using a sparse matrix representation to speed up multiplication
        self._coefficients[WD.N3] = sparse.csr_matrix(wn3)
        self._coefficients[WD.PSI3] = self._coefficients[WD.N3]

    def _calc_n2_coeff(self):
        wn2 = np.zeros((self._ana.n, self._ana.n))
        rp = self._wc.r
        R = self._size_sphere/2
        wn2[0, self._radius_sphere] = 4*np.pi*R**2
        for i in range(1, self._ana.n - self._radius_sphere):
            r = i*self._ana.dr
            r0 = np.abs(i-self._radius_sphere)
            wn2[i, :] = self._wc.get_weights(
                2*np.pi*R/r*rp,
                self._ana.n,
                r0, i + self._radius_sphere,
                self._ana.dr
            )
        self._coefficients[WD.N2] = sparse.csr_matrix(wn2)
        self._coefficients[WD.N1] = sparse.csr_matrix(wn2/(4*np.pi*R))
        self._coefficients[WD.N0] = sparse.csr_matrix(wn2/(4*np.pi*R**2))
        self._coefficients[WD.PSI2] = self._coefficients[WD.N2]
        self._coefficients[WD.PSI1] = self._coefficients[WD.N1]
        self._coefficients[WD.PSI0] = self._coefficients[WD.N0]

    def _calc_n2v_coeff(self):
        wn2v = np.zeros((self._ana.n, self._ana.n))
        wpsi2v = np.zeros((self._ana.n, self._ana.n))
        rp = self._wc.r
        R = self._size_sphere/2
        wpsi2v[0, self._radius_sphere] = 4*np.pi*R**2
        for i in range(1, self._ana.n - self._radius_sphere):
            r = i*self._ana.dr
            r0 = np.abs(i-self._radius_sphere)
            wn2v[i, :] = self._wc.get_weights(
                np.pi/r*rp*(2*(r-rp) + (R**2 - (r-rp)**2)/r),
                self._ana.n,
                r0, i + self._radius_sphere,
                self._ana.dr
            )
            wpsi2v[i, :] = self._wc.get_weights(
                np.pi/r*(R**2-(r-rp)**2 - 2*rp*(r-rp)),
                self._ana.n,
                r0, i + self._radius_sphere,
                self._ana.dr
            )
        self._coefficients[WD.N2V] = sparse.csr_matrix(wn2v)
        self._coefficients[WD.N1V] = sparse.csr_matrix(wn2v/(4*np.pi*R))
        self._coefficients[WD.PSI2V] = sparse.csr_matrix(wpsi2v)
        self._coefficients[WD.PSI1V] = sparse.csr_matrix(wpsi2v/(4*np.pi*R))

    def _calc_n11_coeff(self):
        wn11 = np.zeros((self._ana.n, self._ana.n))
        wpsi11 = np.zeros((self._ana.n, self._ana.n))
        rp = self._wc.r
        R = self._size_sphere/2
        wn11[0, :] = np.zeros(self._ana.n)
        wpsi11[0, self._radius_sphere] = -4./3.*np.pi*R**2
        for i in range(1, self._ana.n - self._radius_sphere):
            r = i*self._ana.dr
            r0 = np.abs(i-self._radius_sphere)
            wn11[i, :] = self._wc.get_weights(
                np.pi/(4*r**3*R)*rp*(4*r**2*rp**2-(r**2+rp**2-R**2)**2),
                self._ana.n,
                r0, i + self._radius_sphere,
                self._ana.dr
            )
            wpsi11[i, :] = self._wc.get_weights(
                np.pi/r*rp**2*((4*rp**2*r**2-(rp**2+r**2-R**2)**2)/(4*rp**3*R) -2*R/(3*rp)),
                self._ana.n,
                r0, i + self._radius_sphere,
                self._ana.dr
            )
        wn2 = self._coefficients[WD.N2].toarray() / 3
        wn2[0, :] = np.zeros(self._ana.n)
        self._coefficients[WD.N11] = sparse.csr_matrix(wn11 - wn2)
        self._coefficients[WD.PSI11] = sparse.csr_matrix(wpsi11)

    def calc_density(self, which: WD, rho: np.array):
        nn = self._coefficients[which].dot(rho)
        fitrange = (self._ana.n - 3*self._radius_sphere, self._ana.n - self._radius_sphere)
        extrapolate = (self._ana.n - self._radius_sphere, self._ana.n)
        coeffs = np.polyfit(self._r[fitrange[0]:fitrange[1]], nn[fitrange[0]:fitrange[1]], 1)
        nn[extrapolate[0]:extrapolate[1]] = coeffs[0] * self._r[extrapolate[0]:extrapolate[1]] + coeffs[1]
        return nn

    def calc_densities(self, which: list[WD], rho: np.array):
        res = tuple()
        for wd in which:
            res = (*res, self._coefficients[wd].dot(rho))
        return res


def test_weighted_densities():
    dr = 2**-3
    n = 32
    ana = Analysis(dr, n)
    wc = WeightCalculator()

    wd = WeightedDensity(ana, wc)
    rho = np.ones(n)

    n3 = wd.calc_density(WD.N3, rho)
    # the edge is tricky, need to extrapolate
    assert n3 == approx(np.ones(n)*np.pi/6)

    n2 = wd.calc_density(WD.N2, rho)
    assert n2 == approx(np.ones(n)*np.pi)

    n2v = wd.calc_density(WD.N2V, rho)
    assert n2v == approx(np.zeros(n))

    n11 = wd.calc_density(WD.N11, rho)
    assert n11 == approx(np.zeros(n))
