import os
import sys
import numpy as np
import hashlib

from typing import List, Dict, Any
from enum import Enum, unique, auto
from scipy import sparse
from analysis import Analysis
from fexc.calculate_weights import WeightCalculator


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
    _cache_dir = './.cache/'
    _version = 1

    def __init__(self: 'WeightedDensity', analysis: Analysis, wc: WeightCalculator, size_sphere: float = 1.0) -> None:
        self._ana = analysis
        self._size_sphere = size_sphere
        # measured in bins
        self._radius_sphere = int(size_sphere / analysis.dr) // 2  # hope you chose values that work
        self._coefficients: Dict[WD, Any] = dict()
        self._wc = wc
        self.__calc_coefficients()
        self._r = np.arange(self._ana.n)*self._ana.dr

    def __calc_coefficients(self: 'WeightedDensity') -> None:
        signature = "{n:.0f}-{dr:.10e}-{r_sphere:.0f}-{version:.0f}".format(n=self._ana.n, dr=self._ana.dr,
                                                                            r_sphere=self._radius_sphere,
                                                                            version=self._version)
        filename = self.__get_filename(signature)
        try:
            print(" > trying to load weights from cache", file=sys.stderr)
            with np.load(filename) as cached:
                for wd in WD:
                    self._coefficients[wd] = sparse.csr_matrix(cached["{}".format(wd)])
            print(" > cached weights loaded successfully", file=sys.stderr)
        except Exception:
            print(" > cache not found or error loading, calculating weights", file=sys.stderr)
            self.__calc_n3_coeff()
            self.__calc_n2_coeff()
            self.__calc_n2v_coeff()
            self.__calc_n11_coeff()
            weights = {"{}".format(wd): self._coefficients[wd].toarray() for wd in WD}
            if not os.path.isdir(self._cache_dir):
                os.mkdir(self._cache_dir)
            np.savez_compressed(filename, **weights)

    def __get_filename(self: 'WeightedDensity', signature: str) -> str:
        hash = hashlib.sha1(signature.encode('utf-8')).hexdigest()
        return "{cache}/{hash}.npz".format(hash=hash, cache=self._cache_dir)

    def __calc_n3_coeff(self: 'WeightedDensity') -> None:
        wn3 = np.zeros((self._ana.n, self._ana.n))
        wn3[0, :] = self._wc.get_weights(4*np.pi*self._wc.r**2, self._ana.n, 0, self._radius_sphere, self._ana.dr)
        rp = self._wc.r
        R = self._size_sphere/2
        for i in range(1, self._radius_sphere):
            r = i*self._ana.dr
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

    def __calc_n2_coeff(self: 'WeightedDensity') -> None:
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

    def __calc_n2v_coeff(self: 'WeightedDensity') -> None:
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

    def __calc_n11_coeff(self: 'WeightedDensity') -> None:
        wn11 = np.zeros((self._ana.n, self._ana.n))
        wpsi11 = np.zeros((self._ana.n, self._ana.n))
        rp = self._wc.r
        R = self._size_sphere/2
        wpsi11[0, self._radius_sphere] = -4./3.*np.pi*R**2
        for i in range(1, self._ana.n - self._radius_sphere):
            r = i*self._ana.dr
            r0 = np.abs(i-self._radius_sphere)
            wn11[i, :] = self._wc.get_weights(
                np.pi/(4*r**3*R)*rp*(4*r**2*rp**2-(r**2+rp**2-R**2)**2) - 2*np.pi*R/(3*r)*rp,
                self._ana.n,
                r0, i + self._radius_sphere,
                self._ana.dr
            )
            wpsi11[i, :] = self._wc.get_weights(
                np.pi/r*rp**2*((4*rp**2*r**2-(rp**2+r**2-R**2)**2)/(4*rp**3*R) - 2*R/(3*rp)),
                self._ana.n,
                r0, i + self._radius_sphere,
                self._ana.dr
            )
        self._coefficients[WD.N11] = sparse.csr_matrix(wn11)
        self._coefficients[WD.PSI11] = sparse.csr_matrix(wpsi11)

    def calc_density(self: 'WeightedDensity', which: WD, rho: np.ndarray) -> np.ndarray:
        nn = self._coefficients[which].dot(rho)
        return self._ana.extrapolate(nn,
                                     (self._ana.n - 3 * self._radius_sphere, self._ana.n - self._radius_sphere),
                                     (self._ana.n - self._radius_sphere, self._ana.n))

    def calc_densities(self: 'WeightedDensity', which: List[WD], rho: np.ndarray) -> List[np.ndarray]:
        return [self.calc_density(wd, rho) for wd in which]
