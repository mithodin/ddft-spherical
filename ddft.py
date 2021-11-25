import numpy as np
from typing import Tuple
from analysis import Analysis
from cutoff import Cutoff
from fexc import Fexc


class DDFT:
    _ana: Analysis

    def __init__(self: 'DDFT', analysis: Analysis, dt: float, f_exc: Fexc, rho0: Tuple[np.ndarray, np.ndarray],
                 cutoff: Cutoff = None) -> None:
        self._ana = analysis
        self._dt = dt
        self._f_exc = f_exc
        self._cutoff = (lambda arr: cutoff.cutoff(arr)) if cutoff is not None else (lambda arr: arr)
        self._rho_s = self._cutoff(rho0[0]).copy()
        self._rho_d = self._cutoff(rho0[1]).copy()
        n = np.arange(self._ana.n, dtype=np.float64)
        self._radius_bin = self._ana.dr*((n+0.5)**3 + (n+0.5)/4.)**(1./3.)
        self._volume_bin = 4./3.*np.pi*(self._radius_bin**3 - np.roll(self._radius_bin**3, 1))
        self._volume_bin[0] = 4./3.*np.pi*self._radius_bin[0]**3

    def __boundary_condition(self: 'DDFT', d_rho_s: np.ndarray, d_rho_d: np.ndarray) -> None:
        size_zero = int(2./self._ana.dr)
        size_fade = int(2./self._ana.dr)
        fade_out = np.zeros((size_zero + size_fade), dtype=np.float64)
        fade_out[-(size_zero + size_fade):-size_fade] \
            = (np.flip(np.arange(size_fade, dtype=np.float64))/(size_fade-1))**2
        d_rho_s[-(size_zero + size_fade):] *= fade_out
        d_rho_d[-(size_zero + size_fade):] *= fade_out

    def step(self: 'DDFT', f_ext: Tuple[np.ndarray, np.ndarray] = None) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        d_fexc_self, d_fexc_dist = self._f_exc.d_fexc_d_rho((self._rho_s, self._rho_d))
        grad_d_fexc_self = self._ana.forward_gradient(d_fexc_self)
        grad_d_fexc_dist = self._ana.forward_gradient(d_fexc_dist)
        if f_ext is not None:
            grad_d_fexc_self -= f_ext[0]
            grad_d_fexc_dist -= f_ext[1]
        grad_rho_self = self._ana.forward_gradient(self._rho_s)
        grad_rho_dist = self._ana.forward_gradient(self._rho_d)
        delta_self = 4*np.pi*self._radius_bin**2*(grad_rho_self + self._rho_s*grad_d_fexc_self)
        delta_dist = 4*np.pi*self._radius_bin**2*(grad_rho_dist + self._rho_d*grad_d_fexc_dist)
        d_rho_s = (delta_self - np.roll(delta_self, 1))
        d_rho_d = (delta_dist - np.roll(delta_dist, 1))
        d_rho_s[0] = delta_self[0]
        d_rho_d[0] = delta_dist[0]
        d_rho_s *= self._dt/self._volume_bin
        d_rho_d *= self._dt/self._volume_bin
        self.__boundary_condition(d_rho_s, d_rho_d)
        self._rho_s[:] = self._cutoff(self._rho_s + d_rho_s)
        self._rho_d[:] = self._cutoff(self._rho_d + d_rho_d)
        return self._rho_s, self._rho_d, np.zeros(self._ana.n), np.zeros(self._ana.n)

    def j_exc(self: 'DDFT') -> Tuple[np.ndarray, np.ndarray]:
        d_fexc_d_rho = self._f_exc.d_fexc_d_rho((self._rho_s, self._rho_d))
        j_s = - self._rho_s * self._ana.gradient(d_fexc_d_rho[0])
        j_d = - self._rho_d * self._ana.gradient(d_fexc_d_rho[1])
        return j_s, j_d

    def norms(self: 'DDFT') -> Tuple[float, float]:
        return self._ana.integrate(self._rho_s), self._ana.integrate(self._rho_d)
