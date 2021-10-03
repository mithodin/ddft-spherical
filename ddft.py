import numpy as np
from analysis import Analysis
from cutoff import Cutoff
from fexc.fexc import Fexc


class DDFT:
    def __init__(self, analysis: Analysis, dt: float, f_exc: Fexc, rho0: (np.array, np.array), cutoff: Cutoff = None):
        self._ana = analysis
        self._dt = dt
        self._f_exc = f_exc
        self._cutoff = (lambda arr: cutoff.cutoff(arr)) if cutoff is not None else (lambda arr: arr)
        self._rho_s = self._cutoff(rho0[0]).copy()
        self._rho_d = self._cutoff(rho0[1]).copy()
        

    def step(self, f_ext: (np.array, np.array) = None) -> (np.array, np.array, np.array, np.array):
        j_exc = self.j_exc()
        np.savetxt('tmp_jexc',np.transpose([j_exc[0],j_exc[1]]))
        j_s = - self._ana.gradient(self._rho_s) + j_exc[0]
        j_d = - self._ana.gradient(self._rho_d) + j_exc[1]
        if f_ext is not None:
            j_s += self._rho_s * f_ext[0]
            j_d += self._rho_d * f_ext[1]
        d_rho_s = - self._ana.divergence(j_s) * self._dt
        d_rho_d = - self._ana.divergence(j_d) * self._dt
        self.boundary_condition(d_rho_s, d_rho_d)
        self._rho_s[:] = self._cutoff(self._rho_s + d_rho_s)
        self._rho_d[:] = self._cutoff(self._rho_d + d_rho_d)
        return self._rho_s, self._rho_d, j_s, j_d

    def j_exc(self):
        d_fexc_d_rho = self._f_exc.d_fexc_d_rho((self._rho_s, self._rho_d))
        j_s = - self._rho_s * self._ana.gradient(d_fexc_d_rho[0])
        j_d = - self._rho_d * self._ana.gradient(d_fexc_d_rho[1])
        return j_s, j_d

    def boundary_condition(self, d_rho_s, d_rho_d):
        size_zero = int(2./self._ana.dr)
        size_fade = int(2./self._ana.dr)
        fade_out = np.zeros((size_zero + size_fade), dtype=np.float64)
        fade_out[-(size_zero + size_fade):-size_fade] = (np.flip(np.arange(size_fade, dtype=np.float64))/(size_fade-1))**2
        d_rho_s[-(size_zero + size_fade):] *= fade_out
        d_rho_d[-(size_zero + size_fade):] *= fade_out
