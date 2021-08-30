import numpy as np
from analysis import Analysis
from cutoff import Cutoff
from diffusion import Diffusion
from fexc.fexc import Fexc


class DDFT:
    def __init__(self, analysis: Analysis, dt: float, f_exc: Fexc, rho0: (np.array, np.array), cutoff: Cutoff = None):
        self._ana = analysis
        self._dt = dt
        self._f_exc = f_exc
        self._cutoff = (lambda arr: cutoff.cutoff(arr)) if cutoff is not None else (lambda arr: arr)
        self._rho_s = self._cutoff(rho0[0])
        self._rho_d = self._cutoff(rho0[1])

    def step(self, f_ext: (np.array, np.array) = None) -> (np.array, np.array):
        j_exc = self.j_exc()
        j_s = - self._ana.gradient(self._rho_s) + j_exc[0]
        j_d = - self._ana.gradient(self._rho_d) + j_exc[1]
        if f_ext is not None:
            j_s += self._rho_s * f_ext[0]
            j_d += self._rho_d * f_ext[1]
        self._rho_s = self._cutoff(self._rho_s - self._ana.divergence(j_s) * self._dt)
        self._rho_d = self._cutoff(self._rho_d - self._ana.divergence(j_d) * self._dt)
        return self._rho_s, self._rho_d

    def j_exc(self):
        d_fexc_d_rho = self._f_exc.d_fexc_d_rho((self._rho_s, self._rho_d))
        j_s = - self._rho_s * self._ana.gradient(d_fexc_d_rho[0])
        j_d = - self._rho_d * self._ana.gradient(d_fexc_d_rho[1])
        return j_s, j_d


def test_j_exc():
    dr = 2**-7
    n = 4096
    dt = 10**-7
    ana = Analysis(dr, n)
    f_exc = Fexc(ana)

    sigma0 = 5.0
    gauss = np.exp(-(np.arange(n)*dr/sigma0)**2/2)/sigma0/np.sqrt(2*np.pi)
    gauss /= ana.integrate(gauss)

    ddft = DDFT(ana, dt, f_exc, (gauss, np.zeros(n)))
    j_s, j_d = ddft.j_exc()
    assert np.all(j_s == np.zeros(n))
    assert np.all(j_d == np.zeros(n))


def test_free_diffusion():
    dr = 2**-7
    n = 4096
    dt = 10**-7
    ana = Analysis(dr, n)
    f_exc = Fexc(ana)

    sigma0 = 5.0
    gauss = np.exp(-(np.arange(n)*dr/sigma0)**2/2)/sigma0/np.sqrt(2*np.pi)
    gauss /= ana.integrate(gauss)

    ddft = DDFT(ana, dt, f_exc, (gauss, np.zeros(n)))
    diffusion = Diffusion(ana, dt)

    f = gauss
    for _ in range(100):
        rho, _ = ddft.step()
        f = diffusion.step(f)
        assert np.all(rho == f)
