import numpy as np

from analysis import Analysis
from ddft import DDFT
from diffusion import Diffusion
from fexc import Fexc


def test_free_diffusion() -> None:
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
        rho, _, _, _ = ddft.step()
        f = diffusion.step(f)
        np.testing.assert_almost_equal(rho, f)


def test_j_exc() -> None:
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
    np.testing.assert_almost_equal(j_s, np.zeros(n))
    np.testing.assert_almost_equal(j_d, np.zeros(n))
