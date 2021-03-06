import numpy as np
from _pytest.python_api import approx

from analysis import Analysis
from diffusion import Diffusion


def test_diffusion() -> None:
    dr = 2**-7
    n = 4096
    dt = 10**-7

    r2 = (np.arange(n)*dr)**2

    sigma0 = 5.0
    gauss = np.exp(-(np.arange(n)*dr/sigma0)**2/2)/sigma0/np.sqrt(2*np.pi)

    ana = Analysis(dr, n)
    diff = Diffusion(ana, dt)

    def msd(f: np.ndarray) -> float:
        return ana.integrate(f*r2)/ana.integrate(f)

    msd0 = msd(gauss)
    assert msd0 == approx(3*sigma0**2)

    norm0 = ana.integrate(gauss)

    for t in range(10):
        for _ in range(10**3):
            gauss = diff.step(gauss)
        norm = ana.integrate(gauss)
        assert msd(gauss) - msd0 == approx(6*(t+1)*10**3*dt, rel=10**-5)
        assert norm - norm0 == approx(0, abs=10**-8)
