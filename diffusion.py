from pytest import approx
from analysis import Analysis
import numpy as np


class Diffusion:
    def __init__(self, analysis: Analysis, dt: float):
        self._ana = analysis
        self.dt = dt

    def step(self, rho):
        j = -self._ana.gradient(rho)
        return rho + self.continuity_step(j)

    def continuity_step(self, j):
        weights = self._ana.weights
        dr = self._ana.dr
        n = self._ana.n
        jk2 = j*np.arange(n)**2
        drho = (np.roll(jk2, 1) - np.roll(jk2, -1))/weights
        drho[0] = -jk2[1]/weights[0]
        drho[-1] = (jk2[-2] - jk2[-1]/(n-1)**2*n**2)/weights[-1]
        # there is a factor of /2 here that I do not entirely understand
        return 4*np.pi*dr**2*drho*self.dt/2


def test_continuity():
    dr = 2**-7
    n = 4096
    dt = 10**-7

    r = np.arange(n)*dr
    j = r/3.

    ana = Analysis(dr, n)
    diff = Diffusion(ana, dt)

    # edges are special, disregard them
    assert diff.continuity_step(j)[15:-5] == approx(-(np.zeros(n-20) + 1.0)*dt, rel=10**-3)


def test_diffusion():
    dr = 2**-7
    n = 4096
    dt = 10**-7

    r2 = (np.arange(n)*dr)**2

    sigma0 = 5.0
    gauss = np.exp(-(np.arange(n)*dr/sigma0)**2/2)/sigma0/np.sqrt(2*np.pi)

    ana = Analysis(dr, n)
    diff = Diffusion(ana, dt)

    def msd(f):
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
