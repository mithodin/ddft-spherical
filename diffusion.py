from analysis import Analysis


class Diffusion:
    def __init__(self, analysis: Analysis, dt: float):
        self._ana = analysis
        self._dt = dt

    def step(self, rho):
        j = -self._ana.gradient(rho)
        return rho - self._ana.divergence(j)*self._dt


