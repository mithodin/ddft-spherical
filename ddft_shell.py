import numpy as np
from analysis import Analysis
from cutoff import Cutoff
from fexc.fexc import Fexc
from ddft import DDFT


class DDFTShell(DDFT):
    def __init__(self, analysis: Analysis, dt: float, f_exc: Fexc, rho0: (np.array, np.array), rho_bulk, cutoff: Cutoff = None):
        super(DDFTShell, self).__init__(analysis, dt, f_exc, rho0, cutoff)

        self._dr = self._ana.dr
        self.n = self._ana.n
        self._D = 1.
        self._rho_bulk_s = 0.
        self._rho_bulk_d = rho_bulk

        self._rho_shell_bulk_s = self._rho_bulk_s * 4. * np.pi * self._dr**2 * self.n**2
        self._rho_shell_bulk_d = self._rho_bulk_d * 4. * np.pi * self._dr**2 * self.n**2
        
        self._radius_extended = np.arange(self.n + 1)
        self._radius_extended[0] = -1
        self._radius_extended *= self._dr

        self._normalized_shell_surfaces = 4. * np.pi * self._dr**2 * np.power(range(1, self.n), 2)

        self._rho_shell_s = self.to_shell_density(self._cutoff(rho0[0]).copy())
        self._rho_shell_d = self.to_shell_density(self._cutoff(rho0[1]).copy())

    def step(self, f_ext: (np.array, np.array) = None) -> (np.array, np.array, np.array, np.array):
        j_exc = self.j_exc()

        #rho_s, j_s = self.step_s_d()

        j_s = j_exc[0]
        j_d = j_exc[1]
        
        rho_s = self.to_volume_density(self._rho_shell_s)
        rho_d = self.to_volume_density(self._rho_shell_d)

        if f_ext is not None:
            j_s += rho_s * f_ext[0]
            j_d += rho_d * f_ext[1]
        
        j_shell_s = self.to_shell_density(j_s)
        j_shell_d = self.to_shell_density(j_d)
        
        rho_minus_s = np.concatenate(([rho_s[0]], rho_s[:-1]))
        rho_minus_d = np.concatenate(([rho_d[0]], rho_d[:-1]))
        rho_plus_s  = np.concatenate((rho_s[1:], [self._rho_bulk_s]))
        rho_plus_d  = np.concatenate((rho_d[1:], [self._rho_bulk_d]))

        j_s += - self._D * (rho_plus_s - rho_minus_s) / (2 * self._dr)
        j_d += - self._D * (rho_plus_d - rho_minus_d) / (2 * self._dr)
        
        rho_shell_extended_s = np.concatenate(([0],self._rho_shell_s,[self._rho_shell_bulk_s]))
        rho_shell_extended_d = np.concatenate(([0],self._rho_shell_d,[self._rho_shell_bulk_d]))

        j_shell_extended_s = np.concatenate(([-j_shell_s[0]], j_shell_s, [j_shell_s[-1]]))
        j_shell_extended_d = np.concatenate(([-j_shell_d[0]], j_shell_d, [j_shell_d[-1]]))

        D_rho_shell_s = - 2 / self._dr * self._rho_shell_s
        D_rho_shell_d = - 2 / self._dr * self._rho_shell_d
        
        rho_shell_over_radius_extended_s = np.divide(rho_shell_extended_s, self._radius_extended)
        rho_shell_over_radius_extended_d = np.divide(rho_shell_extended_d, self._radius_extended)

        D_rho_shell_s += rho_shell_extended_s[2:]  / self._dr - rho_shell_over_radius_extended_s[2:]  - j_shell_extended_s[2:]  / (2 * self._D)
        D_rho_shell_d += rho_shell_extended_d[2:]  / self._dr - rho_shell_over_radius_extended_d[2:]  - j_shell_extended_d[2:]  / (2 * self._D)
        
        D_rho_shell_s += rho_shell_extended_s[:-2] / self._dr + rho_shell_over_radius_extended_s[:-2] + j_shell_extended_s[:-2] / (2 * self._D)
        D_rho_shell_d += rho_shell_extended_d[:-2] / self._dr + rho_shell_over_radius_extended_d[:-2] + j_shell_extended_d[:-2] / (2 * self._D)
        
        D_rho_shell_s *= self._D * self._dt / self._dr
        D_rho_shell_d *= self._D * self._dt / self._dr
        
        self._rho_shell_s += D_rho_shell_s
        self._rho_shell_d += D_rho_shell_d

        rho_s = self.to_volume_density(self._rho_shell_s)
        rho_d = self.to_volume_density(self._rho_shell_d)

        return rho_s, rho_d, j_s, j_d

    def to_shell_density(self, volume_density):
        return volume_density[1:] * self._normalized_shell_surfaces

    def to_volume_density(self,shell_density):
        result = shell_density / self._normalized_shell_surfaces

        result0 = self.extrapolate_to_zero_logarithmically(result[0], result[1])
        if not np.isfinite(result0):
            result0 = result[0]

        return np.concatenate(([result0],result))

    def norms(self):
        return self._ana.integrate_shell(self._rho_shell_s), self._ana.integrate_shell(self._rho_shell_d)
    
    def extrapolate_to_zero_logarithmically(self, x1, x2):
        log_x1 = np.log(x1)
        log_x2 = np.log(x2)

        log_x0 = (4 * log_x1 - log_x2) / 3

        return np.exp(log_x0)
