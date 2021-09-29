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
        
        self._radius = np.multiply(range(self.n),self._dr)
        self._radius_minus = self._radius - self._dr
        self._radius_plus  = self._radius + self._dr

        self._radius_minus[1] = -self._dr

        self._normalized_shell_surfaces = np.power(range(self.n), 2)
        self._normalized_shell_surfaces[0] = 1
        self._normalized_shell_surfaces = self._normalized_shell_surfaces * 4. * np.pi * self._dr**2

        self._rho_shell_s = self.to_shell_density(self._cutoff(rho0[0]).copy())
        self._rho_shell_d = self.to_shell_density(self._cutoff(rho0[1]).copy())

    def step(self, f_ext: (np.array, np.array) = None) -> (np.array, np.array, np.array, np.array):
        j_exc = self.j_exc()
        # j_s = - self._ana.gradient(self._rho_s) + j_exc[0]
        # j_d = - self._ana.gradient(self._rho_d) + j_exc[1]
        j_s = j_exc[0]
        j_d = j_exc[1]
        if f_ext is not None:
            j_s += self._rho_s * f_ext[0]
            j_d += self._rho_d * f_ext[1]
        
        j_shell_s = self.to_shell_density(j_s)
        j_shell_d = self.to_shell_density(j_d)

        rho_shell_minus_s = np.concatenate(([0], self._rho_shell_s[:-1]))
        rho_shell_minus_d = np.concatenate(([0], self._rho_shell_d[:-1]))
        rho_shell_plus_s  = np.concatenate((self._rho_shell_s[1:], [self._rho_shell_bulk_s]))
        rho_shell_plus_d  = np.concatenate((self._rho_shell_d[1:], [self._rho_shell_bulk_d]))

        j_shell_minus_s = np.concatenate(([-j_shell_s[0]], j_shell_s[:-1]))
        j_shell_minus_d = np.concatenate(([-j_shell_d[0]], j_shell_d[:-1]))
        j_shell_plus_s  = np.concatenate((j_shell_s[1:], [0]))
        j_shell_plus_d  = np.concatenate((j_shell_d[1:], [0]))

        D_rho_shell_s = - 2 / self._dr * self._rho_shell_s
        D_rho_shell_d = - 2 / self._dr * self._rho_shell_d
        
        D_rho_shell_s += rho_shell_plus_s  / self._dr - np.divide(rho_shell_plus_s ,self._radius_plus ) - j_shell_plus_s  / (2 * self._D)
        D_rho_shell_d += rho_shell_plus_d  / self._dr - np.divide(rho_shell_plus_d ,self._radius_plus ) - j_shell_plus_d  / (2 * self._D)
        D_rho_shell_s += rho_shell_minus_s / self._dr + np.divide(rho_shell_minus_s,self._radius_minus) + j_shell_minus_s / (2 * self._D)
        D_rho_shell_d += rho_shell_minus_d / self._dr + np.divide(rho_shell_minus_d,self._radius_minus) + j_shell_minus_d / (2 * self._D)
        
        D_rho_shell_s *= self._D * self._dt / self._dr
        D_rho_shell_d *= self._D * self._dt / self._dr

        self._rho_shell_s += D_rho_shell_s
        self._rho_shell_d += D_rho_shell_d

        return self.to_volume_density(self._rho_shell_s), self.to_volume_density(self._rho_shell_d), j_s, j_d

    def to_shell_density(self, volume_density):
        return volume_density * self._normalized_shell_surfaces

    def to_volume_density(self,shell_density):
        result = shell_density / self._normalized_shell_surfaces
        result[0] = self.extrapolate_to_zero_logarithmically(result[1], result[2])

        return result

    def norms(self):
        return self._ana.integrate_shell(self._rho_shell_s), self._ana.integrate_shell(self._rho_shell_d)
    
    def extrapolate_to_zero_logarithmically(self, x1, x2):
        log_x1 = np.log(x1)
        log_x2 = np.log(x2)

        log_x0 = (4 * log_x1 - log_x2) / 3

        return np.exp(log_x0)
