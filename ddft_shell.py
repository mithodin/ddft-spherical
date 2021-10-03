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

        self._radius = np.arange(self.n+1)

        self._normalized_shell_surfaces = 4. * np.pi * self._radius * self._radius

        self._delta_rho_shell_s = self.to_shell_density(self._cutoff(rho0[0]).copy(), self._rho_bulk_s)
        self._delta_rho_shell_d = self.to_shell_density(self._cutoff(rho0[1]).copy(), self._rho_bulk_d)

    def step(self, f_ext: (np.array, np.array) = None) -> (np.array, np.array, np.array, np.array):

        #initialize present quantities
        rho_s = self.to_volume_density(self._delta_rho_shell_s, self._rho_bulk_s)
        rho_d = self.to_volume_density(self._delta_rho_shell_d, self._rho_bulk_d)

        j_exc = self.j_exc()
        #np.savetxt('tmp_jexc_'+str(self.n)+'_'+ "%.6f" %self._dt,np.transpose([j_exc[0], j_exc[1]]))

        #non-ideal current
        j_s = j_exc[0]
        j_d = j_exc[1]
        
        if f_ext is not None:
            j_s += rho_s * f_ext[0]
            j_d += rho_d * f_ext[1]
        

        j_shell_s = self.to_shell_density(j_s)
        j_shell_d = self.to_shell_density(j_d)

        j_shell_s[0] = -j_shell_s[1]
        j_shell_d[0] = -j_shell_d[1]
        
        #add ideal current just for output
        rho_minus_s = np.concatenate(([rho_s[0]], rho_s[:-1]))
        rho_minus_d = np.concatenate(([rho_d[0]], rho_d[:-1]))
        rho_plus_s  = np.concatenate((rho_s[1:], [self._rho_bulk_s]))
        rho_plus_d  = np.concatenate((rho_d[1:], [self._rho_bulk_d]))

        j_s += - self._D * (rho_plus_s - rho_minus_s) / (2 * self._dr)
        j_d += - self._D * (rho_plus_d - rho_minus_d) / (2 * self._dr)
        
        #temporal integration of density
        DRs = self._delta_rho_shell_s
        DRd = self._delta_rho_shell_d

        delta_rho_shell_over_radius_s = np.divide(DRs, np.where(self._radius == 0, 1, self._radius))
        delta_rho_shell_over_radius_d = np.divide(DRd, np.where(self._radius == 0, 1, self._radius))

        D_delta_rho_shell_s = (DRs[2:] - 2 * DRs[1:-1] + DRs[:-2]) / self._dr
        D_delta_rho_shell_d = (DRd[2:] - 2 * DRd[1:-1] + DRd[:-2]) / self._dr
        
        D_delta_rho_shell_s += - delta_rho_shell_over_radius_s[2:] + delta_rho_shell_over_radius_s[:-2]
        D_delta_rho_shell_d += - delta_rho_shell_over_radius_d[2:] + delta_rho_shell_over_radius_d[:-2]
        
        D_delta_rho_shell_s += - (j_shell_s[2:] - j_shell_s[:-2]) / (2 * self._D)
        D_delta_rho_shell_d += - (j_shell_d[2:] - j_shell_d[:-2]) / (2 * self._D)
        
        D_delta_rho_shell_s *= self._D * self._dt / self._dr
        D_delta_rho_shell_d *= self._D * self._dt / self._dr
        
        self._delta_rho_shell_s[1:-1] += D_delta_rho_shell_s
        self._delta_rho_shell_d[1:-1] += D_delta_rho_shell_d

        rho_s = self.to_volume_density(self._delta_rho_shell_s, self._rho_bulk_s)
        rho_d = self.to_volume_density(self._delta_rho_shell_d, self._rho_bulk_d)

        return rho_s, rho_d, j_s, j_d

    def to_shell_density(self, volume_density, bulk = 0):
        return np.concatenate(((volume_density - bulk) * self._normalized_shell_surfaces[:-1], [0]))

    def to_volume_density(self,shell_density, bulk = 0):
        result = (shell_density / self._normalized_shell_surfaces + bulk)[:-1]

        result[0] = self.extrapolate_to_zero_logarithmically(result[1], result[2])
        if not np.isfinite(result[0]):
            result[0] = result[1]

        return result

    def norms(self):
        return self._ana.integrate_shell(self._delta_rho_shell_s), self._ana.integrate_shell(self._delta_rho_shell_d)
    
    def extrapolate_to_zero_logarithmically(self, x1, x2):
        if x1 == 0 or x2 == 0:
            return x1
        
        log_x1 = np.log(x1)
        log_x2 = np.log(x2)

        log_x0 = (4 * log_x1 - log_x2) / 3

        return np.exp(log_x0)
