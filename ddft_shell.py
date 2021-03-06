from typing import Tuple

import numpy as np
from analysis import Analysis
from cutoff import Cutoff
from fexc import Fexc
from ddft import DDFT


def extrapolate_to_zero_logarithmically(x1: float, x2: float) -> float:
    if x1 == 0 or x2 == 0:
        return x1

    log_x1 = np.log(x1)
    log_x2 = np.log(x2)

    log_x0 = (4 * log_x1 - log_x2) / 3

    return np.exp(log_x0)


class DDFTShell(DDFT):
    def __init__(self: 'DDFTShell', analysis: Analysis, dt: float, f_exc: Fexc,
                 rho0: Tuple[np.ndarray, np.ndarray], rho_bulk: float, cutoff: Cutoff = None) -> None:
        super(DDFTShell, self).__init__(analysis, dt, f_exc, rho0, cutoff)

        self._dr = self._ana.dr
        self._n = self._ana.n
        self._D = 1.
        self._rho_bulk_s = 0.
        self._rho_bulk_d = rho_bulk
        delta_rho_bulk_s = 0.
        delta_rho_bulk_d = 0.

        self._delta_rho_shell_bulk_s = delta_rho_bulk_s * 4. * np.pi * self._dr ** 2 * self._n ** 2
        self._delta_rho_shell_bulk_d = delta_rho_bulk_d * 4. * np.pi * self._dr ** 2 * self._n ** 2

        self._radius_extended = np.arange(self._n + 1)
        self._radius_extended[0] = -1
        self._radius_extended = self._radius_extended * self._dr

        self._normalized_shell_surfaces = 4. * np.pi * self._dr ** 2 * np.arange(1, self._n) ** 2

        self._delta_rho_shell_s, self._delta_rho_shell_d = self.__to_delta_shell_density(self._rho_s, self._rho_d)

    def __calculate_shell_density_update(self: 'DDFTShell', j_shell_s: np.ndarray, j_shell_d: np.ndarray) \
            -> Tuple[np.ndarray, np.ndarray]:
        delta_rho_shell_extended_s = np.concatenate(([0], self._delta_rho_shell_s, [self._delta_rho_shell_bulk_s]),
                                                    dtype=np.float64)
        delta_rho_shell_extended_d = np.concatenate(([0], self._delta_rho_shell_d, [self._delta_rho_shell_bulk_d]),
                                                    dtype=np.float64)
        delta_rho_shell_over_radius_extended_s = np.divide(delta_rho_shell_extended_s, self._radius_extended)
        delta_rho_shell_over_radius_extended_d = np.divide(delta_rho_shell_extended_d, self._radius_extended)
        j_shell_extended_s = np.concatenate(([-j_shell_s[0]], j_shell_s, [j_shell_s[-1]]))
        j_shell_extended_d = np.concatenate(([-j_shell_d[0]], j_shell_d, [j_shell_d[-1]]))
        d_delta_rho_shell_s = (delta_rho_shell_extended_s[2:] - 2 * delta_rho_shell_extended_s[1:-1]
                               + delta_rho_shell_extended_s[:-2]) / self._dr
        d_delta_rho_shell_d = (delta_rho_shell_extended_d[2:] - 2 * delta_rho_shell_extended_d[1:-1]
                               + delta_rho_shell_extended_d[:-2]) / self._dr
        d_delta_rho_shell_s += - delta_rho_shell_over_radius_extended_s[2:] \
            + delta_rho_shell_over_radius_extended_s[:-2]
        d_delta_rho_shell_d += - delta_rho_shell_over_radius_extended_d[2:] \
            + delta_rho_shell_over_radius_extended_d[:-2]
        d_delta_rho_shell_s += - (j_shell_extended_s[2:] - j_shell_extended_s[:-2]) / (2 * self._D)
        d_delta_rho_shell_d += - (j_shell_extended_d[2:] - j_shell_extended_d[:-2]) / (2 * self._D)
        d_delta_rho_shell_s *= self._D * self._dt / self._dr
        d_delta_rho_shell_d *= self._D * self._dt / self._dr
        return d_delta_rho_shell_s, d_delta_rho_shell_d

    def __calculate_ideal_current(self: 'DDFTShell', rho_s: np.ndarray, rho_d: np.ndarray)\
            -> Tuple[np.ndarray, np.ndarray]:
        rho_minus_s = np.concatenate(([rho_s[0]], rho_s[:-1]))
        rho_minus_d = np.concatenate(([rho_d[0]], rho_d[:-1]))
        rho_plus_s = np.concatenate((rho_s[1:], [self._rho_bulk_s]))
        rho_plus_d = np.concatenate((rho_d[1:], [self._rho_bulk_d]))
        return - self._D * (rho_plus_s - rho_minus_s) / (2 * self._dr), \
               - self._D * (rho_plus_d - rho_minus_d) / (2 * self._dr)

    def _to_shell_density(self: 'DDFTShell', volume_density: np.ndarray) -> np.ndarray:
        return volume_density[1:] * self._normalized_shell_surfaces

    def __to_delta_shell_density(self: 'DDFTShell', rho_s: np.ndarray, rho_d: np.ndarray)\
            -> Tuple[np.ndarray, np.ndarray]:
        return (rho_s[1:] - self._rho_bulk_s) * self._normalized_shell_surfaces,\
               (rho_d[1:] - self._rho_bulk_d) * self._normalized_shell_surfaces

    def _to_volume_density(self: 'DDFTShell', shell_density: np.ndarray, offset: float = 0.0) -> np.ndarray:
        result = shell_density / self._normalized_shell_surfaces + offset

        result0 = extrapolate_to_zero_logarithmically(result[0], result[1])
        if not np.isfinite(result0):
            result0 = result[0]

        return np.concatenate(([result0], result))

    def __shell_density_to_volume_density(self: 'DDFTShell', shell_density: Tuple[np.ndarray, np.ndarray]) \
            -> Tuple[np.ndarray, np.ndarray]:
        return self._to_volume_density(shell_density[0], self._rho_bulk_s), \
               self._to_volume_density(shell_density[1], self._rho_bulk_d)

    def step(self: 'DDFTShell', f_ext: Tuple[np.ndarray, np.ndarray] = None) \
            -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        # initialize present quantities
        rho_s, rho_d = self.__shell_density_to_volume_density((self._delta_rho_shell_s, self._delta_rho_shell_d))

        # non-ideal current
        j_s, j_d = self.j_exc()

        if f_ext is not None:
            j_s += rho_s * f_ext[0]
            j_d += rho_d * f_ext[1]

        j_shell_s = self._to_shell_density(j_s)
        j_shell_d = self._to_shell_density(j_d)

        # add ideal current just for output
        j_id = self.__calculate_ideal_current(rho_s, rho_d)
        j_s += j_id[0]
        j_d += j_id[1]

        # temporal integration of density
        d_delta_rho_shell_s, d_delta_rho_shell_d = self.__calculate_shell_density_update(j_shell_s, j_shell_d)

        self._delta_rho_shell_s += d_delta_rho_shell_s
        self._delta_rho_shell_d += d_delta_rho_shell_d

        self._rho_s, self._rho_d = self.__shell_density_to_volume_density((self._delta_rho_shell_s,
                                                                           self._delta_rho_shell_d))
        self._cutoff(self._rho_s)
        self._cutoff(self._rho_d)

        return self._rho_s, self._rho_d, j_s, j_d

    def norms(self: 'DDFTShell') -> Tuple[float, float]:
        return self._ana.integrate_shell(self._delta_rho_shell_s), self._ana.integrate_shell(self._delta_rho_shell_d)
