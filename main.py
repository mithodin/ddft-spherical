import numpy as np
import sys
from analysis import Analysis
from ddft_shell import DDFTShell
from cutoff import Cutoff
from fexc.calculate_weights import WeightCalculator
from fexc.rosenfeld import Rosenfeld
from fexc.fexc import Fexc
from fexc.weighted_density import WeightedDensity
from initial import load_initial
from tqdm import tqdm

timesteps = {
    'initial': {
        'small_steps': 10**5,
        'big_steps': 10**2,
        'simulation_time': 10**-2
    },
    'main': {
        'small_steps': 10**4,
        'big_steps': 10**2,
        'simulation_time': 1.0 - 10**-2
    }
}

log = lambda *args: print(*args, file=sys.stderr)

if __name__ == "__main__":
    log("*** initializing ***")
    dr, num_bins, bulk_density, rho_self, rho_dist = load_initial("vanhove.h5")

    analysis = Analysis(dr, num_bins)
    wc = WeightCalculator()
    wd = WeightedDensity(analysis, wc)
    f_exc = Rosenfeld(analysis, wd)
    #f_exc = Fexc(analysis)
    cutoff = lambda a: Cutoff(1e-70).cutoff(a)

    rho_self = cutoff(rho_self)
    rho_dist = cutoff(rho_dist)

    log(" > done.")
    log("*** starting integration ***")

    j_s = np.zeros(num_bins)
    j_d = np.zeros(num_bins)
    D_rho_shell_self = np.zeros(num_bins)
    D_rho_shell_dist = np.zeros(num_bins)
    
    t0 = 0
    for phase in timesteps.keys():
        log(" > {} phase".format(phase))
        small_steps = timesteps[phase]['small_steps']
        big_steps = timesteps[phase]['big_steps']
        simulation_time = timesteps[phase]['simulation_time']
        ddft = DDFTShell(analysis, 1.0/small_steps/big_steps, f_exc, (rho_self, rho_dist), bulk_density, Cutoff(1e-70))
        for t in tqdm(range(int(simulation_time*big_steps)), position=0, desc='big steps', file=sys.stderr):
            # norm_self = analysis.integrate(rho_self)
            # norm_dist = analysis.integrate(rho_dist)
            norm_self, norm_dist = ddft.norms()
            np.savetxt(sys.stdout.buffer, np.hstack((rho_self.reshape(-1, 1), rho_dist.reshape(-1, 1), j_s.reshape(-1, 1), j_d.reshape(-1, 1), D_rho_shell_self.reshape(-1, 1), D_rho_shell_dist.reshape(-1, 1))),
                       header='# t = {}\n# norm = {:.30f}\t{:.30f}'.format(t / big_steps + t0, norm_self, norm_dist), footer='\n', comments='')
            for tt in tqdm(range(small_steps), position=1, desc='small steps', file=sys.stderr):
                rho_self, rho_dist, j_s, j_d, D_rho_shell_self, D_rho_shell_dist = ddft.step()
            if not (np.all(np.isfinite(rho_self)) and np.all(np.isfinite(rho_dist))):
                log("ERROR: invalid number detected in rho")
                sys.exit(1)
        t0 += simulation_time
        log(" > {} phase done".format(phase))
    # norm_self = analysis.integrate(rho_self)
    # norm_dist = analysis.integrate(rho_dist)
    norm_self, norm_dist = ddft.norms()
    np.savetxt(sys.stdout.buffer, np.hstack((rho_self.reshape(-1, 1), rho_dist.reshape(-1, 1), j_s.reshape(-1, 1), j_d.reshape(-1, 1), D_rho_shell_self.reshape(-1, 1), D_rho_shell_dist.reshape(-1, 1))),
               header='# t = {}\n# norm = {:.30f}\t{:.30f}'.format(t0, norm_self, norm_dist), footer='\n', comments='')
    log("*** done, have a nice day ***")
