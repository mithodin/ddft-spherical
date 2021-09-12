import numpy as np
import sys
from analysis import Analysis
from ddft import DDFT
from cutoff import Cutoff
from fexc.calculate_weights import WeightCalculator
from fexc.fexc import Fexc
from fexc.rosenfeld import Rosenfeld
from fexc.weighted_density import WeightedDensity
from initial import load_initial

small_steps = 1
big_steps = 10**6
simulation_time = 10**-4

if __name__ == "__main__":
    print("*** initializing ***", file=sys.stderr)
    dr, num_bins, rho_self, rho_dist = load_initial("vanhove.h5")

    analysis = Analysis(dr, num_bins)
    wc = WeightCalculator()
    wd = WeightedDensity(analysis, wc)
    f_exc = Rosenfeld(analysis, wd)
    cutoff = lambda a: Cutoff(1e-70).cutoff(a)

    rho_self = cutoff(rho_self)
    rho_dist = cutoff(rho_dist)

    print(" > done.", file=sys.stderr)
    print("*** starting integration ***", file=sys.stderr)

    j_s = np.zeros(num_bins)
    j_d = np.zeros(num_bins)
    ddft = DDFT(analysis, 1.0/small_steps/big_steps, f_exc, (rho_self, rho_dist))
    for t in range(int(simulation_time*big_steps)+1):
        norm_self = analysis.integrate(rho_self)
        norm_dist = analysis.integrate(rho_dist)
        np.savetxt(sys.stdout.buffer, np.hstack((rho_self.reshape(-1, 1), rho_dist.reshape(-1, 1), j_s.reshape(-1, 1), j_d.reshape(-1, 1))),
                   header='# t = {}\n# norm = {:.30f}\t{:.30f}'.format(t / big_steps, norm_self, norm_dist), footer='\n', comments='')
        print(' > big step: {}'.format(t), file=sys.stderr)
        for tt in range(small_steps):
            rho_self, rho_dist, j_s, j_d = ddft.step()
            rho_self = cutoff(rho_self)
            rho_dist = cutoff(rho_dist)
        if not (np.all(np.isfinite(rho_self)) and np.all(np.isfinite(rho_dist))):
            print("ERROR: invalid number detected in rho", file=sys.stderr)
            sys.exit(1)

    print("*** done, have a nice day ***", file=sys.stderr)
