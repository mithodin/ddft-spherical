import numpy as np
import sys
from analysis import Analysis
from ddft import DDFT
from cutoff import Cutoff
from fexc.calculate_weights import WeightCalculator
from fexc.rosenfeld import Rosenfeld
from fexc.weighted_density import WeightedDensity
from initial import load_initial

small_steps = 10**4
big_steps = 100
simulation_time = 1.0

if __name__ == "__main__":
    dr, num_bins, rho_self, rho_dist = load_initial("vanhove.h5")

    analysis = Analysis(dr, num_bins)
    wc = WeightCalculator()
    wd = WeightedDensity(analysis, wc)
    f_exc = Rosenfeld(analysis, wd)
    #f_exc = Fexc(analysis)
    cutoff = lambda a: Cutoff(1e-70).cutoff(a)

    rho_self = cutoff(rho_self)
    rho_dist = cutoff(rho_dist)

    ddft = DDFT(analysis, 1.0/small_steps/big_steps, f_exc, (rho_self, rho_dist))
    for t in range(big_steps+1):
        norm_self = analysis.integrate(rho_self)
        norm_dist = analysis.integrate(rho_dist)
        np.savetxt(sys.stdout.buffer, np.hstack((rho_self.reshape(-1, 1), rho_dist.reshape(-1, 1))),
                   header='# t = {}\n# norm = {:.30f}\t{:.30f}'.format(t / big_steps, norm_self, norm_dist), footer='\n', comments='')
        print('big step: {}'.format(t), file=sys.stderr)
        for tt in range(small_steps):
            rho_self, rho_dist = ddft.step()
            rho_self = cutoff(rho_self)
            rho_dist = cutoff(rho_dist)
