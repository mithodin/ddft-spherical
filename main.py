import numpy as np
import sys
from analysis import Analysis
from ddft import DDFT
from cutoff import Cutoff
from fexc.fexc import Fexc

num_bins = 4096
dr = 2**-7
small_steps = 10**4
big_steps = 100
simulation_time = 1.0


if __name__ == "__main__":
    analysis = Analysis(dr, num_bins)
    f_exc = Fexc(analysis)
    cutoff = lambda a: Cutoff(1e-70).cutoff(a)

    rho0 = analysis.delta()
    rho = cutoff(rho0)

    ddft = DDFT(analysis, 1.0/small_steps/big_steps, f_exc, (rho, np.zeros(num_bins)))
    for t in range(big_steps+1):
        norm = analysis.integrate(rho)
        np.savetxt(sys.stdout.buffer, rho,
                   header='# t = {}\n# norm = {:.30f}'.format(t/big_steps, norm), footer='\n', comments='')
        print('big step: {}'.format(t), file=sys.stderr)
        for tt in range(small_steps):
            rho = cutoff(ddft.step()[0])

