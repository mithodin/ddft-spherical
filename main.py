import numpy as np
import sys
from analysis import Analysis
from diffusion import Diffusion

num_bins = 4096
dr = 2**-7
small_steps = 10**6
big_steps = 100
simulation_time = 1.0

if __name__ == "__main__":
    analysis = Analysis(dr, num_bins)
    diffusion = Diffusion(analysis, 1.0/small_steps)

    rho0 = analysis.delta() + 1e-70
    rho = rho0
    for t in range(big_steps+1):
        norm = analysis.integrate(rho)
        np.savetxt(sys.stdout.buffer, rho,
                   header='# t = {}\n# norm = {:.30f}'.format(t/big_steps, norm), footer='\n', comments='')
        print('big step: {}'.format(t), file=sys.stderr)
        for tt in range(small_steps//big_steps):
            rho = diffusion.step(rho)

