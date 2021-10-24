import numpy as np
import sys
import commentjson
from tqdm import tqdm

from analysis import Analysis
from ddft_shell import DDFTShell
from cutoff import Cutoff
from fexc.loader import load_functional
from initial import load_initial

log = lambda *args: print(*args, file=sys.stderr)


if __name__ == "__main__":
    log("*** trying to load config ***")
    try:
        configfile = sys.argv[1]
    except IndexError:
        configfile = "config.jsonc"
    with open(configfile, "r") as cf_file:
        config = commentjson.load(cf_file)

    log("*** initializing ***")
    dr, num_bins, bulk_density, rho_self, rho_dist = load_initial(config["initial_state"])
    cutoff = lambda a: Cutoff(1e-70).cutoff(a)
    rho_self = cutoff(rho_self)
    rho_dist = cutoff(rho_dist)
    log(" > initial state loaded.")

    try:
        base_functional = config["functional"]["base"]
    except KeyError:
        base_functional = None
    try:
        variant = config["functional"]["variant"]
    except KeyError:
        variant = None
    analysis = Analysis(dr, num_bins)
    f_exc = load_functional(base_functional, variant, analysis)

    log(" > functional loaded.")
    log("*** starting integration ***")

    j_s = np.zeros(num_bins)
    j_d = np.zeros(num_bins)
    D_rho_shell_self = np.zeros(num_bins)
    D_rho_shell_dist = np.zeros(num_bins)
    
    t0 = 0
    for phase in config["integration"]:
        log(" > {} phase".format(phase["name"]))
        small_steps = phase["small_steps"]
        big_steps = phase["big_steps"]
        simulation_time = phase["simulation_time"]
        ddft = DDFTShell(analysis, 1.0/small_steps/big_steps, f_exc, (rho_self, rho_dist), bulk_density, Cutoff(1e-70))
        for t in tqdm(range(int(simulation_time*big_steps)), position=0, desc='big steps', file=sys.stderr):
            norm_self, norm_dist = ddft.norms()
            np.savetxt(sys.stdout.buffer, np.hstack((rho_self.reshape(-1, 1), rho_dist.reshape(-1, 1), j_s.reshape(-1, 1), j_d.reshape(-1, 1), D_rho_shell_self.reshape(-1, 1), D_rho_shell_dist.reshape(-1, 1))),
                       header='# t = {}\n# norm = {:.30f}\t{:.30f}'.format(t / big_steps + t0, norm_self, norm_dist), footer='\n', comments='')
            for tt in tqdm(range(small_steps), position=1, desc='small steps', file=sys.stderr):
                rho_self, rho_dist, j_s, j_d, D_rho_shell_self, D_rho_shell_dist = ddft.step()
            if not (np.all(np.isfinite(rho_self)) and np.all(np.isfinite(rho_dist))):
                log("ERROR: invalid number detected in rho")
                sys.exit(1)
        t0 += simulation_time
        log(" > {} phase done".format(phase["name"]))
    norm_self, norm_dist = ddft.norms()
    np.savetxt(sys.stdout.buffer, np.hstack((rho_self.reshape(-1, 1), rho_dist.reshape(-1, 1), j_s.reshape(-1, 1), j_d.reshape(-1, 1), D_rho_shell_self.reshape(-1, 1), D_rho_shell_dist.reshape(-1, 1))),
               header='# t = {}\n# norm = {:.30f}\t{:.30f}'.format(t0, norm_self, norm_dist), footer='\n', comments='')
    log("*** done, have a nice day ***")
