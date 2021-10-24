import numpy as np
import sys
import commentjson
from tqdm import tqdm

from analysis import Analysis
from ddft_shell import DDFTShell
from cutoff import Cutoff
from fexc.loader import load_functional
from initial import load_initial
from util import log_state, log, get_functional_config

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
    log(" > initial state loaded.")

    base_functional, variant = get_functional_config(config)
    analysis = Analysis(dr, num_bins)
    f_exc = load_functional(base_functional, variant, analysis)

    log(" > functional loaded.")
    log("*** starting integration ***")

    r = np.arange(num_bins) * dr
    j_s = np.zeros(num_bins)
    j_d = np.zeros(num_bins)
    D_rho_shell_self = np.zeros(num_bins)
    D_rho_shell_dist = np.zeros(num_bins)

    t0 = 0
    ddft = None
    for phase in config["integration"]:
        log(" > {} phase".format(phase["name"]))
        small_steps = phase["small_steps"]
        big_steps = phase["big_steps"]
        simulation_time = phase["simulation_time"]
        ddft = DDFTShell(analysis, 1.0 / small_steps / big_steps, f_exc, (rho_self, rho_dist), bulk_density,
                         Cutoff(1e-70))
        for t in tqdm(range(int(simulation_time * big_steps)), position=0, desc='big steps', file=sys.stderr):
            norm_self, norm_dist = ddft.norms()
            log_state([
                ("r", r),
                ("rho_self", rho_self),
                ("rho_dist", rho_dist),
                ("j_self", j_s),
                ("j_dist", j_d),
                ("D_rho_shell_self", D_rho_shell_self),
                ("D_rho_shell_dist", D_rho_shell_dist)
            ], {
                "t": t / big_steps + t0,
                "norm_self": norm_self,
                "norm_dist": norm_dist
            })
            for tt in tqdm(range(small_steps), position=1, desc='small steps', file=sys.stderr):
                rho_self, rho_dist, j_s, j_d, D_rho_shell_self, D_rho_shell_dist = ddft.step()
            if not (np.all(np.isfinite(rho_self)) and np.all(np.isfinite(rho_dist))):
                log("ERROR: invalid number detected in rho")
                sys.exit(1)
        t0 += simulation_time
        log(" > {} phase done".format(phase["name"]))
    norm_self, norm_dist = ddft.norms()
    log_state([
        ("r", r),
        ("rho_self", rho_self),
        ("rho_dist", rho_dist),
        ("j_self", j_s),
        ("j_dist", j_d),
        ("D_rho_shell_self", D_rho_shell_self),
        ("D_rho_shell_dist", D_rho_shell_dist)
    ], {
        "t": t0,
        "norm_self": norm_self,
        "norm_dist": norm_dist
    })
    log("*** done, have a nice day ***")
