# ddft-spherical: dynamic density functional theory in the test particle limit
# Copyright (C) 2021  Lucas L. Treffenstädt and Thomas Schindler
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
from typing import cast

import numpy as np
import sys
import commentjson

from tqdm import tqdm
from analysis import Analysis
from cutoff import Cutoff
from ddft import DDFT
from ddft_shell import DDFTShell
from fexc import load_functional
from initial import load_initial_density_profiles
from logger import load_logger
from util import log, get_functional_config


def main() -> None:
    log("*** trying to load config ***")
    try:
        configfile = sys.argv[1]
    except IndexError:
        configfile = "config.jsonc"
    with open(configfile, "r") as cf_file:
        config = commentjson.load(cf_file)
    log("*** initializing ***")
    dr, num_bins, bulk_density, rho_self, rho_dist = load_initial_density_profiles(config["initial_state"])
    log(" > initial state loaded.")
    base_functional, variant = get_functional_config(config)
    analysis = Analysis(dr, num_bins)
    f_exc = load_functional(base_functional, variant, analysis)
    log(" > functional loaded.")
    r = np.arange(num_bins) * dr
    j_s = np.zeros(num_bins)
    j_d = np.zeros(num_bins)
    t0 = 0
    ddft = cast(DDFT, None)
    with load_logger(config["logger"]) as logger:
        log("*** starting integration ***")
        for phase in config["integration"]:
            log(" > {} phase".format(phase["name"]))
            small_steps = phase["small_steps"]
            big_steps = phase["big_steps"]
            simulation_time = phase["simulation_time"]
            ddft = DDFTShell(analysis, 1.0 / small_steps / big_steps, f_exc, (rho_self, rho_dist), bulk_density,
                             Cutoff(1e-70))
            for t in tqdm(range(int(simulation_time * big_steps)), position=0, desc='big steps', file=sys.stderr):
                norm_self, norm_dist = ddft.norms()
                logger.log_state([
                    ("r", r),
                    ("rho_self", rho_self),
                    ("rho_dist", rho_dist),
                    ("j_self", j_s),
                    ("j_dist", j_d)
                ], {
                    "t": t / big_steps + t0,
                    "norm_self": norm_self,
                    "norm_dist": norm_dist
                })
                for _ in tqdm(range(small_steps), position=1, desc='small steps', file=sys.stderr):
                    rho_self, rho_dist, j_s, j_d = ddft.step()
                if not (np.all(np.isfinite(rho_self)) and np.all(np.isfinite(rho_dist))):
                    log("ERROR: invalid number detected in rho")
                    sys.exit(1)
            t0 += simulation_time
            log(" > {} phase done".format(phase["name"]))
        norm_self, norm_dist = ddft.norms()
        logger.log_state([
            ("r", r),
            ("rho_self", rho_self),
            ("rho_dist", rho_dist),
            ("j_self", j_s),
            ("j_dist", j_d)
        ], {
            "t": t0,
            "norm_self": norm_self,
            "norm_dist": norm_dist
        })
    log("*** done, have a nice day ***")


if __name__ == "__main__":
    main()
