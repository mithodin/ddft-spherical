import numpy as np


def load_initial(filename) -> (float, int, float, np.array, np.array):
    with np.load(filename) as loaded:
        dr = loaded["dr"]
        num_bins = loaded["num_bins"]
        bulk_density = loaded["bulk_density"]
        rho_self = loaded["rho_self"]
        rho_dist = loaded["rho_dist"]
    return dr, num_bins, bulk_density, rho_self, rho_dist
