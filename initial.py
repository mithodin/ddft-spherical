import numpy as np
from typing import Tuple
from cutoff import Cutoff


def load_initial_density_profiles(filename: str) -> Tuple[float, int, float, np.ndarray, np.ndarray]:
    with np.load(filename) as loaded:
        dr = loaded["dr"]
        num_bins = loaded["num_bins"]
        bulk_density = loaded["bulk_density"]
        rho_self = loaded["rho_self"]
        rho_dist = loaded["rho_dist"]
    cutoff = Cutoff(1e-70)
    rho_self = cutoff.cutoff(rho_self)
    rho_dist = cutoff.cutoff(rho_dist)
    return dr, num_bins, bulk_density, rho_self, rho_dist


def export(
        filename: str, dr: float, num_bins: int, bulk_density: float, rho_self: np.ndarray, rho_dist: np.ndarray
) -> None:
    """
    Save a compressed numpy file in the correct format for the loader to load
    :param filename: the name of the file (should include the .npz file extension)
    :param dr: the bin size of the discretised density profiles
    :param num_bins: the number of points of the discretised density profiles
    :param bulk_density: the mean density of the system
    :param rho_self: the density profile of the self particle
    :param rho_dist: the density profile of the distinct particles
    """
    data = {
        "dr": dr,
        "num_bins": num_bins,
        "bulk_density": bulk_density,
        "rho_self": rho_self,
        "rho_dist": rho_dist
    }
    np.savez_compressed(filename, **data)
