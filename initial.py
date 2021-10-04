import h5py
import numpy as np
from analysis import Analysis


def load_initial(filename) -> (float, int, np.array, np.array):
    with h5py.File(filename, "r") as h5_file:
        groupname = list(h5_file.keys())[0]
        group = h5_file[groupname]
        dr = group.attrs['dr'][0]
        num_bins = int(group.attrs['num_bins'][0] / 2)
        bulk_density = group.attrs['bulk_density'][0]
        data = group['vanhove']

        ana = Analysis(dr, num_bins)

        rho_self = data[0]['vanhove_self'][:num_bins]
        rho_dist = data[0]['vanhove_distinct'][:num_bins]

        rho_self /= ana.integrate(rho_self)
        rho_self -= 1e-13
        rho_dist *= bulk_density
        return dr, num_bins, bulk_density, rho_self, rho_dist
