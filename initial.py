import h5py
import numpy as np
from analysis import Analysis


def load_initial(filename) -> (float, int, np.array, np.array):
    with h5py.File(filename, "r") as h5_file:
        groupname = list(h5_file.keys())[0]
        group = h5_file[groupname]
        dr = group.attrs['dr'][0]
        num_bins = group.attrs['num_bins'][0]
        bulk_density = group.attrs['bulk_density'][0]
        data = group['vanhove']

        ana = Analysis(dr, num_bins)

        rho_self = data[0]['vanhove_self']
        rho_dist = data[0]['vanhove_distinct']

        rho_self /= ana.integrate(rho_self)
        rho_dist *= bulk_density
        return dr, num_bins, rho_self, rho_dist