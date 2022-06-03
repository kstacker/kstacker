import numpy as np
from astropy.table import Table, vstack

from ._utils import compute_snr
from .orbit import orbit


def compute_snr_values(params, x):
    """Compute SNR for a set of parameters."""

    x = np.atleast_2d(np.asarray(x, dtype='float32'))

    if x.shape[1] != 6:
        raise ValueError('x should have 6 columns for a,e,t0,omega,i,theta0')
    orbital_grid, projection_grid = np.split(x, 2, axis=1)

    # load the images and the noise/background profiles
    images, bkg_profiles, noise_profiles = params.load_data()

    # total time of the observation (years)
    ts = params.get_ts(use_p_prev=True)

    # solve kepler equation on the a/e/t0 grid for all images
    positions = orbit.positions_at_multiple_times(ts, orbital_grid, params.m0)
    # (2, Nimages, Norbits) -> (Norbits, Nimages, 2)
    positions = np.ascontiguousarray(np.transpose(positions))

    # pre-compute the projection matrices
    omega, i, theta_0 = projection_grid.T
    proj_matrices = orbit.compute_projection_matrices(omega, i, theta_0)
    proj_matrices = np.ascontiguousarray(proj_matrices)

    norbits = orbital_grid.shape[0]
    nproj = proj_matrices.shape[0]
    tables = []
    names = "a e t0 omega i theta_0 signal noise snr".split()

    for j in range(norbits):
        out = np.zeros((nproj, 3))
        compute_snr(
            images,
            positions[j],
            bkg_profiles,
            noise_profiles,
            proj_matrices,
            params.r_mask,
            params.scale,
            params.n,
            params.upsampling_factor,
            out,
            debug=1,
        )
        out = np.broadcast_arrays(orbital_grid, projection_grid, out)
        tables.append(Table(np.concatenate(out, axis=1), names=names))

    res = vstack(tables)
    res.pprint(max_lines=-1, max_width=-1)
    return res
