import numpy as np

# from ._utils import compute_snr_debug
from .imagerie.analyze import photometry, photometry_preprocessed
from .orbit import orbit

# from astropy.table import Table, vstack


def compute_snr_values(params, x):
    """Compute SNR for a set of parameters."""

    x = np.atleast_2d(np.asarray(x, dtype="float32"))

    if x.shape[1] != 6:
        raise ValueError("x should have 6 columns for a,e,t0,omega,i,theta0")
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

    # tables = []
    # names = "a e t0 omega i theta_0 signal noise snr".split()
    size = params.n
    x_profile = np.linspace(0, size // 2 - 1, size // 2)

    for j in range(orbital_grid.shape[0]):
        signal, noise = [], []

        # project positions -> (Nimages, 2, Nvalid)
        position = np.dot(proj_matrices, positions[j].T).T
        position *= params.scale

        # distance to the center
        temp_d = np.hypot(position[:, 0, :], position[:, 1, :])

        # convert position into pixel in the image
        position += size // 2

        for k in range(len(images)):
            # compute the signal by integrating flux on a PSF, and correct it for
            # background (using pre-computed background profile)
            if params.method == "convolve":
                sig = photometry_preprocessed(
                    images[k], position[k, 0], position[k, 1], params.upsampling_factor
                )
            elif params.method == "aperture":
                sig = photometry(images[k], position[k], 2 * params.fwhm)
            else:
                raise ValueError(f"invalid method {params.method}")

            sig -= np.interp(temp_d[k], x_profile, bkg_profiles[k])

            if params.r_mask is not None:
                sig[temp_d[k] <= params.r_mask] = 0.0

            signal.append(sig)

            # get noise at position using pre-computed radial noise profil
            noise.append(np.interp(temp_d[k], x_profile, noise_profiles[k]))

        __import__("pdb").set_trace()
        # signal = np.nansum(signal, axis=0)
        # noise = np.sqrt(np.nansum(np.array(noise) ** 2, axis=0))
        # # if the value of total noise is 0 (i.e. all values of noise are 0,
        # # i.e. the orbit is completely out of the image) then snr=0
        # noise[np.isnan(noise) | (noise == 0)] = 1

        # orbit_idx = np.full(signal.shape[0], j, dtype=dtype_index)
        # res.append([orbit_idx, projection_grid_index[valid], signal, noise])

        # out = np.broadcast_arrays(orbital_grid, projection_grid, out)
        # tables.append(Table(np.concatenate(out, axis=1), names=names))

    # res = vstack(tables)
    # res.pprint(max_lines=-1, max_width=-1)
    # return res
