import numpy as np
from astropy.table import Table, vstack

from .imagerie.analyze import photometry, photometry_preprocessed
from .orbit import orbit


def compute_snr_values(params, x, method=None, verbose=False):
    """Compute SNR for a set of parameters."""

    x = np.atleast_2d(np.asarray(x, dtype="float32"))

    if x.shape[1] != 6:
        raise ValueError("x should have 6 columns for a,e,t0,omega,i,theta0")
    orbital_grid, projection_grid = np.split(x, 2, axis=1)

    # load the images and the noise/background profiles
    images, bkg_profiles, noise_profiles = params.load_data(method=method)

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
    nimg = len(images)
    x_profile = np.linspace(0, size // 2 - 1, size // 2)
    out = []
    method = method or params.method

    for j in range(orbital_grid.shape[0]):
        signal, noise = [], []

        # project positions -> (Nimages, 2, Nvalid)
        position = np.dot(proj_matrices[j], positions[j].T).T
        position *= params.scale

        # distance to the center
        temp_d = np.hypot(position[:, 0], position[:, 1])

        # convert position into pixel in the image
        position += size // 2

        for k in range(nimg):
            # compute the signal by integrating flux on a PSF, and correct it for
            # background (using pre-computed background profile)
            if method == "convolve":
                sig = photometry_preprocessed(
                    images[k],
                    position[k, :1],
                    position[k, 1:],
                    params.upsampling_factor,
                )[0]
            elif method == "aperture":
                sig = photometry(images[k], position[k], 2 * params.fwhm)
            else:
                raise ValueError(f"invalid method {method}")

            if temp_d[k] <= params.r_mask or temp_d[k] >= params.r_mask_ext:
                sig = 0.0
            else:
                sig -= np.interp(temp_d[k], x_profile, bkg_profiles[k])
            signal.append(sig)

            # get noise at position using pre-computed radial noise profil
            if sig == 0:
                noise.append(0)
            else:
                noise.append(np.interp(temp_d[k], x_profile, noise_profiles[k]))

        res = Table(position, names=("xpix", "ypix"))
        res["signal"] = signal
        res["noise"] = noise
        res.add_column(np.arange(nimg), index=0, name="image")
        res.add_column([j], index=0, name="orbit")
        out.append(res)

        if verbose:
            print(f"\nValues for orbit {j}, x = {x[j]}")
            print("Detail per image:")
            res.pprint(max_lines=-1, max_width=-1)
            signal = np.nansum(signal, axis=0)
            noise = np.sqrt(np.nansum(np.array(noise) ** 2, axis=0))
            print(f"Total signal={signal}, noise={noise}, SNR={signal / noise}")

    out = vstack(out)
    out["xpix"].format = ".2f"
    out["ypix"].format = ".2f"
    return out
