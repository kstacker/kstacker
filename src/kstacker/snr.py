import numpy as np
from astropy.table import Table, vstack

from .imagerie import compute_noise_apertures, photometry, photometry_preprocessed
from .orbit import orbit


def compute_signal_and_noise_grid(
    x,
    ts,
    size,
    scale,
    fwhm,
    data,
    upsampling_factor,
    r_mask=None,
    method="convolve",
):
    nimg = len(data["images"])
    a, e, t0, m0, omega, i, theta_0 = x.T
    signal, noise = [], []

    # compute position
    for k in range(nimg):
        position = orbit.position(ts[k], a, e, t0, m0)
        position = orbit.project_position(position, omega, i, theta_0).T
        xx, yy = position

        # convert position into pixel in the image
        position = scale * position + size // 2
        temp_d = np.sqrt(xx**2 + yy**2) * scale  # get the distance to the center

        # compute the signal by integrating flux on a PSF, and correct it for
        # background (using pre-computed background profile)
        if method == "convolve":
            sig = photometry_preprocessed(
                data["images"][k], position[0], position[1], upsampling_factor
            )
        elif method == "aperture":
            sig = photometry(data["images"][k], position, 2 * fwhm)
        else:
            raise ValueError(f"invalid method {method}")

        sig -= np.interp(temp_d, data["x"], data["bkg"][k])

        if r_mask is not None:
            sig[temp_d <= r_mask] = 0.0

        signal.append(sig)

        # get noise at position using pre-computed radial noise profil
        noise.append(np.interp(temp_d, data["x"], data["noise"][k]))

    signal = np.nansum(signal, axis=0)
    noise = np.sqrt(np.nansum(np.array(noise) ** 2, axis=0))
    # if the value of total noise is 0 (i.e. all values of noise are 0,
    # i.e. the orbit is completely out of the image) then snr=0
    noise[np.isnan(noise) | (noise == 0)] = 1

    return signal, noise


def compute_snr_detailed(params, x, method=None, verbose=False):
    """Compute SNR for a set of parameters with detail per image."""

    x = np.atleast_2d(np.asarray(x, dtype="float32"))

    if x.shape[1] != 7:
        raise ValueError("x should have 7 columns for a,e,t0,m,omega,i,theta0")
    orbital_grid = x[:, :4]
    projection_grid = x[:, 4:]

    # load the images and the noise/background profiles
    data = params.load_data(method=method)

    # total time of the observation (years)
    ts = params.get_ts(use_p_prev=True)

    # solve kepler equation on the a/e/t0 grid for all images
    positions = orbit.positions_at_multiple_times(ts, orbital_grid)
    # (2, Nimages, Norbits) -> (Norbits, Nimages, 2)
    positions = np.ascontiguousarray(np.transpose(positions))

    # pre-compute the projection matrices
    omega, i, theta_0 = projection_grid.T
    proj_matrices = orbit.compute_projection_matrices(omega, i, theta_0)
    proj_matrices = np.ascontiguousarray(proj_matrices)

    # tables = []
    # names = "a e t0 omega i theta_0 signal noise snr".split()
    size = params.n
    nimg = len(data["images"])
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
                    data["images"][k],
                    position[k, :1],
                    position[k, 1:],
                    params.upsampling_factor,
                )[0]
            elif method == "aperture":
                sig = photometry(data["images"][k], position[k], 2 * params.fwhm)
            else:
                raise ValueError(f"invalid method {method}")

            if temp_d[k] <= params.r_mask or temp_d[k] >= params.r_mask_ext:
                sig = 0.0
            else:
                sig -= np.interp(temp_d[k], data["x"], data["bkg"][k])
            signal.append(sig)

            # get noise at position using pre-computed radial noise profil
            if sig == 0:
                noise.append(0)
            else:
                noise.append(np.interp(temp_d[k], data["x"], data["noise"][k]))

        res = Table(position, names=("xpix", "ypix"))
        res["signal"] = signal
        res["noise"] = noise
        res["xpix"].format = ".2f"
        res["ypix"].format = ".2f"
        res.add_column(np.arange(nimg), index=0, name="image")
        res.add_column([j], index=0, name="orbit")
        res.add_column((res["xpix"] - size // 2) * params.resol, index=4, name="xmas")
        res.add_column((res["ypix"] - size // 2) * params.resol, index=5, name="ymas")
        out.append(res)

        signal = np.nansum(signal, axis=0)
        noise = np.sqrt(np.nansum(np.array(noise) ** 2, axis=0))
        res.meta["signal_sum"] = signal
        res.meta["noise_sum"] = noise
        res.meta["snr_sum"] = signal / noise

        # With inverse variance weighting
        signal = np.sum(res["signal"] / res["noise"] ** 2) / np.sum(
            1 / res["noise"] ** 2
        )
        noise = np.sqrt(1 / np.sum(1 / res["noise"] ** 2))
        res.meta["signal_invvar"] = signal
        res.meta["noise_invvar"] = noise
        res.meta["snr_invvar"] = signal / noise

        if verbose:
            print(f"\nValues for orbit {j}, x = {x[j]}")
            print("Detail per image:")
            res.pprint(max_lines=-1, max_width=-1)
            print(f"Total signal={signal}, noise={noise}, SNR={signal / noise}")

    out = vstack(out)
    return out


def compute_snr(x, ts, size, scale, fwhm, data, invvar_weighted):
    """Compute theoretical snr in combined image."""

    nimg = len(data["images"])
    a, e, t0, m0, omega, i, theta_0 = x
    signal, noise = [], []

    # compute position
    positions = orbit.project_position(
        orbit.position(ts, a, e, t0, m0), omega, i, theta_0
    )
    # convert to pixel in the image
    positions = positions * scale + size // 2

    for k in range(nimg):
        # compute signal by integrating flux on a PSF, and correct it for
        # background
        img = data["images"][k]
        x, y = positions[k]
        # grid for photutils is centered on pixels hence the - 0.5
        bg, std, _ = compute_noise_apertures(
            img, x - 0.5, y - 0.5, fwhm, exclude_source=True, exclude_lobes=True
        )
        signal.append(photometry(img, positions[k], 2 * fwhm) - bg)
        noise.append(std)

    signal = np.array(signal)
    noise = np.array(noise)

    null = np.isnan(signal) | np.isclose(signal, 0)
    if np.all(null):
        return 0
    if np.any(null):
        noise = noise[~null]
        signal = signal[~null]

    if invvar_weighted:
        sigma_inv2 = np.sum(1 / noise**2)
        signal = np.sum(signal / noise**2) / sigma_inv2
        noise = np.sqrt(1 / sigma_inv2)
    else:
        signal = np.sum(signal)
        noise = np.sqrt(np.sum(noise**2))

    return -signal / noise
