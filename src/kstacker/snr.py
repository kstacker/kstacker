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
        position = orbit.project_position_full(ts[k], a, e, t0, m0, omega, i, theta_0).T
        position *= scale  # convert position into pixel in the image
        temp_d = np.hypot(position[0], position[1])  # get the distance to the center
        position += size // 2

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


def compute_snr_detailed(
    params,
    x,
    method=None,
    invvar_weighted=False,
    exclude_source=False,
    exclude_lobes=False,
    use_interp_bgnoise=False,
    verbose=False,
):
    """Compute SNR for a set of parameters with detail per image."""

    x = np.atleast_2d(np.asarray(x, dtype="float32"))

    if x.shape[1] != 7:
        raise ValueError("x should have 7 columns for a,e,t0,m,omega,i,theta0")

    data = params.load_data(method=method)
    ts = params.get_ts(use_p_prev=True)
    size = params.n
    out = []
    method = method or params.method

    for j in range(len(x)):
        res = compute_snr(
            x[j],
            ts,
            size,
            params.scale,
            params.fwhm,
            data,
            invvar_weighted=invvar_weighted,
            exclude_source=exclude_source,
            exclude_lobes=exclude_lobes,
            method=method,
            upsampling_factor=params.upsampling_factor,
            use_interp_bgnoise=use_interp_bgnoise,
            return_all=True,
        )
        res.add_column([j], index=0, name="orbit")
        res.add_column((res["xpix"] - size // 2) * params.resol, index=4, name="xmas")
        res.add_column((res["ypix"] - size // 2) * params.resol, index=5, name="ymas")
        out.append(res)

        if verbose:
            print(f"\nValues for orbit {j}, x = {x[j]}")
            print("Detail per image:")
            res.pprint(max_lines=-1, max_width=-1)
            # print(f"Total signal={signal}, noise={noise}, SNR={signal / noise}")

    out = vstack(out)
    return out


def compute_snr(
    x,
    ts,
    size,
    scale,
    fwhm,
    data,
    invvar_weighted=False,
    exclude_source=True,
    exclude_lobes=True,
    method="aperture",
    upsampling_factor=None,
    use_interp_bgnoise=False,
    return_all=False,
):
    """Compute theoretical snr in combined image."""

    if method == "convolve" and not use_interp_bgnoise:
        print("Using interpolated bg/noise with convolve")
        use_interp_bgnoise = True

    # compute position
    a, e, t0, m0, omega, i, theta_0 = x
    positions = orbit.project_position_full(ts, a, e, t0, m0, omega, i, theta_0)
    # convert to pixel in the image
    positions *= scale
    # distance to the center
    if use_interp_bgnoise:
        temp_d = np.hypot(positions[:, 0], positions[:, 1])
    positions += size // 2

    signal, noise = [], []
    images = data["images"]
    for k in range(len(images)):
        # compute signal by integrating flux on a PSF, and correct it for background
        x, y = positions[k]

        # TODO: if temp_d[k] <= r_mask or temp_d[k] >= r_mask_ext:

        if use_interp_bgnoise:
            bg = np.interp(temp_d[k], data["x"], data["bkg"][k])
            std = np.interp(temp_d[k], data["x"], data["noise"][k])
        else:
            # grid for photutils is centered on pixels hence the - 0.5
            bg, std, _ = compute_noise_apertures(
                images[k],
                x - 0.5,
                y - 0.5,
                fwhm,
                exclude_source=exclude_source,
                exclude_lobes=exclude_lobes,
            )

        if method == "convolve":
            sig = photometry_preprocessed(
                images[k], positions[k, :1], positions[k, 1:], upsampling_factor
            )[0]
        elif method == "aperture":
            sig = photometry(images[k], positions[k], 2 * fwhm)
        else:
            raise ValueError(f"invalid method {method}")

        signal.append(sig - bg)
        noise.append(std)

    signal = np.array(signal)
    noise = np.array(noise)

    if return_all:
        tbl = Table(
            [np.arange(len(images)), positions[:, 0], positions[:, 1], signal, noise],
            names=("image", "xpix", "ypix", "signal", "noise"),
        )
        tbl["xpix"].format = ".2f"
        tbl["ypix"].format = ".2f"

    null = np.isnan(signal) | np.isclose(signal, 0)
    if np.all(null):
        if return_all:
            return tbl
        else:
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

    if return_all:
        key = "invvar" if invvar_weighted else "sum"
        tbl.meta[f"signal_{key}"] = signal
        tbl.meta[f"noise_{key}"] = noise
        tbl.meta[f"snr_{key}"] = signal / noise
        return tbl
    else:
        return signal / noise
