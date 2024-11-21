import numpy as np
from astropy.table import Table, vstack

from .imagerie import compute_noise_apertures, photometry, photometry_preprocessed
from .orbit import orbit
from mpmath import mp


def compute_log_likelihood(
    x,
    ts,
    size,
    scale,
    fwhm,
    data,
    exclude_source=True,
    exclude_lobes=True,
    method="aperture",
    upsampling_factor=None,
    use_interp_bgnoise=False,
    r_mask=None,
    r_mask_ext=None,
    return_all=False,
):
    """Compute likelihood in combined image."""

    if method == "convolve" and not use_interp_bgnoise:
        print("Using interpolated bg/noise with convolve")
        use_interp_bgnoise = True

    # compute position
    a, e, t0, m0, omega, i, theta_0 = x
    positions = orbit.project_position_full(ts, a, e, t0, m0, omega, i, theta_0)
    # convert to pixel in the image
    positions *= scale
    # distance to the center
    temp_d = np.hypot(positions[:, 0], positions[:, 1])
    positions += size // 2

    if r_mask is None:
        r_mask = fwhm
    if r_mask_ext is None:
        r_mask_ext = size // 2

    signal, noise = [], []
    images = data["images"]
    for k in range(len(images)):
        # compute signal by integrating flux on a PSF, and correct it for background
        x, y = positions[k]

        if temp_d[k] <= r_mask or temp_d[k] >= r_mask_ext:
            signal.append(np.nan)
            noise.append(np.nan)
            continue

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
            return -np.inf

    if np.any(null):
        noise = noise[~null]
        signal = signal[~null]

    if np.any(np.isnan(noise)):
        return -np.inf

    # loglikelihood = 0.5 * ((np.sum(signal / noise ** 2)) ** 2) / sigma_inv2 - 0.5 * np.sum(signal **2 / noise ** 2)

    try:
        sigma_inv2 = np.sum(1 / noise ** 2)
        loglikelihood = 0.5 * ((np.sum(signal / noise ** 2)) ** 2) / sigma_inv2
        if np.isnan(loglikelihood):
            return -np.inf
        return loglikelihood
    except ZeroDivisionError:
        return -np.inf