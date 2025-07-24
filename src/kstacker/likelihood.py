import numpy as np
from astropy.table import Table, vstack

from .imagerie import compute_noise_apertures, photometry, photometry_preprocessed
from .orbit import orbit
from mpmath import mp

from kstacker.PSF_shape_mcmc import aperture, PSF
# from kstacker.run_matrix_mcmc import MCMCCstData


def compute_log_likelihood(
    x,
    CstData,
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
    positions = orbit.project_position_full(CstData.ts, a, e, t0, m0, omega, i, theta_0)
    # convert to pixel in the image
    positions *= CstData.scale
    # distance to the center
    temp_d = np.hypot(positions[:, 0], positions[:, 1])
    positions += CstData.size // 2
    N,M,_ = np.shape(CstData.data['images'])
    
    all_mask, all_pixel_indices = aperture(positions,N,M,CstData.fwhm,CstData.PSF_shape)
    g_values = PSF(positions,N,M,CstData,all_pixel_indices,all_mask)

    if r_mask is None:
        r_mask = CstData.fwhm
    if r_mask_ext is None:
        r_mask_ext = CstData.size // 2

    signal, noise, background = [], [], []
    images = CstData.data["images"]
    for k in range(len(images)):
        # compute signal by integrating flux on a PSF, and correct it for background
        x, y = positions[k]

        if temp_d[k] <= r_mask or temp_d[k] >= r_mask_ext:
            signal.append(np.nan)
            noise.append(np.nan)
            background.append(np.nan)
            continue

        if use_interp_bgnoise:
            bg = np.interp(temp_d[k], CstData.data["x"], CstData.data["bkg"][k])
            std = np.interp(temp_d[k], CstData.data["x"], CstData.data["noise"][k])
        else:
            # grid for photutils is centered on pixels hence the - 0.5
            bg, std, _ = compute_noise_apertures(
                images[k],
                x - 0.5,
                y - 0.5,
                CstData.fwhm,
                exclude_source=exclude_source,
                exclude_lobes=exclude_lobes,
            )

        if method == "convolve":
            sig = photometry_preprocessed(
                images[k], positions[k, :1], positions[k, 1:], upsampling_factor
            )[0]
        elif method == "aperture":
            sig = photometry(images[k], positions[k], 2 * CstData.fwhm)
        else:
            raise ValueError(f"invalid method {method}")

        signal.append(sig - bg)
        noise.append(std)
        background.append(bg)

    signal = np.array(signal)
    noise = np.array(noise)
    background = np.array(background)
    
    if (np.sum(signal/noise**2)/np.sum(1/noise**2)) <0:
        return -np.inf

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
        background = background[~null]

    if np.any(np.isnan(noise)):
        return -np.inf

    # loglikelihood = 0.5 * ((np.sum(signal / noise ** 2)) ** 2) / sigma_inv2 - 0.5 * np.sum(signal **2 / noise ** 2)

    try:
        if CstData.PSF_shape == "Circle":
            sigma_inv2 = np.sum(1 / noise ** 2)
            loglikelihood = 0.5 * ((np.sum(signal / noise ** 2)) ** 2) / sigma_inv2
        
        
        if CstData.PSF_shape == "Bessel":
            n = 0
            d = 0
            cpt = 0
            for k in range(N):
                if not null[k]:
                    image_size_y, image_size_x = all_pixel_indices[k][0]
                    resize_y, resize_x = all_pixel_indices[k][1]
                    S = images[k][image_size_y, image_size_x]
                    G = g_values[k][resize_y, resize_x]
                    Mask = all_mask[k][resize_y, resize_x]
                    # Mask is photutils mask, it is needed to extract the flux
                    n += (np.sum(S*G*Mask)-background[cpt])/noise[cpt]**2
                    d += (np.sum(G*Mask)**2/noise[cpt]**2)
                    cpt+=1
                    
            loglikelihood = 0.5*n**2/d
        
        if np.isnan(loglikelihood):
            return -np.inf
        return loglikelihood
    except ZeroDivisionError:
        return -np.inf