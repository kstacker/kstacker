import itertools
import os
import shutil

import numpy as np
import yaml

from .imagerie import photometry, photometry_preprocessed
from .orbit import orbit as orb


def create_output_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def get_image_suffix(method):
    if method == "convolve":
        print("Using pre-convolved images")
        return "_resampled"
    elif method == "aperture":
        print("Using photutils apertures")
        return "_preprocessed"
    else:
        raise ValueError(f"invalid method {method}")


def compute_signal_and_noise_grid(
    x,
    ts,
    m0,
    size,
    scale,
    fwhm,
    images,
    x_profile,
    bkg_profiles,
    noise_profiles,
    upsampling_factor,
    r_mask=None,
    method="convolve",
):
    nimg = len(images)
    a, e, t0, omega, i, theta_0 = x.T
    signal, noise = [], []

    # compute position
    for k in range(nimg):
        position = orb.position(ts[k], a, e, t0, m0)
        position = orb.project_position(position, omega, i, theta_0).T
        xx, yy = position

        # convert position into pixel in the image
        position = scale * position + size // 2
        temp_d = np.sqrt(xx**2 + yy**2) * scale  # get the distance to the center

        # compute the signal by integrating flux on a PSF, and correct it for
        # background (using pre-computed background profile)
        if method == "convolve":
            sig = photometry_preprocessed(images[k], position, upsampling_factor)
        elif method == "aperture":
            sig = photometry(images[k], position, 2 * fwhm)
        else:
            raise ValueError(f"invalid method {method}")

        sig -= np.interp(temp_d, x_profile, bkg_profiles[k])

        if r_mask is not None:
            sig[temp_d <= r_mask] = 0.0

        signal.append(sig)

        # get noise at position using pre-computed radial noise profil
        noise.append(np.interp(temp_d, x_profile, noise_profiles[k]))

    signal = np.nansum(signal, axis=0)
    noise = np.sqrt(np.nansum(np.array(noise) ** 2, axis=0))
    # if the value of total noise is 0 (i.e. all values of noise are 0,
    # i.e. the orbit is completely out of the image) then snr=0
    noise[np.isnan(noise) | (noise == 0)] = 1

    return signal, noise


class Grid:
    """
    Contains the information for each parameter of the grid:
    - min of the range
    - max of the range
    - original number of steps
    - number of splits

    """

    def __init__(self, params):
        self._params = params
        self._grid_params = ("a", "e", "t0", "omega", "i", "theta_0")

    def __repr__(self):
        out = ["Grid("]
        for name in self._grid_params:
            min_, max_, nsteps = self.limits(name)
            out.append(f"    {name}: {min_} → {max_}, {nsteps} steps")
        out.append(")")
        steps = [self.limits(name)[2] for name in self._grid_params]
        out.append(f"{np.prod(steps)} orbits")
        return "\n".join(out)

    def limits(self, name):
        """Return (min, max, nsteps) for a given grid parameter."""
        if name not in self._grid_params:
            raise ValueError(f"'{name}' is not a parameter of the grid")

        min_ = self._params[f"{name}_min"]
        max_ = self._params[f"{name}_max"]
        nsteps = self._params[f"N{name}"]
        return min_, max_, nsteps

    def bounds(self):
        """Return (min, max, nsteps) for a given grid parameter."""
        return [self.limits(name)[:2] for name in self._grid_params]

    def range(self, name):
        """Return a slice object for a given grid parameter."""
        if name not in self._grid_params:
            raise ValueError(f"'{name}' is not a parameter of the grid")
        min_, max_, nsteps = self.limits(name)
        return slice(min_, max_, (max_ - min_) / nsteps)

    def ranges(self):
        """Return the ranges for all grid params."""
        return [self.range(name) for name in self._grid_params]

    def make_2d_grid(self, params):
        lrange = [self.range(name) for name in params]
        grid = np.mgrid[lrange].astype(np.float32)
        # reshape grid to a 2D array: Norbits x Nparams
        grid = grid.reshape(grid.shape[0], -1).T
        return grid


class Params:
    """Handle parameters.

    Parameters are read from the YAML file and can be accessed as attributes or
    with a dict interface::

        >>> params = Params.read("parameters/near_alphacenA_b_fast.yml")
        >>> params.m0
        ... 1.133
        >>> params["m0"]
        ... 1.133

    """

    def __init__(self, params):
        self._params = params
        self.grid = Grid(params)

    def __getitem__(self, attr):
        if attr in self._params:
            return self._params[attr]
        else:
            raise KeyError

    def __getattr__(self, attr):
        if attr in self._params:
            return self._params[attr]
        else:
            raise AttributeError

    @classmethod
    def read(cls, filename):
        with open(filename) as f:
            params = yaml.safe_load(f)
        return cls(params)

    def get_path(self, key):
        return os.path.join(os.path.expanduser(self.work_dir), self._params[key])

    @property
    def wav(self):
        # force wav to be float since '2e-6' is parsed as string by pyyaml
        return float(self._params["wav"])

    @property
    def fwhm(self):
        """Apodized fwhm of the PSF (in pixels)."""
        return (
            (1.028 * self.wav / self.d) * (180.0 / np.pi) * 3600 / (self.resol / 1000.0)
        )

    @property
    def scale(self):
        """Scale factor used to convert pixel to astronomical unit (in pixel/a.u.)."""
        return 1.0 / (self.dist * (self.resol / 1000.0))
