import os
import shutil

import numpy as np
import yaml
from astropy.io import fits
from astropy.table import Table, vstack

from .imagerie.analyze import photometry, photometry_preprocessed
from .orbit import orbit

__doctest_skip__ = ["Params"]


def create_output_dir(path, remove_if_exist=False):
    if remove_if_exist and os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path, exist_ok=True)


def compute_signal_and_noise_grid(
    x,
    ts,
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
                images[k], position[0], position[1], upsampling_factor
            )
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


def compute_snr_detailed(params, x, method=None, verbose=False):
    """Compute SNR for a set of parameters with detail per image."""

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
        res["xpix"].format = ".2f"
        res["ypix"].format = ".2f"
        res.add_column(np.arange(nimg), index=0, name="image")
        res.add_column([j], index=0, name="orbit")
        res.add_column((res["xpix"] - size // 2) * params.resol, index=4, name="xmas")
        res.add_column((res["ypix"] - size // 2) * params.resol, index=5, name="ymas")
        out.append(res)

        if verbose:
            print(f"\nValues for orbit {j}, x = {x[j]}")
            print("Detail per image:")
            res.pprint(max_lines=-1, max_width=-1)
            signal = np.nansum(signal, axis=0)
            noise = np.sqrt(np.nansum(np.array(noise) ** 2, axis=0))
            print(f"Total signal={signal}, noise={noise}, SNR={signal / noise}")

    out = vstack(out)
    return out


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
        self.grid_params = ("a", "e", "t0", "omega", "i", "theta_0")

    def __repr__(self):
        out = ["Grid("]
        for name in self.grid_params:
            min_, max_, nsteps = self.limits(name)
            out.append(f"    {name}: {min_} → {max_}, {nsteps} steps")
        out.append(")")
        steps = [self.limits(name)[2] for name in self.grid_params]
        nb = f"{np.prod(steps):,}".replace(",", "'")
        out.append(f"{nb} orbits")
        return "\n".join(out)

    def limits(self, name):
        """Return (min, max, nsteps) for a given grid parameter."""
        if name not in self.grid_params:
            raise ValueError(f"'{name}' is not a parameter of the grid")

        min_ = self._params[f"{name}_min"]
        max_ = self._params[f"{name}_max"]
        nsteps = self._params[f"N{name}"]
        return min_, max_, nsteps

    def bounds(self):
        """Return (min, max, nsteps) for a given grid parameter."""
        return [self.limits(name)[:2] for name in self.grid_params]

    def range(self, name):
        """Return a slice object for a given grid parameter."""
        if name not in self.grid_params:
            raise ValueError(f"'{name}' is not a parameter of the grid")
        min_, max_, nsteps = self.limits(name)
        return slice(min_, max_, (max_ - min_) / nsteps)

    def make_2d_grid(self, params):
        lrange = [self.range(name) for name in params]
        # TODO: use np.meshgrid(np.linspace(min, max, step), ...,
        # indexing='ij') instead
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
            raise KeyError(f"parameter {attr} is not defined")

    def __getattr__(self, attr):
        if attr in self._params:
            return self._params[attr]
        else:
            raise AttributeError(f"parameter {attr} is not defined")

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["grid"]
        return state

    def __setstate__(self, state):
        self._params = state["_params"]
        self.grid = Grid(self._params)

    @classmethod
    def read(cls, filename):
        with open(filename) as f:
            params = yaml.safe_load(f)
        return cls(params)

    def get_path(self, key, remove_if_exist=False):
        path = os.path.join(os.path.expanduser(self.work_dir), self._params[key])
        create_output_dir(path, remove_if_exist=remove_if_exist)
        return path

    def get_ts(self, use_p_prev=False):
        """Return the time for all the observations in years."""

        # 0 for real observations (time will be used)
        total_time = float(self.total_time)

        if total_time == 0:
            ts = np.array([float(x) for x in self.time.split("+")])
            if use_p_prev:
                ts = ts[self.p_prev :]
        else:
            # number of images
            if use_p_prev:
                nimg = self.p + self.p_prev
            else:
                nimg = self.p
            ts = np.linspace(0, total_time, nimg)

        print("time vector: ", ts)
        return ts

    def get_image_suffix(self, method=None):
        method = method or self.method
        if method == "convolve":
            print("Using pre-convolved images")
            return "_resampled"
        elif method == "aperture":
            print("Using photutils apertures")
            return "_preprocessed"
        else:
            raise ValueError(f"invalid method {method}")

    def load_data(self, selected=None, method=None):
        """Load the fits images and the noise/background profiles."""
        images_dir = self.get_path("images_dir")
        profile_dir = self.get_path("profile_dir")
        img_suffix = self.get_image_suffix(method=method)
        nimg = self.p + self.p_prev  # number of timesteps

        images, bkg_profiles, noise_profiles = [], [], []
        for k in range(nimg):
            if selected is not None and k not in selected:
                continue
            i = k + self.p_prev
            im = fits.getdata(f"{images_dir}/image_{i}{img_suffix}.fits")
            images.append(im.astype("float", order="C", copy=False))
            bkg_profiles.append(np.load(f"{profile_dir}/background_prof{i}.npy"))
            noise_profiles.append(np.load(f"{profile_dir}/noise_prof{i}.npy"))

        images = np.array(images)
        bkg_profiles = np.array(bkg_profiles)
        noise_profiles = np.array(noise_profiles)
        return images, bkg_profiles, noise_profiles

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
        return 1.0 / (self.dist * self.resol / 1000)
