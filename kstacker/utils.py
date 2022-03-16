import itertools
import os
import shutil

import numpy as np
import yaml


def create_output_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


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
            min_, max_, nsteps, nsplits = self.limits(name)
            out.append(f"    {name}: {min_} → {max_}, {nsteps} steps, {nsplits} splits")
        out.append(")")
        return "\n".join(out)

    def limits(self, name):
        """Return (min, max, nsteps, nsplits) for a given grid parameter."""
        if name not in self._grid_params:
            raise ValueError(f"'{name}' is not a parameter of the grid")

        min_ = self._params[f"{name}_min"]
        max_ = self._params[f"{name}_max"]
        nsteps = self._params[f"N{name}"]
        nsplits = 1 if name == "t0" else self._params[f"S{name}"]
        return min_, max_, nsteps, nsplits

    def range(self, name):
        """Return a slice object for a given grid parameter."""
        if name not in self._grid_params:
            raise ValueError(f"'{name}' is not a parameter of the grid")
        min_, max_, nsteps, nsplits = self.limits(name)
        return slice(min_, max_, (max_ - min_) / nsteps)

    @property
    def ranges(self):
        """Return the ranges for all grid params."""
        return [self.range(name) for name in self._grid_params]

    def split_range(self, name):
        """
        Returns a list of slice objects (the sub_ranges for the brute force).
        """

        min_, max_, nsteps, nsplits = self.limits(name)
        n = nsteps / nsplits

        if nsteps == 1:
            return np.array([slice(min_, max_, max_ - min_)], dtype=object)
        if n < 2.0:
            raise Exception("Number of values in each splited list under 2 : ABORTING")
        elif int(n) != n:
            print("n not int")
            if n - int(n) >= 0.5:
                n = float(int(n)) + 1.0
            elif n - int(n) < 0.5:
                n = float(int(n))

        print("min/max/nsteps/nsplits/n:", min_, max_, nsteps, nsplits, n)
        table = np.zeros(nsplits, dtype=object)
        delta = (max_ - min_) / nsplits
        x = min_
        for k in range(nsplits):
            # Dans version Antoine (bug):
            # if x==0.2 or delta==0.2 and x==0.4 or delta==0.4:
            # Pyhon decid that 0.2+0.4=6.000000001 and not 6.0
            # --> Split version is not working with this ><'
            #        x_max=0.6
            # else :
            #        x_max=x+delta
            x_max = round(x + delta, 12)
            table[k] = slice(x, x_max, delta / n)
            x = x_max
        return table

    def split_ranges(self):
        """Compute the combination of all splitted ranges."""
        ranges = [self.split_range(name) for name in self._grid_params]
        table = np.array(list(itertools.product(*ranges)), dtype=object)
        return table


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
