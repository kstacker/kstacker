import itertools
import os
import shutil

import numpy as np
import yaml


def create_output_dir(path):
    if os.path.exists(path):
        shutil.rmtree(path)
    os.mkdir(path)


def brute(func, ranges, args=(), nchunks=1):
    """Evaluate a function over a given range by brute force.

    Adapted from `scipy.optimize.brute`.

    Parameters
    ----------
    func : callable
        The objective function to be minimized. Must be in the
        form ``f(x, *args)``, where ``x`` is the argument in
        the form of a 1-D array and ``args`` is a tuple of any
        additional fixed parameters needed to completely specify
        the function.
    ranges : tuple
        Each component of the `ranges` tuple must be a "slice object".
        The program uses these to create the grid of points on which
        the objective function will be computed.
    args : tuple, optional
        Any additional fixed parameters needed to completely specify
        the function.

    Returns
    -------
    x0 : ndarray
        A 1-D array containing the coordinates of a point at which the
        objective function had its minimum value. (See `Note 1` for
        which point is returned.)
    fval : float
        Function value at the point `x0`.
    grid : tuple
        Representation of the evaluation grid. It has the same
        length as `x0`.
    Jout : ndarray
        Function values at each point of the evaluation
        grid, i.e., ``Jout = func(*grid)``.

    """
    N = len(ranges)
    if N == 1:
        raise ValueError("this case is not supported here")

    lrange = list(ranges)
    grid = np.mgrid[lrange]

    # obtain an array of parameters that is iterable by a map-like callable
    inpt_shape = grid.shape
    grid = np.reshape(grid, (inpt_shape[0], np.prod(inpt_shape[1:]))).T
    print('Grid shape:', grid.shape)

    Jout = []
    for i, chunk in enumerate(np.array_split(grid, nchunks)):
        # Jout += [func(x, *args) for x in chunk]
        print('- chunk', i)
        Jout.append(func(chunk, *args))

    Jout = np.concatenate(Jout, axis=1)
    Jout = np.reshape(Jout, (-1, ) + inpt_shape[1:])
    grid = np.reshape(grid.T, inpt_shape)
    return grid, Jout


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
