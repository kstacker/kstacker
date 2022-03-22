import argparse

import numpy as np

from .noise_profile import compute_noise_profiles
from .optimize import brute_force
from .utils import Params

# to plot in a file without a display screen (cluster)
# isort: off
import matplotlib

matplotlib.use("Agg")  # noqa
# isort: on


def main():
    parser = argparse.ArgumentParser(description="K-Stacker")
    parser.add_argument("--verbose", action="store_true", help="verbose flag")
    subparsers = parser.add_subparsers(title="subcommands", help="")

    sub1 = subparsers.add_parser("noise_profiles", help="compute noise profiles")
    sub1.add_argument("parameter_file", help="Parameter file (yml)")
    sub1.add_argument("--seed", type=int, help="seed for random numbers")
    sub1.set_defaults(func=noise_profiles)

    sub2 = subparsers.add_parser(
        "optimize", help="compute signal and noise on a grid (brute force)"
    )
    sub2.add_argument("parameter_file", help="Parameter file (yml)")
    sub2.set_defaults(func=optimize)

    args = parser.parse_args()
    args.func(args)


def noise_profiles(args):
    if args.seed:
        np.random.seed(args.seed)
    params = Params.read(args.parameter_file)
    compute_noise_profiles(params)


def optimize(args):
    params = Params.read(args.parameter_file)
    brute_force(params)
