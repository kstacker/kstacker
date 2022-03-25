import argparse

import numpy as np

from .best_solutions import find_best_solutions
from .gradient_reoptimization import reoptimize_gradient
from .noise_profile import compute_noise_profiles
from .optimize import brute_force
from .utils import Params
from .version import version

# to plot in a file without a display screen (cluster)
# isort: off
import matplotlib

matplotlib.use("Agg")  # noqa
# isort: on


def main():
    parser = argparse.ArgumentParser(description="K-Stacker")
    parser.add_argument("--verbose", action="store_true", help="verbose flag")
    parser.add_argument("--version", action="version", version=f"%(prog)s {version}")
    subparsers = parser.add_subparsers(title="subcommands", help="")

    sub_prof = subparsers.add_parser("noise_profiles", help="compute noise profiles")
    sub_prof.add_argument("parameter_file", help="Parameter file (yml)")
    sub_prof.add_argument("--seed", type=int, help="seed for random numbers")
    sub_prof.set_defaults(func=noise_profiles)

    sub_opt = subparsers.add_parser(
        "optimize", help="compute signal and noise on a grid (brute force)"
    )
    sub_opt.add_argument("parameter_file", help="Parameter file (yml)")
    sub_opt.set_defaults(func=optimize)

    sub_best = subparsers.add_parser("find_best", help="find the best solutions")
    sub_best.add_argument("parameter_file", help="Parameter file (yml)")
    sub_best.set_defaults(func=find_best)

    sub_reopt = subparsers.add_parser(
        "reopt",
        help="re-optimize the best values of SNR with a gradient descent method",
    )
    sub_reopt.add_argument("parameter_file", help="Parameter file (yml)")
    sub_reopt.set_defaults(func=reoptimize)

    try:
        args = parser.parse_args()
        args.func(args)
    except AttributeError:
        parser.print_usage()


def noise_profiles(args):
    if args.seed:
        np.random.seed(args.seed)
    params = Params.read(args.parameter_file)
    compute_noise_profiles(params)


def optimize(args):
    params = Params.read(args.parameter_file)
    brute_force(params)


def find_best(args):
    params = Params.read(args.parameter_file)
    find_best_solutions(params)


def reoptimize(args):
    params = Params.read(args.parameter_file)
    reoptimize_gradient(params)
