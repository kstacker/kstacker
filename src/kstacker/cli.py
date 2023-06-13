import argparse
import sys
import time

import numpy as np

from .gradient_reoptimization import reoptimize_gradient
from .mcmc_reoptimization import reoptimize_mcmc
from .noise_profile import compute_noise_profiles, compute_snr_plots
from .optimize import brute_force, extract_best_solutions
from .utils import Params
from .version import version


def main():
    parser = argparse.ArgumentParser(description="K-Stacker")
    parser.add_argument("--debug", action="store_true", help="debug flag")
    parser.add_argument("--verbose", action="store_true", help="verbose flag")
    parser.add_argument("--version", action="version", version=f"%(prog)s {version}")
    subparsers = parser.add_subparsers(title="subcommands", help="")

    # noise_profiles parser
    sub_prof = subparsers.add_parser("noise_profiles", help="compute noise profiles")
    sub_prof.add_argument("parameter_file", help="Parameter file (yml)")
    sub_prof.add_argument("--seed", type=int, help="seed for random numbers")
    sub_prof.set_defaults(func=noise_profiles)

    # optimize parser
    sub_opt = subparsers.add_parser(
        "optimize", help="compute signal and noise on a grid (brute force)"
    )
    sub_opt.add_argument("parameter_file", help="Parameter file (yml)")
    sub_opt.add_argument("--nthreads", type=int, default=0, help="number of threads")
    sub_opt.add_argument("--progress", action="store_true", help="show progress")
    sub_opt.add_argument(
        "--dry-run", action="store_true", help="do not run computation"
    )
    sub_opt.set_defaults(func=optimize)

    # extractbest parser
    sub_bestsol = subparsers.add_parser(
        "extractbest",
        help=(
            "Sort on the SNR column and store the q best results "
            "(already done at the end of optimize)"
        ),
    )
    sub_bestsol.add_argument("parameter_file", help="Parameter file (yml)")
    sub_bestsol.add_argument(
        "--nbest", type=int, help="number of orbits (params.q by default)"
    )
    sub_bestsol.set_defaults(func=extract_best)

    # reopt parser
    sub_reopt = subparsers.add_parser(
        "reopt", help="re-optimize the best SNR values with a gradient descent"
    )
    sub_reopt.add_argument("parameter_file", help="Parameter file (yml)")
    sub_reopt.add_argument(
        "--njobs", type=int, default=1, help="number of processes (-1 to use all CPUs)"
    )
    sub_reopt.add_argument(
        "--norbits", type=int, help="number of orbits (all by default)"
    )
    sub_reopt.set_defaults(func=reoptimize)

    # mcmc parser
    sub_mcmc = subparsers.add_parser(
        "mcmc", help="re-optimize the best SNR values with mcmc"
    )

    sub_mcmc.add_argument("parameter_file", help="Parameter file (yml)")
    sub_mcmc.add_argument(
        "--n_walkers", type=int, default=2, help="n_walkers = njobs = 2 by default"
    )
    sub_mcmc.add_argument(
        "--n_steps", type=int, default=1000, help="number of mcmc steps (1000 by default)"
    )
    sub_mcmc.set_defaults(func=reopt_mcmc)

    args = parser.parse_args()

    if args.debug:

        def run_pdb(type, value, tb):
            import pdb
            import traceback

            traceback.print_exception(type, value, tb)
            pdb.pm()

        sys.excepthook = run_pdb

    if "func" in args:
        t0 = time.time()
        args.func(args)
        print(f"Done: took {time.time() - t0:.2f} sec.")
    else:
        parser.print_usage()


def noise_profiles(args):
    if args.seed:
        np.random.seed(args.seed)
    params = Params.read(args.parameter_file)
    if params.noise_prof == "yes":
        compute_noise_profiles(params)
    if params.snr_plot == "yes":
        compute_snr_plots(params)


def optimize(args):
    params = Params.read(args.parameter_file)
    brute_force(
        params,
        dry_run=args.dry_run,
        num_threads=args.nthreads,
        show_progress=args.progress,
    )


def reoptimize(args):
    params = Params.read(args.parameter_file)
    reoptimize_gradient(params, n_jobs=args.njobs, n_orbits=args.norbits)


def extract_best(args):
    params = Params.read(args.parameter_file)
    extract_best_solutions(params, nbest=args.nbest)


def reopt_mcmc(args):
    params = Params.read(args.parameter_file)
    reoptimize_mcmc(params, n_walkers=args.n_walkers, n_steps=args.n_steps)
