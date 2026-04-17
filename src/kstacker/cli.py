# cli.py
# ======
#
# Command-line interface for the reduced K-Stacker workflow.
#
# Supported subcommands:
#   - noise_profiles
#   - optimize
#   - reopt
#   - mcmc
#
# This CLI intentionally exposes only the commands that are still part of the
# current workflow.

import argparse
import sys
import time

import numpy as np

from .gradient_reoptimization import reoptimize_gradient
from .mcmc import run_mcmc_from_yaml
from .noise_profile import (
    compute_mcmc_noise_signal_profil,
    compute_noise_profiles,
    compute_snr_plots,
)
from .optimize import brute_force
from .utils import Params
from .version import version


def main():
    """
    Build the CLI parser, dispatch the selected subcommand, and report runtime.
    """
    parser = argparse.ArgumentParser(
        description="K-Stacker command-line interface"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="drop into pdb if an exception is raised",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="reserved verbose flag",
    )
    parser.add_argument(
        "--version",
        action="version",
        version=f"%(prog)s {version}",
    )

    subparsers = parser.add_subparsers(
        title="subcommands",
        dest="command",
        help="available commands",
    )

    # -------------------------------------------------------------------------
    # noise_profiles
    # -------------------------------------------------------------------------
    noise_parser = subparsers.add_parser(
        "noise_profiles",
        help="compute background/noise profiles and diagnostic products",
    )
    noise_parser.add_argument(
        "parameter_file",
        help="path to the YAML parameter file",
    )
    noise_parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="random seed used by profile-related stochastic steps",
    )
    noise_parser.set_defaults(func=run_noise_profiles_command)

    # -------------------------------------------------------------------------
    # optimize
    # -------------------------------------------------------------------------
    optimize_parser = subparsers.add_parser(
        "optimize",
        help="run the brute-force grid search",
    )
    optimize_parser.add_argument(
        "parameter_file",
        help="path to the YAML parameter file",
    )
    optimize_parser.add_argument(
        "--nthreads",
        type=int,
        default=0,
        help="number of OpenMP/Cython threads (0 lets the backend decide)",
    )
    optimize_parser.add_argument(
        "--progress",
        action="store_true",
        help="show progress information during the brute-force loop",
    )
    optimize_parser.add_argument(
        "--dry-run",
        action="store_true",
        help="build and validate the grids without running the full computation",
    )
    optimize_parser.set_defaults(func=run_optimize_command)

    # -------------------------------------------------------------------------
    # reopt
    # -------------------------------------------------------------------------
    reopt_parser = subparsers.add_parser(
        "reopt",
        help="re-optimize the best brute-force solutions with gradient descent",
    )
    reopt_parser.add_argument(
        "parameter_file",
        help="path to the YAML parameter file",
    )
    reopt_parser.add_argument(
        "--njobs",
        type=int,
        default=1,
        help="number of parallel processes (-1 uses all available CPUs)",
    )
    reopt_parser.add_argument(
        "--norbits",
        type=int,
        default=None,
        help="number of candidate orbits to re-optimize (default: all available)",
    )
    reopt_parser.set_defaults(func=run_reopt_command)

    # -------------------------------------------------------------------------
    # mcmc
    # -------------------------------------------------------------------------
    mcmc_parser = subparsers.add_parser(
        "mcmc",
        help="run the YAML-driven MCMC orbital inference pipeline",
    )
    mcmc_parser.add_argument(
        "parameter_file",
        help="path to the YAML parameter file",
    )
    mcmc_parser.set_defaults(func=run_mcmc_command)

    args = parser.parse_args()

    if args.debug:
        install_post_mortem_debug_hook()

    if hasattr(args, "func"):
        start_time = time.time()
        args.func(args)
        elapsed_time = time.time() - start_time
        print(f"Done: took {elapsed_time:.2f} sec.")
        return

    parser.print_usage()


def install_post_mortem_debug_hook():
    """
    Replace the default exception hook with a post-mortem pdb hook.
    """

    def run_pdb(exc_type, exc_value, traceback_obj):
        import pdb
        import traceback

        traceback.print_exception(exc_type, exc_value, traceback_obj)
        pdb.pm()

    sys.excepthook = run_pdb


def run_noise_profiles_command(args):
    """
    Compute radial noise/background profiles and optional diagnostic products.
    """
    if args.seed is not None:
        np.random.seed(args.seed)
        print(f"[cli] Random seed set to {args.seed}")

    params = Params.read(args.parameter_file)

    print("[cli] Running noise profile workflow")
    print(f"[cli] Parameter file: {args.parameter_file}")

    if getattr(params, "noise_prof", "no") == "yes":
        print("[cli] Computing radial background and noise profiles")
        compute_noise_profiles(params)
    else:
        print("[cli] Skipping radial profile computation because noise_prof != 'yes'")

    if getattr(params, "snr_plot", "no") == "yes":
        print("[cli] Computing SNR diagnostic plots")
        compute_snr_plots(params)
    else:
        print("[cli] Skipping SNR plots because snr_plot != 'yes'")

    print("[cli] Computing MCMC profile products")
    compute_mcmc_noise_signal_profil(params)


def run_optimize_command(args):
    """
    Run the brute-force search on the orbital grid.
    """
    params = Params.read(args.parameter_file)

    print("[cli] Running brute-force optimization")
    print(f"[cli] Parameter file: {args.parameter_file}")
    print(f"[cli] Threads: {args.nthreads}")
    print(f"[cli] Dry run: {args.dry_run}")
    print(f"[cli] Progress enabled: {args.progress}")

    brute_force(
        params,
        dry_run=args.dry_run,
        num_threads=args.nthreads,
        show_progress=args.progress,
    )


def run_reopt_command(args):
    """
    Run the gradient-based re-optimization starting from brute-force solutions.
    """
    params = Params.read(args.parameter_file)

    print("[cli] Running gradient re-optimization")
    print(f"[cli] Parameter file: {args.parameter_file}")
    print(f"[cli] Parallel jobs: {args.njobs}")
    print(f"[cli] Number of orbits to re-optimize: {args.norbits}")

    reoptimize_gradient(
        params,
        n_jobs=args.njobs,
        n_orbits=args.norbits,
    )


def run_mcmc_command(args):
    """
    Run the end-to-end MCMC pipeline from the YAML configuration file.
    """
    print("[cli] Running MCMC pipeline")
    print(f"[cli] Parameter file: {args.parameter_file}")

    sampler, flat_chain = run_mcmc_from_yaml(args.parameter_file)

    values_directory = Params.read(args.parameter_file).get_path("values_dir")

    print(f"[mcmc] Saved sampling outputs to: {values_directory}")
    print(f"[mcmc] Saved artifacts in: {values_directory}")
    print(f"[mcmc] Flat chain shape: {flat_chain.shape}")


if __name__ == "__main__":
    main()