import argparse
import sys
import time
import ast

import numpy as np

from .gradient_reoptimization import compute_detailed_positions, reoptimize_gradient
from .mcmc_reoptimization import reoptimize_mcmc
from .mcmc_starting_pos import build_mcmc_starting_position
from .run_matrix_mcmc import compute_mcmc_matrix
from .noise_profile import compute_noise_profiles, compute_snr_plots, compute_mcmc_noise_signal_profil
from .optimize import brute_force, extract_best_solutions
from .utils import Params
from .version import version

def main():
    parser = argparse.ArgumentParser(description="K-Stacker")
    parser.add_argument("--debug", action="store_true", help="debug flag")
    parser.add_argument("--verbose", action="store_true", help="verbose flag")
    parser.add_argument("--version", action="version", version=f"%(prog)s {version}")
    subparsers = parser.add_subparsers(title="subcommands", help="")

    # ---------------------------------------------------------------
    # noise_profiles parser
    sub_prof = subparsers.add_parser("noise_profiles", help="compute noise profiles")
    sub_prof.add_argument("parameter_file", help="Parameter file (yml)")
    sub_prof.add_argument("--seed", type=int, help="seed for random numbers")
    sub_prof.set_defaults(func=noise_profiles)

    # ---------------------------------------------------------------
    # matrix mcmc image builder parser
    sub_mcmc_image = subparsers.add_parser("mcmc_matrix_image", help="Building the images for mcmc matrix")
    sub_mcmc_image.add_argument("parameter_file", help="Parameter file (yml)")
    sub_mcmc_image.add_argument("--angle", type=float, default=1, help="fraction of the circle in witch we take for the noise profil (default 1)")
    sub_mcmc_image.set_defaults(func=build_mcmc_matrix_images)

    # ---------------------------------------------------------------
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

    # ---------------------------------------------------------------
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

    # ---------------------------------------------------------------
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

    # ---------------------------------------------------------------
    # mcmc parser
    sub_mcmc = subparsers.add_parser(
        "mcmc", help="Search for the best K-stacker solutions by optimizing a likelihood function using the emcee sampler."
    )
    sub_mcmc.add_argument("parameter_file", help="Parameter file (yml)")
    sub_mcmc.add_argument(
        "--njobs", type=int, default=1, help="number of processes (default=1; -1 to use all CPUs)"
    )
    sub_mcmc.add_argument(
        "--nwalkers", type=int, default=14, help="number of walkers (default=14)"
    )
    sub_mcmc.add_argument(
        "--nsteps", type=int, default=150000, help="number of max mcmc steps (default 150 000)"
    )
    sub_mcmc.add_argument(
        "--norbits", type=int, default=1000, help="number of mcmc orbits saved (default 1000)"
    )
    sub_mcmc.add_argument(
        "--ncheck", type=int, default=100, help="number of mcmc check (default 100)"
    )
    sub_mcmc.add_argument(
        "--fixedparams", type=str, default=None, help="define the fixed parameters, example \"{'inc': 60, 'e': 0.1}\" (default None)"
    )
    sub_mcmc.add_argument(
        "--nbrpsf", type=float, default=1., help="number of psf to define the searching bounds (default 1.)"
    )
    sub_mcmc.add_argument(
        "--initposprecomputed", type=bool, default=False, help="define if the initial postion are precompute or not (default False)"
    )
    sub_mcmc.add_argument(
        "--PSFshape", type=str, default='Circle', help="name of the used PSF, possible value 'Bessel', 'Circle'. (default 'Circle')."
    )
    sub_mcmc.set_defaults(func=reopt_mcmc)

    # ---------------------------------------------------------------
    # matrix mcmc parser
    sub_mcmc_matrix = subparsers.add_parser(
        "mcmc_matrix", help="Search for the best K-stacker solutions by optimizing a likelihood function using the emcee sampler."
    )
    sub_mcmc_matrix.add_argument("parameter_file", help="Parameter file (yml)")
    sub_mcmc_matrix.add_argument(
        "--njobs", type=int, default=4, help="number of processes (default=1; -1 to use all CPUs)"
    )
    sub_mcmc_matrix.add_argument(
        "--nwalkers", type=int, default=28, help="number of walkers (default=14)"
    )
    sub_mcmc_matrix.add_argument(
        "--nsteps", type=int, default=1000000, help="number of max mcmc steps (default 150 000)"
    )
    sub_mcmc_matrix.add_argument(
        "--norbits", type=int, default=1000, help="number of mcmc orbits saved (default 1000)"
    )
    sub_mcmc_matrix.add_argument(
        "--ncheck", type=int, default=1000, help="number of mcmc check (default 100)"
    )
    sub_mcmc_matrix.add_argument(
        "--fixedparams", type=str, default=None, help="define the fixed parameters, example \"{'i': 60, 'e': 0.1}\" (default None)"
    )
    sub_mcmc_matrix.add_argument(
        "--nbrpsf", type=float, default=1., help="number of psf to define the searching bounds (default 1.)"
    )
    sub_mcmc_matrix.add_argument(
        "--initposprecomputed", type=bool, default=False, help="define if the initial postion are precompute or not (default False)"
    )
    sub_mcmc_matrix.add_argument(
        "--PSFshape", type=str, default='Bessel', help="name of the used PSF, possible value 'Bessel', 'Circle'. (default 'Bessel')."
    )
    sub_mcmc_matrix.set_defaults(func=reopt_mcmc_matrix)

    # ---------------------------------------------------------------
    # matrix mcmc init pos parser
    sub_mcmc_matrix = subparsers.add_parser(
        "mcmc_starting_pos", help="Create starting position for every walkers in both Likelihood methods"
    )
    sub_mcmc_matrix.add_argument("parameter_file", help="Parameter file (yml)")
    sub_mcmc_matrix.add_argument(
        "--nwalkers", type=int, default=28, help="number of walkers (default=14)"
    )
    sub_mcmc_matrix.add_argument(
        "--fixedparams", type=str, default=None, help="define the fixed parameters, example \"{'i': 60, 'e': 0.1}\" (default None)"
    )
    sub_mcmc_matrix.add_argument(
        "--nbrpsf", type=float, default=1., help="number of psf to define the searching bounds (default 1.)"
    )
    sub_mcmc_matrix.add_argument(
        "--fixbounds", type=str, default=None, help="define an upper, a lower or both limits for a parameters parameters, example \"{'i': {'bounds': 'lower', 'value': 60}, 'e': {'bounds': 'lower', 'value': 60}, 'e': {'bounds': 'both', 'value': [60,50]}}\" (default None)"
    )
    sub_mcmc_matrix.set_defaults(func=mcmc_starting_pos)

    # ---------------------------------------------------------------
    # recompute_positions parser
    sub_pos = subparsers.add_parser(
        "recompute_positions",
        help=(
            "recompute (after the gradient optimization) the positions, "
            "signal and noise in each image"
        ),
    )
    sub_pos.add_argument("parameter_file", help="Parameter file (yml)")
    sub_pos.add_argument(
        "--method",
        default="aperture",
        help="method to integrate the signal: aperture (default) or convolve",
    )
    sub_pos.add_argument(
        "--invvar_weight",
        type=int,
        help=(
            "1 to use inverse variance weighting, 0 to disable it, by default "
            "the value from the parameter file is used (invvar_weight)"
        ),
    )
    sub_pos.set_defaults(func=recompute_positions)

    # ---------------------------------------------------------------
    # parse arguments
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


if __name__ == "__main__":
    main()


def noise_profiles(args):
    if args.seed:
        np.random.seed(args.seed)
    params = Params.read(args.parameter_file)
    if params.noise_prof == "yes":
        compute_noise_profiles(params)
    if params.snr_plot == "yes":
        compute_snr_plots(params)
        

def build_mcmc_matrix_images(args):
    params = Params.read(args.parameter_file)
    compute_mcmc_noise_signal_profil(params,angle=args.angle)


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
    if args.fixedparams is not None:
        fixedparams = ast.literal_eval(args.fixedparams)
    else:
        fixedparams = None
    params = Params.read(args.parameter_file)
    reoptimize_mcmc(params, n_jobs=args.njobs, n_walkers=args.nwalkers, n_steps=args.nsteps, n_orbits=args.norbits, n_check=args.ncheck, fixed_params=fixedparams, nbr_psf=args.nbrpsf, init_pos_precomputed=args.initposprecomputed,PSF_shape=args.PSFshape)


def reopt_mcmc_matrix(args):
    if args.fixedparams is not None:
        fixedparams = ast.literal_eval(args.fixedparams)
    else:
        fixedparams = None
    params = Params.read(args.parameter_file)
    compute_mcmc_matrix(params, n_jobs=args.njobs, n_walkers=args.nwalkers, n_steps=args.nsteps, n_orbits=args.norbits, n_check=args.ncheck, fixed_params=fixedparams, nbr_psf=args.nbrpsf, init_pos_precomputed=args.initposprecomputed,PSF_shape=args.PSFshape)


def mcmc_starting_pos(args):
    if args.fixedparams is not None:
        fixedparams = ast.literal_eval(args.fixedparams)
    else:
        fixedparams = None
    if args.fixbounds is not None:
        fixbounds = ast.literal_eval(args.fixbounds)
    else:
        fixbounds = None
    params = Params.read(args.parameter_file)
    build_mcmc_starting_position(params, n_walkers=args.nwalkers, fixed_params=fixedparams, nbr_psf=args.nbrpsf, fix_bounds=fixbounds)

def recompute_positions(args):
    params = Params.read(args.parameter_file)
    invvar_weight = (
        bool(args.invvar_weight)
        if args.invvar_weight is not None
        else params.invvar_weight
    )
    compute_detailed_positions(
        params,
        method=args.method,
        invvar_weighted=invvar_weight,
        exclude_source=True,
        exclude_lobes=True,
        use_interp_bgnoise=False,
    )