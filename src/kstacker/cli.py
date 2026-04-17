import argparse
import sys
import time
import ast
import os 

import numpy as np

from .gradient_reoptimization import compute_detailed_positions, reoptimize_gradient
from .inject_planet_in_data import inject_planet
from .mcmc_reoptimization import reoptimize_mcmc
from .mcmc_starting_pos import build_mcmc_starting_position
from .run_matrix_mcmc import compute_mcmc_matrix
from .noise_profile import compute_noise_profiles, compute_snr_plots, compute_mcmc_noise_signal_profil
from .optimize import brute_force, extract_best_solutions
from .utils import Params
from .version import version
from .mcmc import run_mcmc_from_yaml

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
     # mcmc parser
    sub_mcmc = subparsers.add_parser(
        "mcmc",
        help=(
            "Run the vectorized, YAML-driven emcee sampler. "
            "All options (t_ref_mode/t_ref, priors, weighting, MCMC, parallel, plots) "
            "are read from the YAML file."
        ),
        description=(
            "Execute the YAML-configured MCMC driver.\n\n"
            "The YAML should define, at minimum:\n"
            "  - t_ref_mode (\"min_ts\" or \"fixed\") and t_ref if fixed,\n"
            "  - priors (a_bounds, m0_bounds, la0_bounds, ecc_prior, e_max, ...),\n"
            "  - init / init_spread for walker initialization,\n"
            "  - global controls (weighting, snr_scale, noise_floor),\n"
            "  - mcmc (nwalkers, burnin, nsteps, thin, progress),\n"
            "  - parallel (max_workers, chunk_size),\n"
            "  - plots configuration."
        ),
    )
    sub_mcmc.add_argument(
        "parameter_file",
        metavar="PARAMS_YAML",
        help="Path to the YAML parameter file (see mcmc.py docstrings for schema).",
    )
    sub_mcmc.set_defaults(func=run_mcmc_cli)
    
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
    # inject planet parser
    sub_prof = subparsers.add_parser("inject_planet", help="inject a planet into a data set")
    sub_prof.add_argument("parameter_file", help="Parameter file (yml)")
    sub_prof.set_defaults(func=inject_planet_in_data)

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

def str2bool(s):
    return {"true": True, "false": False}[s.lower()]


def noise_profiles(args):
    if args.seed:
        np.random.seed(args.seed)
    params = Params.read(args.parameter_file)
    if params.noise_prof == "yes":
        compute_noise_profiles(params)
    if params.snr_plot == "yes":
        compute_snr_plots(params)
    compute_mcmc_noise_signal_profil(params)


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

def run_mcmc_cli(args):
    """
    Execute the complete YAML-configured MCMC workflow as a CLI entry point.

    This function is meant to be called by an argparse subcommand that provides
    one argument: `args.parameter_file`, pointing to a YAML configuration file.

    The YAML file controls:
      • Data loading (image cubes, radial profiles, timestamps)
      • Prior bounds and orbital parameterization
      • Walker initialization (initial guess + spread)
      • MCMC sampling settings (walkers, burn-in, nsteps, thin, etc.)
      • Multiprocessing settings
      • Plotting options (corner plot, posterior histograms, coadded native stack, etc.)

    This CLI convenience layer performs *no inference logic itself*.  Instead:
      1. It loads the YAML using `Params.read(...)` (which also resolves paths).
      2. It calls the main workflow `run_mcmc_from_yaml(...)`.
      3. It saves essential sampling artifacts to disk so that the user can:
         - Reload chains later,
         - Inspect log-probabilities,
         - Validate acceptance rate,
         - Run new plots without re-sampling.

    Parameters
    ----------
    args : argparse.Namespace
        Must contain:
            args.parameter_file : str
                Path to the YAML configuration file.

    Side Effects
    ------------
    - Creates the directory specified by `values_dir` (from YAML).
    - Saves:
        `mcmc_flat_chain.npy`   → flattened posterior samples (after thinning).
        `mcmc_log_prob.npy`     → corresponding log-probability values.
        `mcmc_acceptance.txt`   → acceptance statistics and shapes.
    - If the YAML enables plotting, figures will already have been saved by
      `run_mcmc_from_yaml`.

    Prints
    ------
    A short message stating where results were saved.
    """

    # Parse YAML and construct the Params object.  This also resolves paths such as:
    #   values_dir, images_dir, plots_dir, etc., based on the YAML contents.
    params = Params.read(args.parameter_file)

    # Directory where chain + metadata will be stored.
    values_dir = params.get_path("values_dir")
    os.makedirs(values_dir, exist_ok=True)

    # Run the full sampling pipeline:
    #   - Read priors & reference epoch configuration.
    #   - Construct vectorized likelihood/log-posterior.
    #   - Initialize walkers.
    #   - Run emcee (burn-in + production).
    #   - Run plot suite if enabled in YAML.
    sampler, flat = run_mcmc_from_yaml(args.parameter_file)

    # Save the flattened chain (already thinned inside sampler.get_chain).
    # Shape is (Nsamples, 7) for parameters [a, la0, m0, h, k, p, q].
    np.save(os.path.join(values_dir, "mcmc_flat_chain.npy"), flat)

    # Also save the corresponding log-probabilities.  We compute them using
    # the *same thinning factor* the user requested in the YAML.
    thin = int(params._params.get("mcmc", {}).get("thin", 100))
    log_prob = sampler.get_log_prob(discard=0, thin=thin, flat=True)
    np.save(os.path.join(values_dir, "mcmc_log_prob.npy"), log_prob)

    # Compute and save basic acceptance diagnostics.  A healthy sampler
    # typically has mean acceptance fraction between ~0.2 and ~0.5.
    acc_frac = float(np.mean(sampler.acceptance_fraction))
    with open(os.path.join(values_dir, "mcmc_acceptance.txt"), "w") as f:
        f.write(f"mean_acceptance_fraction: {acc_frac:.6f}\n")
        f.write(f"flat_chain_shape: {flat.shape}\n")
        f.write(f"log_prob_shape: {log_prob.shape}\n")

    # Final user-friendly message.
    print(f"[mcmc] Saved sampling outputs to: {values_dir}")
    print(f"[mcmc] Saved artifacts in: {values_dir}")


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


def inject_planet_in_data(args):
    params = Params.read(args.parameter_file)
    inject_planet(params)