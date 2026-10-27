"""Command-line entry points for the K-Stacker workflow."""
import argparse

from .mcmc import run_mcmc_from_yaml
from .mcmc_init_search import search_mcmc_initial_parameters_from_yaml
from .scientific_plots import create_figures_from_yaml
from .console import _print_section, console_status, console_success, console_error


def main():
    """Dispatch the requested workflow stage from the command line."""
    parser = argparse.ArgumentParser(description="K-Stacker orbit inference")

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser("init", help="run the initial orbital grid search")
    init_parser.add_argument("parameter_file")

    mcmc_parser = subparsers.add_parser("mcmc", help="run the MCMC posterior sampling")
    mcmc_parser.add_argument("parameter_file")

    plot_parser = subparsers.add_parser(
        "plot", help="create scientific figures and sampler diagnostics from a saved chain"
    )
    plot_parser.add_argument("parameter_file")

    args = parser.parse_args()
    try:
        if args.command == "init":
            _print_section("Initial orbit search", tone="blue")
            console_status("config", f"YAML: {args.parameter_file}", tone="muted")
            search_mcmc_initial_parameters_from_yaml(args.parameter_file)
            console_success("Initial search finished.")

        elif args.command == "mcmc":
            _print_section("Posterior sampling", tone="magenta")
            console_status("config", f"YAML: {args.parameter_file}", tone="muted")
            run_mcmc_from_yaml(args.parameter_file)
            console_success("Posterior sampling finished.")

        elif args.command == "plot":
            _print_section("Scientific figures and diagnostics", tone="green")
            console_status("config", f"YAML: {args.parameter_file}", tone="muted")
            create_figures_from_yaml(args.parameter_file)
            console_success("Figures and diagnostics written to disk.")
    except Exception as exc:  # pragma: no cover - user-facing safety net
        console_error(str(exc))
        raise


if __name__ == "__main__":
    main()
