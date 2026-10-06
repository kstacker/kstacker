import argparse

from .mcmc import run_mcmc_from_yaml
from .mcmc_init_search import search_mcmc_initial_parameters_from_yaml


def main():
    parser = argparse.ArgumentParser(
        description="K-Stacker MCMC"
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    init_parser = subparsers.add_parser(
        "init",
        help="run the initial orbital grid search",
    )
    init_parser.add_argument("parameter_file")

    mcmc_parser = subparsers.add_parser(
        "mcmc",
        help="run the MCMC",
    )
    mcmc_parser.add_argument("parameter_file")

    args = parser.parse_args()

    if args.command == "init":
        search_mcmc_initial_parameters_from_yaml(args.parameter_file)

    elif args.command == "mcmc":
        run_mcmc_from_yaml(args.parameter_file)


if __name__ == "__main__":
    main()