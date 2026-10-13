#!/usr/bin/env bash
# Execute one requested stage with unbuffered, tail-friendly console output.
set -euo pipefail
if (( $# < 1 || $# > 2 )); then
    echo 'Usage: run_orbit_inference.sh PARAMETER_FILE [init|mcmc|plot|all]' >&2
    exit 2
fi
CONFIG=$1
STAGE=${2:-all}
PACKAGE_PARENT=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)
export PYTHONPATH="${PACKAGE_PARENT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONUNBUFFERED=1
export KSTACKER_PROGRESS=${KSTACKER_PROGRESS:-log}
export KSTACKER_COLOR=${KSTACKER_COLOR:-never}
case "$STAGE" in
  init|mcmc|plot)
    python -m kstacker.cli "$STAGE" "$CONFIG" ;;
  all)
    python -m kstacker.cli init "$CONFIG"
    python -m kstacker.cli mcmc "$CONFIG" ;;
  *)
    echo "Unknown stage: $STAGE" >&2
    exit 2 ;;
esac
