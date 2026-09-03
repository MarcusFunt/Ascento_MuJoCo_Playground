#!/bin/sh
set -eu
exec python -m ascento_mjlab.tools.smoke "$@"
