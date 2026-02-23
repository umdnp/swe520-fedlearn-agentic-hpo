#!/usr/bin/env bash

# WARNING : Ray support on Windows is experimental and may not work as expected.
# On Windows, Flower Simulations run best in WSL2

PROJECT_DIR=~/git
VENV_DIR=~/venv

export PYTHONPATH=$PROJECT_DIR/src
export RAY_ACCEL_ENV_VAR_OVERRIDE_ON_ZERO=0
export FLWR_HOME=$PROJECT_DIR/.flwr

source ${VENV_DIR}/activate

echo "Testing Static HPO Configuration ..."
cd $PROJECT_DIR || exit

flwr run . --run-config "experiment='static_hpo'"
