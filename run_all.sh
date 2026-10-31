#!/usr/bin/env bash
# Reproduce every number and figure in the README.
#   python 3.11+ with the packages in pyproject.toml
#   cmake and a C compiler for the host tests
#   arm-none-eabi-gcc and qemu-system-arm for the Cortex-M4 count
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python3}"

"$PY" -m pytest -q
"$PY" scripts/export_vectors.py
cmake -S c -B c/build -DCMAKE_BUILD_TYPE=Release > /dev/null
cmake --build c/build -j > /dev/null
ctest --test-dir c/build --output-on-failure
"$PY" scripts/run_benchmark.py
"$PY" scripts/make_figures.py
"$PY" scripts/count_instructions.py
