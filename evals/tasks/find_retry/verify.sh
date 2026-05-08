#!/usr/bin/env bash
set -e
python3 -m pytest -q
# Verify the parameter exists and has the default 5.
python3 - <<'PY'
import inspect
from network import sleep_and_redo
sig = inspect.signature(sleep_and_redo)
assert "max_attempts" in sig.parameters, "missing max_attempts parameter"
assert sig.parameters["max_attempts"].default == 5, "default must be 5"
PY
