#!/usr/bin/env bash
set -e
python3 -m pytest -q
# No remaining references to OldName anywhere in .py files.
if grep -rn 'OldName' --include='*.py' .; then
  echo "FAIL: OldName still present"
  exit 1
fi
