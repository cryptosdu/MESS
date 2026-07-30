#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "$PACKAGE_DIR/run.py" \
  --dataset sift \
  --profile standard \
  --mode attack \
  --attack-audit-angular-radius 0.10 \
  --reconstruction-xdp-angular-radii 0.05 0.10 \
  --attack-known-anchors 1 \
  --attack-exposure-tail-probabilities 0.05 \
  "$@"
