#!/usr/bin/env bash
set -euo pipefail
PACKAGE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
python3 "$PACKAGE_DIR/run.py" \
  --dataset all \
  --profile standard \
  --mode all \
  "$@"

