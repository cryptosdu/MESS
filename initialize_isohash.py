#!/usr/bin/env python3
"""Initialize IsoHash weights for all MESS datasets."""

from pathlib import Path
import sys


LSH_DIRECTORY = Path(__file__).resolve().parent / "src" / "util" / "LSH"
if str(LSH_DIRECTORY) not in sys.path:
    sys.path.insert(0, str(LSH_DIRECTORY))

from train_isohash import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main(default_datasets=("all",)))
