#!/usr/bin/env python3
"""Stable entry point for the MESS XDP experiment package."""

from __future__ import annotations

from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
ENGINE = ROOT / "engine"
if str(ENGINE) not in sys.path:
    sys.path.insert(0, str(ENGINE))

from run_experiments import main  # noqa: E402


if __name__ == "__main__":
    raise SystemExit(main())

