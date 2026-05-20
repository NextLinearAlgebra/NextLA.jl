#!/usr/bin/env python3
"""Bench METRICS fusion (plan E1).

Path **(s)** `scqr3_full25d`: use `fused_scqr3_metrics.py` (vendor FP64+FP32 + ours, one METRICS line).

Other paths: run `cusolverMp_geqrf_bench` for vendor medians, run the 1D bench, then merge
`vendor_fp*` into the ours `METRICS` line by hand or with a small awk/sed step until a
generic merger lands here.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def main() -> None:
    here = Path(__file__).resolve().parent
    fused = here / "fused_scqr3_metrics.py"
    argv = [sys.executable, str(fused), *sys.argv[1:]]
    raise SystemExit(subprocess.call(argv))


if __name__ == "__main__":
    main()
