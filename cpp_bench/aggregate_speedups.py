#!/usr/bin/env python3
"""Compute geometric-mean and minimum speedup from paired vendor/ours times.

Reads stdin: one pair per line (whitespace-separated):
    T_vendor_ms  T_ours_ms

Speedup r = T_vendor / T_ours (larger is better for "our" code).

Usage:
  paste vendor.txt ours.txt | awk '{print $1,$2}' | python3 aggregate_speedups.py
  echo -e "100 80\\n200 90" | python3 aggregate_speedups.py
"""
from __future__ import annotations

import math
import sys


def main() -> None:
    ratios: list[float] = []
    for line in sys.stdin:
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 2:
            continue
        tv, to = float(parts[0]), float(parts[1])
        if to <= 0 or tv < 0:
            sys.stderr.write(f"skip invalid line: {line}\n")
            continue
        ratios.append(tv / to)
    if not ratios:
        print("geom_mean=nan min_speedup=nan n=0", file=sys.stdout)
        sys.exit(1)
    geom = math.exp(sum(math.log(r) for r in ratios) / len(ratios))
    mn = min(ratios)
    print(f"geom_mean={geom:.6f} min_speedup={mn:.6f} n={len(ratios)}")


if __name__ == "__main__":
    main()
