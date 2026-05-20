#!/usr/bin/env python3
"""Post-process sweep logs: read METRICS JSON from stdin, print aggregate speedups.

Typical use (after parse_bench_log --json):
  python3 parse_bench_log.py sweep.log --json | python3 postprocess_sweep.py

Or pipe a file:
  python3 parse_bench_log.py sweep.log --json | python3 postprocess_sweep.py --json-out summary.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict


def main() -> None:
    ap = argparse.ArgumentParser(description="Aggregate speedups from parse_bench_log.py --json on stdin")
    ap.add_argument("--json-out", help="write full aggregate structure to this path")
    ap.add_argument(
        "--require-geom",
        type=float,
        default=None,
        help="exit 5 if any aggregate row has geom_mean_speedup below this (numeric vendor rows only)",
    )
    ap.add_argument(
        "--require-min",
        type=float,
        default=None,
        help="exit 6 if any aggregate row has min_speedup below this",
    )
    ap.add_argument(
        "--gate-scqr3-fp64-min-geom",
        type=float,
        default=None,
        help="exit 5 if scqr3_full25d row with matrix fp64 has geom_mean below this",
    )
    ap.add_argument(
        "--gate-hgq-min-geom",
        type=float,
        default=None,
        help="exit 5 if householder/givens/qdwh bench has geom_mean below this (any matrix)",
    )
    args = ap.parse_args()
    try:
        rows = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print("stdin: expected JSON array from parse_bench_log.py --json", file=sys.stderr)
        print(e, file=sys.stderr)
        sys.exit(1)
    if not isinstance(rows, list):
        print("stdin: JSON must be an array", file=sys.stderr)
        sys.exit(1)

    def vendor_col(matrix: str) -> str:
        return "vf32" if matrix == "fp32full" else "vf64"

    def grid_key(r: dict) -> str:
        if r.get("grid_kind") == "1d" or r.get("c") is not None:
            return f"1d:c={r.get('c')}"
        lay = r.get("layout") or "slab"
        return f"3d:{lay}:Px={r['Px']}:Py={r['Py']}:Pz={r['Pz']}"

    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        col = vendor_col(r.get("matrix", ""))
        try:
            tv = float(r[col])
            ours = float(r["ours_ms"])
        except (KeyError, TypeError, ValueError):
            continue
        if tv <= 0 or ours <= 0:
            continue
        key = (r.get("bench", "?"), r.get("matrix", "?"), grid_key(r))
        groups[key].append(tv / ours)

    out_obj = []
    for (bench, matrix, gk), ratios in sorted(groups.items()):
        geom = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
        mn = min(ratios)
        rec = {
            "bench": bench,
            "matrix": matrix,
            "grid": gk,
            "geom_mean_speedup": geom,
            "min_speedup": mn,
            "n": len(ratios),
        }
        out_obj.append(rec)
        print(
            f"bench={bench} matrix={matrix} grid={gk} geom_mean={geom:.6f} min_speedup={mn:.6f} n={len(ratios)}"
        )

    if args.json_out:
        with open(args.json_out, "w", encoding="utf-8") as f:
            json.dump(out_obj, f, indent=2)

    if not out_obj:
        sys.exit(1)

    for rec in out_obj:
        g = rec["geom_mean_speedup"]
        m = rec["min_speedup"]
        bench = rec["bench"]
        matrix = rec["matrix"]
        if args.require_geom is not None and g < args.require_geom:
            print(f"gate fail require-geom: {rec}", file=sys.stderr)
            sys.exit(5)
        if args.require_min is not None and m < args.require_min:
            print(f"gate fail require-min: {rec}", file=sys.stderr)
            sys.exit(6)
        if args.gate_scqr3_fp64_min_geom is not None:
            if bench == "scqr3_full25d" and matrix == "fp64" and g < args.gate_scqr3_fp64_min_geom:
                print(f"gate fail scqr3 fp64 geom: {rec}", file=sys.stderr)
                sys.exit(5)
        if args.gate_hgq_min_geom is not None:
            if bench in ("householder_2p5d", "givens_2p5d", "qdwh_2p5d") and g < args.gate_hgq_min_geom:
                print(f"gate fail hgq geom: {rec}", file=sys.stderr)
                sys.exit(5)


if __name__ == "__main__":
    main()
