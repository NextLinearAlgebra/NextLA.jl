#!/usr/bin/env python3
"""Fill EXECUTIVE_SUMMARY dashboard placeholders from postprocess JSON (optional).

Reads JSON from postprocess_sweep.py (--json-out) or stdin; prints a markdown table row
per aggregate record. Does not patch files in-place by default.

  python3 postprocess_sweep.py < summary.json  # if you saved METRICS list
  # Typical: merge logs then:
  python3 parse_bench_log.py combined.log --json | python3 postprocess_sweep.py --json-out agg.json
  python3 gen_executive_dashboard.py agg.json
"""
from __future__ import annotations

import argparse
import json
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("json", nargs="?", help="aggregate JSON from postprocess_sweep.py; default stdin")
    args = ap.parse_args()
    if args.json:
        with open(args.json, encoding="utf-8") as f:
            data = json.load(f)
    else:
        data = json.load(sys.stdin)
    if not isinstance(data, list):
        print("expected JSON array", file=sys.stderr)
        sys.exit(1)
    print("| bench | matrix | grid | geom_mean ↑ | min ↑ | n |")
    print("| --- | --- | --- | --- | --- | --- |")
    for r in data:
        print(
            f"| {r.get('bench','')} | {r.get('matrix','')} | {r.get('grid','')} | "
            f"{r.get('geom_mean_speedup', 0):.4f} | {r.get('min_speedup', 0):.4f} | {r.get('n', 0)} |"
        )


if __name__ == "__main__":
    main()
