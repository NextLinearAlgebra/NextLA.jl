#!/usr/bin/env python3
"""Parse METRICS lines from cpp_bench stdout / log files.

Supports:
  * 3D grid (Path s): Px, Py, Pz, optional layout=
  * 1D partition (h, g, q): c= instead of Px,Py,Pz

Options:
  --json       print JSON array of dicts
  --aggregate  after parsing, print geom_mean and min_speedup per (bench, matrix, grid_key)
  --strict-perf FLOOR   exit 2 if any per-row ours_ms/vendor ratio < FLOOR
  --strict-perf-geom FLOOR  exit 3 if any aggregate group has geom_mean speedup < FLOOR (needs numeric vendor)
  --strict-perf-min FLOOR   exit 4 if any aggregate group has min_speedup < FLOOR
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
from collections import defaultdict

METRICS_3D_RE = re.compile(
    r"^METRICS\s+"
    r"bench=(?P<bench>\S+)\s+"
    r"matrix=(?P<matrix>\S+)\s+"
    r"(?:layout=(?P<layout>\S+)\s+)?"
    r"N=(?P<N>\d+)\s+"
    r"b=(?P<b>\d+)\s+"
    r"Px=(?P<Px>\d+)\s+"
    r"Py=(?P<Py>\d+)\s+"
    r"Pz=(?P<Pz>\d+)\s+"
    r"passes=(?P<passes>\d+)\s+"
    r"vendor_fp64_ms=(?P<vf64>[^ ]+)\s+"
    r"vendor_fp32_ms=(?P<vf32>[^ ]+)\s+"
    r"ours_ms=(?P<ours>[-+eE0-9.]+)"
)

METRICS_1D_RE = re.compile(
    r"^METRICS\s+"
    r"bench=(?P<bench>\S+)\s+"
    r"matrix=(?P<matrix>\S+)\s+"
    r"N=(?P<N>\d+)\s+"
    r"b=(?P<b>\d+)\s+"
    r"c=(?P<c>\d+)\s+"
    r"passes=(?P<passes>\d+)\s+"
    r"vendor_fp64_ms=(?P<vf64>[^ ]+)\s+"
    r"vendor_fp32_ms=(?P<vf32>[^ ]+)\s+"
    r"ours_ms=(?P<ours>[-+eE0-9.]+)"
)


def _parse_metrics_line(line: str) -> dict | None:
    s = line.strip()
    m = METRICS_3D_RE.search(s)
    if m:
        d = m.groupdict()
        d["grid_kind"] = "3d"
        if "c" not in d or d["c"] is None:
            d["c"] = None
        return d
    m = METRICS_1D_RE.search(s)
    if m:
        d = m.groupdict()
        d["grid_kind"] = "1d"
        d["layout"] = d.get("layout") or "1d"
        d["Px"] = None
        d["Py"] = None
        d["Pz"] = None
        return d
    return None


def _normalize_row(d: dict) -> dict:
    out = dict(d)
    out["N"] = int(out["N"])
    out["b"] = int(out["b"])
    out["passes"] = int(out["passes"])
    out["ours_ms"] = float(out["ours"])
    if out.get("Px") is not None:
        out["Px"] = int(out["Px"])
        out["Py"] = int(out["Py"])
        out["Pz"] = int(out["Pz"])
    if out.get("c") is not None:
        out["c"] = int(out["c"])
    return out


def _vendor_for_matrix(matrix: str) -> str:
    if matrix == "fp32full":
        return "vf32"
    return "vf64"


def _speedup(row: dict) -> float | None:
    col = _vendor_for_matrix(row["matrix"])
    v = row.get(col, "NA")
    try:
        tv = float(v)
    except (TypeError, ValueError):
        return None
    ours = row["ours_ms"]
    if tv <= 0 or ours <= 0:
        return None
    return tv / ours


def _grid_key(r: dict) -> str:
    if r.get("grid_kind") == "1d":
        return f"1d:c={r['c']}"
    lay = r.get("layout") or "slab"
    return f"3d:{lay}:Px={r['Px']}:Py={r['Py']}:Pz={r['Pz']}"


def parse_log_text(text: str) -> list[dict]:
    rows: list[dict] = []
    for line in text.splitlines():
        d = _parse_metrics_line(line)
        if d:
            rows.append(_normalize_row(d))
    return rows


def aggregate_groups(rows: list[dict]) -> dict[tuple[str, str, str], tuple[float, float, int]]:
    """Return map key -> (geom_mean, min_speedup, n)."""
    groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for r in rows:
        sp = _speedup(r)
        if sp is None:
            continue
        key = (r["bench"], r["matrix"], _grid_key(r))
        groups[key].append(sp)
    out: dict[tuple[str, str, str], tuple[float, float, int]] = {}
    for key, ratios in groups.items():
        geom = math.exp(sum(math.log(x) for x in ratios) / len(ratios))
        out[key] = (geom, min(ratios), len(ratios))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log", nargs="?", help="log file (default: stdin)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--aggregate", action="store_true", help="geom_mean and min_speedup per bench×matrix×grid")
    ap.add_argument(
        "--strict-perf",
        type=float,
        default=None,
        help="min speedup vs vendor (fp64 vendor for fp64/fp64mp/fp64mp_tf32; fp32 vendor for fp32full)",
    )
    ap.add_argument(
        "--strict-perf-geom",
        type=float,
        default=None,
        help="require every aggregate group's geometric-mean speedup >= FLOOR (exit 3)",
    )
    ap.add_argument(
        "--strict-perf-min",
        type=float,
        default=None,
        help="require every aggregate group's minimum speedup >= FLOOR (exit 4)",
    )
    args = ap.parse_args()

    if args.log is None:
        text = sys.stdin.read()
    else:
        with open(args.log, encoding="utf-8", errors="replace") as f:
            text = f.read()

    rows = parse_log_text(text)

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    if not rows:
        print("no METRICS lines found", file=sys.stderr)
        sys.exit(1)

    ag = aggregate_groups(rows)

    if args.strict_perf is not None:
        for r in rows:
            sp = _speedup(r)
            if sp is None:
                continue
            if sp < args.strict_perf:
                print(f"strict-perf fail: speedup {sp:.4f} < {args.strict_perf} row={r}", file=sys.stderr)
                sys.exit(2)
        print("strict-perf: OK")

    if args.strict_perf_geom is not None or args.strict_perf_min is not None:
        if not ag:
            print("strict-perf-geom/min: no aggregate groups with numeric vendor", file=sys.stderr)
            sys.exit(1)
        for key, (geom, mn, n) in sorted(ag.items()):
            if args.strict_perf_geom is not None and geom < args.strict_perf_geom:
                print(
                    f"strict-perf-geom fail: geom_mean={geom:.6f} < {args.strict_perf_geom} group={key} n={n}",
                    file=sys.stderr,
                )
                sys.exit(3)
            if args.strict_perf_min is not None and mn < args.strict_perf_min:
                print(
                    f"strict-perf-min fail: min_speedup={mn:.6f} < {args.strict_perf_min} group={key} n={n}",
                    file=sys.stderr,
                )
                sys.exit(4)
        if args.strict_perf_geom is not None:
            print(f"strict-perf-geom: OK (>= {args.strict_perf_geom})")
        if args.strict_perf_min is not None:
            print(f"strict-perf-min: OK (>= {args.strict_perf_min})")

    if args.aggregate:
        if not ag:
            print("aggregate: no rows with numeric vendor and positive ours_ms", file=sys.stderr)
            sys.exit(1)
        for (bench, matrix, gk), (geom, mn, n) in sorted(ag.items()):
            print(f"bench={bench} matrix={matrix} grid={gk} geom_mean={geom:.6f} min_speedup={mn:.6f} n={n}")
    elif (
        args.strict_perf is None
        and args.strict_perf_geom is None
        and args.strict_perf_min is None
    ):
        print(f"parsed {len(rows)} METRICS rows")


if __name__ == "__main__":
    main()
