#!/usr/bin/env python3
"""One process: run cuSOLVERMp (FP64 + FP32) and scqr3_full25d, emit one METRICS line.

Parses vendor lines:
    cusolverMpGeqrf  N=...  ...  tmed=...
and ours:
    METRICS bench=scqr3_full25d ...
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys

VENDOR_TMED = re.compile(r"tmed\s*=\s*([0-9.]+)")
OURS_METRICS = re.compile(
    r"^METRICS\s+bench=scqr3_full25d\s+matrix=(?P<matrix>\S+)\s+"
    r"(?:layout=(?P<layout>\S+)\s+)?"
    r"N=(?P<N>\d+)\s+b=(?P<b>\d+)\s+Px=(?P<Px>\d+)\s+Py=(?P<Py>\d+)\s+Pz=(?P<Pz>\d+)\s+"
    r"passes=(?P<passes>\d+)\s+vendor_fp64_ms=\S+\s+vendor_fp32_ms=\S+\s+"
    r"ours_ms=(?P<ours>[-+eE0-9.]+)"
)


def run_capture(cmd: list[str]) -> str:
    p = subprocess.run(cmd, capture_output=True, text=True)
    return p.stdout + "\n" + p.stderr


def parse_vendor_tmed(text: str) -> float | None:
    m = list(VENDOR_TMED.finditer(text))
    if not m:
        return None
    return float(m[-1].group(1))


def parse_ours_ms(text: str) -> tuple[dict[str, str], float] | None:
    for line in text.splitlines():
        m = OURS_METRICS.search(line.strip())
        if m:
            d = m.groupdict()
            return d, float(d["ours"])
    return None


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Fused METRICS: vendor fp64/fp32 from cusolverMp + ours from scqr3 (one Python process)."
    )
    ap.add_argument("--mpirun", default="mpirun", help="mpirun launcher")
    ap.add_argument("--np-vendor", type=int, required=True, help="MPI ranks for cusolverMp (Px*Py)")
    ap.add_argument("--np-ours", type=int, required=True, help="MPI ranks for scqr3_full25d")
    ap.add_argument("--vendor-bin", required=True, help="path to cusolverMp_geqrf_bench")
    ap.add_argument("--ours-bin", required=True, help="path to scqr3_full25d_bench")
    ap.add_argument("--N", type=int, required=True)
    ap.add_argument("--b-vendor", type=int, default=0, help="MB for vendor (0 = argv position default)")
    ap.add_argument("--px-vendor", type=int, required=True)
    ap.add_argument("--py-vendor", type=int, required=True)
    ap.add_argument("--ours-extra", default="", help="extra args for ours bench, quoted string split by spaces")
    ap.add_argument("--matrix", default="fp64", help="ours --matrix= value")
    args = ap.parse_args()

    Nv = str(args.N)
    mb = str(args.b_vendor) if args.b_vendor else str(args.N // 16 if args.N >= 16 else 256)
    pxv, pyv = str(args.px_vendor), str(args.py_vendor)
    vendor_base = [
        args.mpirun,
        "-np",
        str(args.np_vendor),
        args.vendor_bin,
        Nv,
        mb,
        mb,
        pxv,
        pyv,
    ]
    txt64 = run_capture(vendor_base)
    txt32 = run_capture(vendor_base + ["fp32"])
    t64 = parse_vendor_tmed(txt64)
    t32 = parse_vendor_tmed(txt32)
    if t64 is None or t32 is None:
        print("failed to parse vendor tmed from:", file=sys.stderr)
        print(txt64[-2000:], file=sys.stderr)
        print(txt32[-2000:], file=sys.stderr)
        sys.exit(2)

    extras = args.ours_extra.split() if args.ours_extra.strip() else []
    joined = " ".join(extras)
    needs_smoke = (
        "--px=" not in joined
        and "--py=" not in joined
        and "--pz=" not in joined
        and "--M=" not in joined
        and "--smoke" not in joined
    )

    ours_cmd = [args.mpirun, "-np", str(args.np_ours), args.ours_bin, f"--N={args.N}", f"--matrix={args.matrix}"]
    if needs_smoke:
        ours_cmd.append("--smoke")
    ours_cmd += extras
    txt_ours = run_capture(ours_cmd)
    parsed = parse_ours_ms(txt_ours)
    if not parsed:
        print("failed to parse METRICS from ours run:", file=sys.stderr)
        print(txt_ours[-4000:], file=sys.stderr)
        sys.exit(3)
    row, ours_ms = parsed

    vf64 = f"{t64:.4f}"
    vf32 = f"{t32:.4f}"
    layout = row.get("layout") or "slab"
    line = (
        f"METRICS bench=scqr3_full25d matrix={row['matrix']} layout={layout} "
        f"N={row['N']} b={row['b']} Px={row['Px']} Py={row['Py']} Pz={row['Pz']} "
        f"passes={row['passes']} vendor_fp64_ms={vf64} vendor_fp32_ms={vf32} ours_ms={ours_ms:.4f}"
    )
    print(line)


if __name__ == "__main__":
    main()
