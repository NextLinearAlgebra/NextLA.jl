#!/usr/bin/env python3
"""GPU-driven N_max probe: grow N until the benchmark exits non-zero or stderr hints OOM.

Requires a working mpirun + GPU environment. Example:

  python3 n_max_gpu_probe.py \\
    --mpirun "mpirun --oversubscribe -np 4" \\
    --binary ./scqr3_full25d_bench \\
    --args-template "--N={N} --smoke --passes=1 --strict-b --no-la" \\
    --n-start 8000 --n-step 8000 --n-max 200000

Prints the largest N for which the command succeeded (exit code 0).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mpirun", default="mpirun", help="launcher with flags, e.g. mpirun -np 4")
    ap.add_argument("--binary", required=True)
    ap.add_argument(
        "--args-template",
        required=True,
        help="argv template; must contain {N} placeholder",
    )
    ap.add_argument("--n-start", type=int, required=True)
    ap.add_argument("--n-step", type=int, default=0, help="if 0, use geometric *2 steps from n-start")
    ap.add_argument("--n-max", type=int, default=10**9)
    ap.add_argument("--max-tries", type=int, default=32)
    args = ap.parse_args()
    if "{N}" not in args.args_template:
        print("args-template must contain {N}", file=sys.stderr)
        sys.exit(2)

    oom_re = re.compile(r"out of memory|OOM|cudaMalloc|CUBLAS_STATUS_ALLOC_FAILED", re.I)

    def run_n(n: int) -> tuple[bool, str]:
        argv = args.mpirun.split() + [args.binary] + args.args_template.format(N=n).split()
        p = subprocess.run(argv, capture_output=True, text=True)
        out = (p.stdout or "") + "\n" + (p.stderr or "")
        ok = p.returncode == 0 and not oom_re.search(out)
        return ok, out

    last_good = None
    n = args.n_start
    tries = 0
    while n <= args.n_max and tries < args.max_tries:
        tries += 1
        ok, out = run_n(n)
        if ok:
            last_good = n
            if args.n_step > 0:
                n += args.n_step
            else:
                n = max(n + 1, int(n * 1.25)) if n > args.n_start else n * 2
        else:
            break

    if last_good is None:
        print("n_max_probe: no successful N in range", file=sys.stderr)
        sys.exit(1)
    print(f"N_MAX_GOOD={last_good}")


if __name__ == "__main__":
    main()
