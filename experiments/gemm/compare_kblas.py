#!/usr/bin/env python3
"""Join confirmed NextLA fixed-rank cases with matching KBLAS measurements."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


OUTPUT_COLUMNS = (
    "N",
    "q",
    "b",
    "rank",
    "rank_over_b",
    "nextla_median_ms",
    "nextla_min_ms",
    "kblas_median_ms",
    "kblas_min_ms",
    "kblas_time_over_nextla",
    "nextla_dense_median_ms",
    "nextla_speedup_over_dense",
    "nextla_workspace_parameter",
    "nextla_workspace_bytes",
    "nextla_memory_ratio",
    "kblas_workspace_bytes",
    "kblas_memory_ratio",
    "nextla_flop_ratio_ceiling",
    "kblas_flop_ratio_ceiling",
    "nextla_gpu",
    "kblas_gpu",
)


def unused_path(path: Path, *, explicit: bool) -> Path:
    path = path.resolve()
    if explicit and path.exists():
        raise ValueError(f"refusing to overwrite existing output: {path}")
    if not path.exists():
        return path
    index = 1
    while True:
        candidate = path.with_name(f"{path.stem}__{index}{path.suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def nextla_key(row: dict[str, str]) -> tuple[int, int, int, int]:
    return (
        int(row["N"]),
        int(row["tile_divisor"]),
        int(row["tile_size"]),
        int(row["min_rank"]),
    )


def kblas_key(row: dict[str, str]) -> tuple[int, int, int, int]:
    return int(row["N"]), int(row["q"]), int(row["b"]), int(row["rank"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("nextla", type=Path, help="workspace_confirmation CSV")
    parser.add_argument("kblas", type=Path, help="KBLAS result CSV")
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="write the intersection instead of requiring every KBLAS row",
    )
    args = parser.parse_args()

    nextla_rows = [
        row
        for row in read_csv(args.nextla.resolve())
        if row["record_kind"] == "compressed"
        and row["operand_layout"] == "compressed_compressed"
        and row["precision"] == "fp32"
        and row["min_rank"] == row["max_rank"]
    ]
    if not nextla_rows:
        raise ValueError("NextLA input has no confirmed FP32 constant-rank TLR×TLR rows")
    experiments = {row["experiment"] for row in nextla_rows}
    if experiments != {"workspace_confirmation"}:
        raise ValueError(
            f"expected workspace_confirmation input, got {sorted(experiments)}"
        )
    nextla_by_key: dict[tuple[int, int, int, int], dict[str, str]] = {}
    for row in nextla_rows:
        key = nextla_key(row)
        if key in nextla_by_key:
            raise ValueError(f"duplicate NextLA comparison key {key}")
        nextla_by_key[key] = row

    kblas_rows = read_csv(args.kblas.resolve())
    kblas_by_key: dict[tuple[int, int, int, int], dict[str, str]] = {}
    for row in kblas_rows:
        key = kblas_key(row)
        if key in kblas_by_key:
            raise ValueError(f"duplicate KBLAS comparison key {key}")
        kblas_by_key[key] = row

    missing = sorted(set(kblas_by_key).difference(nextla_by_key))
    if missing and not args.allow_partial:
        raise ValueError(
            f"NextLA confirmation is missing {len(missing)} KBLAS cases; "
            f"first missing keys: {missing[:5]}"
        )
    keys = sorted(set(nextla_by_key).intersection(kblas_by_key))
    if not keys:
        raise ValueError("NextLA and KBLAS inputs have no matching cases")

    output_rows = []
    for key in keys:
        n, q, b, rank = key
        nextla, kblas = nextla_by_key[key], kblas_by_key[key]
        nextla_time = float(nextla["numeric_median_ms"])
        kblas_time = float(kblas["tlr_median_ms"])
        executed_flops = float(nextla["executed_flops"])
        output_rows.append(
            {
                "N": n,
                "q": q,
                "b": b,
                "rank": rank,
                "rank_over_b": rank / b,
                "nextla_median_ms": nextla_time,
                "nextla_min_ms": nextla["numeric_min_ms"],
                "kblas_median_ms": kblas_time,
                "kblas_min_ms": kblas["tlr_min_ms"],
                "kblas_time_over_nextla": kblas_time / nextla_time,
                "nextla_dense_median_ms": nextla["dense_median_ms"],
                "nextla_speedup_over_dense": nextla["speedup_median"],
                "nextla_workspace_parameter": nextla["workspace_parameter"],
                "nextla_workspace_bytes": nextla["workspace_bytes"],
                "nextla_memory_ratio": nextla["memory_ratio"],
                "kblas_workspace_bytes": kblas["workspace_bytes"],
                "kblas_memory_ratio": kblas["memory_ratio"],
                "nextla_flop_ratio_ceiling": 2.0 * n**3 / executed_flops,
                "kblas_flop_ratio_ceiling": kblas["flop_ratio_ceiling"],
                "nextla_gpu": nextla["gpu_name"],
                "kblas_gpu": kblas["gpu"],
            }
        )

    requested = args.output
    default = args.nextla.resolve().with_name(
        f"{args.nextla.resolve().stem}__vs_kblas.csv"
    )
    destination = unused_path(requested or default, explicit=requested is not None)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=OUTPUT_COLUMNS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(output_rows)
    print(f"Joined {len(output_rows)} matched cases")
    print(f"Output: {destination}")


if __name__ == "__main__":
    main()
