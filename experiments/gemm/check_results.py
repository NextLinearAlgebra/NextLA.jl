#!/usr/bin/env python3
"""Validate GEMM sweep CSV invariants before plotting or combining files."""

from __future__ import annotations

import argparse
import csv
import math
from collections import Counter
from pathlib import Path


REQUIRED = {
    "record_kind",
    "case_id",
    "baseline_case_id",
    "operand_layout",
    "precision",
    "A_storage_bytes",
    "B_storage_bytes",
    "workspace_bytes",
    "dense_reference_bytes",
    "memory_ratio",
    "numeric_median_ms",
    "dense_median_ms",
    "speedup_median",
}


def number(row: dict[str, str], name: str) -> float:
    value = row[name]
    if not value:
        raise ValueError(f"{row.get('case_id', '<unknown>')}: empty {name}")
    return float(value)


def check(path: Path) -> None:
    with path.open(newline="") as stream:
        reader = csv.DictReader(stream)
        missing = REQUIRED.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path}: missing columns {sorted(missing)}")
        rows = list(reader)

    if not rows:
        raise ValueError(f"{path}: no data rows")
    ids = [row["case_id"] for row in rows]
    duplicate_ids = [case_id for case_id, count in Counter(ids).items() if count > 1]
    if duplicate_ids:
        raise ValueError(f"{path}: duplicate case IDs {duplicate_ids[:5]}")

    baselines = {
        row["case_id"] for row in rows if row["record_kind"] == "baseline"
    }
    compressed = [row for row in rows if row["record_kind"] == "compressed"]
    for row in compressed:
        case_id = row["case_id"]
        if row["baseline_case_id"] not in baselines:
            raise ValueError(f"{path}: {case_id} has no baseline row")
        numerator = (
            number(row, "A_storage_bytes")
            + number(row, "B_storage_bytes")
            + number(row, "workspace_bytes")
        )
        denominator = number(row, "dense_reference_bytes")
        ratio = number(row, "memory_ratio")
        if not math.isclose(ratio, numerator / denominator, rel_tol=2e-12):
            raise ValueError(f"{path}: {case_id} has an inconsistent memory ratio")
        speedup = number(row, "speedup_median")
        expected = number(row, "dense_median_ms") / number(row, "numeric_median_ms")
        if not math.isclose(speedup, expected, rel_tol=2e-12):
            raise ValueError(f"{path}: {case_id} has an inconsistent speedup")

    groups = Counter(
        (row["record_kind"], row["operand_layout"], row["precision"])
        for row in rows
    )
    print(f"{path}: OK ({len(rows)} rows, {len(baselines)} baselines)")
    for key, count in sorted(groups.items()):
        print(f"  {key[0]:10s} {key[1]:23s} {key[2]:5s}: {count}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("csv", nargs="+", type=Path)
    args = parser.parse_args()
    for path in args.csv:
        check(path)


if __name__ == "__main__":
    main()

