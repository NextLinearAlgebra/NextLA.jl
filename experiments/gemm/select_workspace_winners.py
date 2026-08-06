#!/usr/bin/env python3
"""Select the lowest-median workspace candidate for every complete GEMM case."""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path


GROUP_COLUMNS = (
    "N",
    "precision",
    "storage_type",
    "compute_mode",
    "operand_layout",
    "tile_divisor",
    "tile_size",
    "distribution",
    "rank_band",
    "min_rank",
    "max_rank",
    "execution_rank_policy",
    "workspace_policy",
    "seed",
    "factor_fill",
)

REQUIRED_COLUMNS = {
    "experiment",
    "record_kind",
    "case_id",
    "baseline_case_id",
    "run_id",
    "numeric_median_ms",
    "workspace_bytes",
    "workspace_parameter",
    "memory_ratio",
    *GROUP_COLUMNS,
}


def unused_path(path: Path, *, explicit: bool) -> Path:
    path = path.resolve()
    if explicit and path.exists():
        raise ValueError(f"refusing to overwrite existing output: {path}")
    if not path.exists():
        return path
    stem, suffix = path.stem, path.suffix
    index = 1
    while True:
        candidate = path.with_name(f"{stem}__{index}{suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def group_key(row: dict[str, str]) -> tuple[str, ...]:
    return tuple(row[column] for column in GROUP_COLUMNS)


def expected_levels(spec: str, all_last: int) -> set[int]:
    if spec.lower() == "all":
        return set(range(1, all_last + 1))
    levels: set[int] = set()
    for token in spec.split(","):
        pieces = token.strip().split(":")
        if len(pieces) == 1:
            levels.add(int(pieces[0]))
        elif len(pieces) == 2:
            first, last = map(int, pieces)
            if first > last:
                raise ValueError(f"workspace range {token!r} must be increasing")
            levels.update(range(first, last + 1))
        else:
            raise ValueError(
                f"workspace token {token!r} must be an integer or FIRST:LAST"
            )
    if not levels or min(levels) < 1:
        raise ValueError("expected workspace parameters must be positive")
    return levels


def select_winners(
    rows: list[dict[str, str]],
    max_memory_ratio: float | None,
    *,
    allow_incomplete: bool,
    expected_runs: str,
    expected_mixed_stripes: str,
) -> list[dict[str, str]]:
    baselines = [row for row in rows if row["record_kind"] == "baseline"]
    compressed = [row for row in rows if row["record_kind"] == "compressed"]
    if not compressed:
        raise ValueError("input contains no compressed candidates")

    duplicate_ids = [
        case_id
        for case_id, count in Counter(row["case_id"] for row in rows).items()
        if count > 1
    ]
    if duplicate_ids:
        raise ValueError(f"input contains duplicate case IDs: {duplicate_ids[:5]}")

    groups: dict[tuple[str, ...], list[dict[str, str]]] = {}
    for row in compressed:
        groups.setdefault(group_key(row), []).append(row)

    winners: list[dict[str, str]] = []
    for key, candidates in groups.items():
        if not allow_incomplete:
            q = int(candidates[0]["tile_divisor"])
            present = {int(row["workspace_parameter"]) for row in candidates}
            policy = candidates[0]["workspace_policy"]
            if policy == "tlr_tlr_runs":
                expected = expected_levels(expected_runs, 64)
            elif policy == "one_or_more_tile_stripes":
                expected = {
                    min(level, q)
                    for level in expected_levels(expected_mixed_stripes, q)
                }
            else:
                raise ValueError(f"unknown workspace policy {policy!r}")
            if present != expected:
                description = ", ".join(
                    f"{column}={value}" for column, value in zip(GROUP_COLUMNS, key)
                )
                raise ValueError(
                    f"incomplete workspace sweep for {description}: "
                    f"expected {sorted(expected)}, got {sorted(present)}; "
                    f"use --allow-incomplete only for an intentional subset"
                )
        eligible = candidates
        if max_memory_ratio is not None:
            eligible = [
                row
                for row in candidates
                if float(row["memory_ratio"]) <= max_memory_ratio
            ]
            if not eligible:
                description = ", ".join(
                    f"{column}={value}" for column, value in zip(GROUP_COLUMNS, key)
                )
                raise ValueError(
                    f"no workspace candidate satisfies memory ratio "
                    f"<= {max_memory_ratio:g} for {description}"
                )
        winners.append(
            min(
                eligible,
                key=lambda row: (
                    float(row["numeric_median_ms"]),
                    int(row["workspace_bytes"]),
                    row["case_id"],
                ),
            )
        )

    selected = baselines + winners
    for row in selected:
        row["experiment"] = "workspace_winners"
    selected.sort(
        key=lambda row: (
            int(row["N"]),
            row["precision"],
            row["record_kind"] != "baseline",
            row["operand_layout"],
            int(row["tile_divisor"] or 0),
            row["rank_band"],
            row["distribution"],
        )
    )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path, help="one raw workspace_tuning CSV")
    parser.add_argument(
        "--output",
        type=Path,
        help="winner CSV; an existing explicit path is never overwritten",
    )
    parser.add_argument(
        "--max-memory-ratio",
        type=float,
        help="select the fastest candidate at or below this memory ratio",
    )
    parser.add_argument(
        "--allow-incomplete",
        action="store_true",
        help="allow an intentional subset instead of checking the configured grid",
    )
    parser.add_argument(
        "--expected-runs",
        default="1,2,4,8,16,32,64",
        help="run targets expected for tlr_tlr_runs candidates",
    )
    parser.add_argument(
        "--expected-mixed-stripes",
        default="all",
        help="stripe counts expected for mixed layouts; 'all' means 1:q",
    )
    args = parser.parse_args()
    if args.max_memory_ratio is not None and args.max_memory_ratio <= 0:
        parser.error("--max-memory-ratio must be positive")

    source = args.csv.resolve()
    with source.open(newline="") as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames or []
        missing = REQUIRED_COLUMNS.difference(fieldnames)
        if missing:
            raise ValueError(f"{source}: missing columns {sorted(missing)}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"{source}: no rows")
    experiments = {row["experiment"] for row in rows}
    if experiments != {"workspace_tuning"}:
        raise ValueError(
            f"{source}: expected experiment workspace_tuning, got {sorted(experiments)}"
        )

    requested = args.output
    default = source.with_name(f"{source.stem}__winners.csv")
    destination = unused_path(requested or default, explicit=requested is not None)
    winners = select_winners(
        rows,
        args.max_memory_ratio,
        allow_incomplete=args.allow_incomplete,
        expected_runs=args.expected_runs,
        expected_mixed_stripes=args.expected_mixed_stripes,
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("x", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(winners)

    compressed_count = sum(row["record_kind"] == "compressed" for row in winners)
    print(f"Selected {compressed_count} winners from {source}")
    print(f"Output: {destination}")


if __name__ == "__main__":
    main()
