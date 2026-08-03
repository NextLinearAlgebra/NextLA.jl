#!/usr/bin/env python3
"""Plot a joined confirmed-NextLA versus KBLAS comparison CSV."""

from __future__ import annotations

import argparse
import csv
import os
import tempfile
from pathlib import Path

_mpl_cache = Path(tempfile.gettempdir()) / "nextla-matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def unused_directory(path: Path) -> Path:
    if not path.exists():
        return path
    index = 1
    while True:
        candidate = path.with_name(f"{path.name}__{index}")
        if not candidate.exists():
            return candidate
        index += 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--formats", default="png")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    source = args.csv.resolve()
    rows = read_rows(source)
    formats = [value.strip() for value in args.formats.split(",") if value.strip()]
    unsupported = set(formats).difference({"png", "pdf", "svg"})
    if unsupported:
        parser.error(f"unsupported formats: {sorted(unsupported)}")
    default = ROOT / "experiments" / "figures" / "gemm" / source.stem
    output = unused_directory((args.output_dir or default).resolve())
    output.mkdir(parents=True)

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9.5,
            "axes.grid": True,
            "grid.alpha": 0.75,
            "figure.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )
    qs = sorted({int(row["q"]) for row in rows})
    ratios = sorted({float(row["rank_over_b"]) for row in rows})
    colors = {ratio: color for ratio, color in zip(ratios, ("#0072B2", "#D55E00", "#009E73", "#CC79A7"))}

    fig, axes = plt.subplots(1, len(qs), figsize=(5.0 * len(qs), 3.8), squeeze=False)
    for column, q in enumerate(qs):
        ax = axes[0][column]
        for ratio in ratios:
            series = [
                row for row in rows
                if int(row["q"]) == q
                and abs(float(row["rank_over_b"]) - ratio) < 1e-12
            ]
            series.sort(key=lambda row: int(row["N"]))
            if not series:
                continue
            sizes = [int(row["N"]) for row in series]
            label = f"r/b={ratio:g}"
            ax.plot(
                sizes,
                [float(row["nextla_median_ms"]) for row in series],
                color=colors[ratio], marker="o", label=f"NextLA, {label}",
            )
            ax.plot(
                sizes,
                [float(row["kblas_median_ms"]) for row in series],
                color=colors[ratio], marker="s", linestyle="--",
                label=f"KBLAS, {label}",
            )
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_title(f"q={q}, b=N/{q}")
        ax.set_xlabel("Matrix size N")
        ax.set_ylabel("Median TLR×TLR time [ms]")
        ax.legend(fontsize=8)
    fig.tight_layout()
    for extension in formats:
        fig.savefig(output / f"nextla_vs_kblas_time.{extension}", dpi=args.dpi)
    plt.close(fig)

    fig, axes = plt.subplots(1, len(qs), figsize=(5.0 * len(qs), 3.6), squeeze=False)
    for column, q in enumerate(qs):
        ax = axes[0][column]
        for ratio in ratios:
            series = [
                row for row in rows
                if int(row["q"]) == q
                and abs(float(row["rank_over_b"]) - ratio) < 1e-12
            ]
            series.sort(key=lambda row: int(row["N"]))
            if not series:
                continue
            ax.plot(
                [int(row["N"]) for row in series],
                [float(row["kblas_time_over_nextla"]) for row in series],
                color=colors[ratio], marker="o", label=f"r/b={ratio:g}",
            )
        ax.axhline(1.0, color="#555555", linestyle="--", linewidth=1.3)
        ax.set_xscale("log", base=2)
        ax.set_title(f"q={q}, b=N/{q}")
        ax.set_xlabel("Matrix size N")
        ax.set_ylabel("KBLAS time / NextLA time")
        ax.legend(fontsize=8)
    fig.tight_layout()
    for extension in formats:
        fig.savefig(output / f"nextla_vs_kblas_relative.{extension}", dpi=args.dpi)
    plt.close(fig)
    print(f"Generated {2 * len(formats)} files under {output}")


if __name__ == "__main__":
    main()
