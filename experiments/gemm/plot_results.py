#!/usr/bin/env python3
"""Generate exploratory figures from the dense-output GEMM sweep CSVs.

Every speedup view overlays the per-case arithmetic ceiling

    dense FLOPs / executed compressed FLOPs

as a dashed curve. ``executed_flops`` includes execution-rank padding and the
workspace-dependent fold selected by the implementation, making it the useful
ceiling for the measured algorithm rather than an exact-rank idealization.
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

# Matplotlib reads MPLCONFIGDIR during import. Use a writable cache when the
# shared home configuration directory is read-only in a batch environment.
_mpl_cache = Path(tempfile.gettempdir()) / "nextla-matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "experiments" / "results" / "gemm"
DEFAULT_FIGURES = ROOT / "experiments" / "figures" / "gemm"

LAYOUTS = ("compressed_dense", "dense_compressed", "compressed_compressed")
LAYOUT_LABEL = {
    "compressed_dense": "Compressed A × dense B",
    "dense_compressed": "Dense A × compressed B",
    "compressed_compressed": "Compressed A × compressed B",
}
PRECISIONS = ("bf16", "fp16", "tf32", "fp32")
PRECISION_LABEL = {
    "bf16": "BF16 / FP32 acc.",
    "fp16": "FP16 / FP32 acc.",
    "tf32": "TF32",
    "fp32": "FP32",
}
PRECISION_STYLE = {
    "bf16": ("#0072B2", "o"),
    "fp16": ("#009E73", "s"),
    "tf32": ("#E69F00", "D"),
    "fp32": ("#222222", "^"),
}
LAYOUT_STYLE = {
    "compressed_dense": ("#0072B2", "o"),
    "dense_compressed": ("#D55E00", "s"),
    "compressed_compressed": ("#009E73", "^"),
}
POLICY_ORDER = ("exact", "q8", "q16", "pow2")


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 9.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 8,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.5,
            "grid.color": "#D8D8D8",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.8,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path}: no rows")
    required = {
        "experiment",
        "record_kind",
        "operand_layout",
        "N",
        "tile_divisor",
        "distribution",
        "rank_band",
        "precision",
        "memory_ratio",
        "numeric_median_ms",
        "speedup_median",
        "executed_flops",
    }
    missing = required.difference(rows[0])
    if missing:
        raise ValueError(f"{path}: missing columns {sorted(missing)}")
    return [row for row in rows if row["record_kind"] == "compressed"]


def integer(row: dict[str, str], name: str) -> int:
    return int(row[name])


def number(row: dict[str, str], name: str) -> float:
    return float(row[name])


def theoretical_speedup(row: dict[str, str]) -> float:
    dense_flops = 2.0 * integer(row, "N") ** 3
    executed = number(row, "executed_flops")
    return dense_flops / executed if executed > 0 else math.nan


def ceiling_fraction(row: dict[str, str]) -> float:
    return 100.0 * number(row, "speedup_median") / theoretical_speedup(row)


def ordered(values: Iterable[str], preferred: Sequence[str]) -> list[str]:
    values = set(values)
    return [value for value in preferred if value in values] + sorted(
        values.difference(preferred)
    )


def slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_")


def subplot_grid(nrows: int, ncols: int, *, width=4.0, height=3.2):
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(width * ncols, height * nrows),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    return fig, axes


def finish_axis(ax, *, speedup=False) -> None:
    ax.grid(True, which="major")
    ax.set_axisbelow(True)
    if speedup:
        ax.axhline(1.0, color="#777777", linewidth=1.0, linestyle=":")
        ax.set_ylim(bottom=0)


def add_theory_legend(handles: list[Line2D]) -> list[Line2D]:
    return handles + [
        Line2D(
            [0], [0], color="#555555", linestyle="--", linewidth=1.8,
            label="FLOP-ratio ceiling",
        )
    ]


def save_figure(fig, outdir: Path, stem: str, formats: Sequence[str], dpi: int) -> list[Path]:
    outdir.mkdir(parents=True, exist_ok=True)
    paths = []
    for fmt in formats:
        path = outdir / f"{stem}.{fmt}"
        fig.savefig(path, dpi=dpi if fmt == "png" else None)
        paths.append(path)
    plt.close(fig)
    return paths


def matching(rows, **criteria):
    return [
        row
        for row in rows
        if all(str(row[name]) == str(value) for name, value in criteria.items())
    ]


def precision_speedup_dashboard(rows, outdir, formats, dpi):
    paths = []
    layouts = ordered((r["operand_layout"] for r in rows), LAYOUTS)
    precisions = ordered((r["precision"] for r in rows), PRECISIONS)
    divisors = sorted({integer(r, "tile_divisor") for r in rows}, reverse=True)
    for band in sorted({r["rank_band"] for r in rows}):
        for distribution in sorted({r["distribution"] for r in rows}):
            fig, axes = subplot_grid(len(divisors), len(layouts))
            for row_index, divisor in enumerate(divisors):
                for col_index, layout in enumerate(layouts):
                    ax = axes[row_index][col_index]
                    for precision in precisions:
                        series = matching(
                            rows,
                            rank_band=band,
                            distribution=distribution,
                            tile_divisor=divisor,
                            operand_layout=layout,
                            precision=precision,
                        )
                        series.sort(key=lambda r: integer(r, "N"))
                        if not series:
                            continue
                        color, marker = PRECISION_STYLE.get(
                            precision, (None, "o")
                        )
                        x = [integer(r, "N") for r in series]
                        ax.plot(
                            x,
                            [number(r, "speedup_median") for r in series],
                            color=color,
                            marker=marker,
                            label=PRECISION_LABEL.get(precision, precision),
                        )
                        ax.plot(
                            x,
                            [theoretical_speedup(r) for r in series],
                            color=color,
                            linestyle="--",
                            linewidth=1.5,
                            alpha=0.65,
                        )
                    ax.set_xscale("log", base=2)
                    ax.set_xticks(sorted({integer(r, "N") for r in rows}))
                    ax.set_xticklabels(
                        [f"{n // 1024}k" for n in sorted({integer(r, "N") for r in rows})]
                    )
                    ax.set_title(f"{LAYOUT_LABEL[layout]}\n$b=N/{divisor}$")
                    ax.set_xlabel("Matrix size N")
                    ax.set_ylabel("Speedup over dense")
                    finish_axis(ax, speedup=True)
            handles = [
                Line2D(
                    [0], [0],
                    color=PRECISION_STYLE[p][0],
                    marker=PRECISION_STYLE[p][1],
                    label=PRECISION_LABEL[p],
                )
                for p in precisions
            ]
            fig.legend(
                handles=add_theory_legend(handles),
                loc="upper center",
                ncol=min(5, len(handles) + 1),
                bbox_to_anchor=(0.5, 1.01),
            )
            fig.suptitle(
                f"Measured speedup and arithmetic ceiling — {distribution}, {band}",
                y=1.055,
                fontsize=13,
            )
            fig.tight_layout()
            paths += save_figure(
                fig,
                outdir,
                f"precision_speedup__{slug(band)}__{slug(distribution)}",
                formats,
                dpi,
            )
    return paths


def precision_metric_dashboard(rows, outdir, formats, dpi, *, metric, label, stem, logy=False):
    paths = []
    layouts = ordered((r["operand_layout"] for r in rows), LAYOUTS)
    precisions = ordered((r["precision"] for r in rows), PRECISIONS)
    divisors = sorted({integer(r, "tile_divisor") for r in rows}, reverse=True)
    for band in sorted({r["rank_band"] for r in rows}):
        for distribution in sorted({r["distribution"] for r in rows}):
            fig, axes = subplot_grid(len(divisors), len(layouts))
            for row_index, divisor in enumerate(divisors):
                for col_index, layout in enumerate(layouts):
                    ax = axes[row_index][col_index]
                    for precision in precisions:
                        series = matching(
                            rows,
                            rank_band=band,
                            distribution=distribution,
                            tile_divisor=divisor,
                            operand_layout=layout,
                            precision=precision,
                        )
                        series.sort(key=lambda r: integer(r, "N"))
                        if not series:
                            continue
                        color, marker = PRECISION_STYLE.get(precision, (None, "o"))
                        values = [metric(r) for r in series]
                        ax.plot(
                            [integer(r, "N") for r in series],
                            values,
                            color=color,
                            marker=marker,
                            label=PRECISION_LABEL.get(precision, precision),
                        )
                    ax.set_xscale("log", base=2)
                    if logy:
                        ax.set_yscale("log")
                    ax.set_title(f"{LAYOUT_LABEL[layout]}\n$b=N/{divisor}$")
                    ax.set_xlabel("Matrix size N")
                    ax.set_ylabel(label)
                    finish_axis(ax)
                    if stem == "ceiling_fraction":
                        ax.axhline(100.0, color="#555555", linestyle="--", linewidth=1.4)
                        ax.set_ylim(bottom=0)
            handles = [
                Line2D(
                    [0], [0], color=PRECISION_STYLE[p][0],
                    marker=PRECISION_STYLE[p][1], label=PRECISION_LABEL[p]
                )
                for p in precisions
            ]
            fig.legend(
                handles=handles,
                loc="upper center",
                ncol=min(4, len(handles)),
                bbox_to_anchor=(0.5, 1.01),
            )
            fig.suptitle(f"{label} — {distribution}, {band}", y=1.055, fontsize=13)
            fig.tight_layout()
            paths += save_figure(
                fig,
                outdir,
                f"precision_{stem}__{slug(band)}__{slug(distribution)}",
                formats,
                dpi,
            )
    return paths


def memory_dashboard(rows, outdir, formats, dpi, *, metric, label, stem, speedup=False):
    paths = []
    layouts = ordered((r["operand_layout"] for r in rows), LAYOUTS)
    distributions = sorted({r["distribution"] for r in rows})
    divisors = sorted({integer(r, "tile_divisor") for r in rows}, reverse=True)
    for band in sorted({r["rank_band"] for r in rows}):
        fig, axes = subplot_grid(len(distributions), len(divisors))
        for row_index, distribution in enumerate(distributions):
            for col_index, divisor in enumerate(divisors):
                ax = axes[row_index][col_index]
                for layout in layouts:
                    series = matching(
                        rows,
                        rank_band=band,
                        distribution=distribution,
                        tile_divisor=divisor,
                        operand_layout=layout,
                    )
                    series.sort(key=lambda r: number(r, "memory_ratio"))
                    if not series:
                        continue
                    color, marker = LAYOUT_STYLE[layout]
                    x = [number(r, "memory_ratio") for r in series]
                    ax.plot(
                        x,
                        [metric(r) for r in series],
                        color=color,
                        marker=marker,
                        label=LAYOUT_LABEL[layout],
                    )
                    if speedup:
                        ax.plot(
                            x,
                            [theoretical_speedup(r) for r in series],
                            color=color,
                            linestyle="--",
                            linewidth=1.5,
                            alpha=0.65,
                        )
                ax.set_title(f"{distribution.capitalize()}, $b=N/{divisor}$")
                ax.set_xlabel("(operand storage + workspace) / dense A+B")
                ax.set_xscale("log")
                ax.set_ylabel(label)
                finish_axis(ax, speedup=speedup)
                if stem == "ceiling_fraction":
                    ax.axhline(100.0, color="#555555", linestyle="--", linewidth=1.4)
                    ax.set_ylim(bottom=0)
        handles = [
            Line2D(
                [0], [0], color=LAYOUT_STYLE[l][0], marker=LAYOUT_STYLE[l][1],
                label=LAYOUT_LABEL[l]
            )
            for l in layouts
        ]
        if speedup:
            handles = add_theory_legend(handles)
        fig.legend(
            handles=handles,
            loc="upper center",
            ncol=min(4, len(handles)),
            bbox_to_anchor=(0.5, 1.01),
        )
        fig.suptitle(f"{label} versus memory ratio — {band}", y=1.055, fontsize=13)
        fig.tight_layout()
        paths += save_figure(
            fig, outdir, f"memory_{stem}__{slug(band)}", formats, dpi
        )
    return paths


def ablation_figures(rows, outdir, formats, dpi):
    paths = []
    distributions = sorted({r["distribution"] for r in rows})
    for band in sorted({r["rank_band"] for r in rows}):
        subset = [r for r in rows if r["rank_band"] == band]
        policies = ordered(
            (r["execution_rank_policy"] for r in subset), POLICY_ORDER
        )
        for metric, label, stem, speedup in (
            (lambda r: number(r, "speedup_median"), "Speedup over dense", "speedup", True),
            (lambda r: number(r, "numeric_median_ms"), "Median numerical time [ms]", "time", False),
            (lambda r: number(r, "memory_ratio"), "Memory ratio", "memory", False),
        ):
            fig, axes = subplot_grid(1, len(distributions), width=4.4, height=3.6)
            for col_index, distribution in enumerate(distributions):
                ax = axes[0][col_index]
                series = matching(subset, distribution=distribution)
                by_policy = {r["execution_rank_policy"]: r for r in series}
                present = [p for p in policies if p in by_policy]
                x = list(range(len(present)))
                values = [metric(by_policy[p]) for p in present]
                bars = ax.bar(x, values, color="#4C78A8", width=0.68)
                ax.set_xticks(x, present)
                ax.set_title(distribution.capitalize())
                ax.set_ylabel(label)
                if speedup:
                    limits = [theoretical_speedup(by_policy[p]) for p in present]
                    ax.plot(
                        x, limits, color="#D55E00", linestyle="--", marker="_",
                        markersize=12, linewidth=1.8, label="FLOP-ratio ceiling"
                    )
                if stem == "memory":
                    for bar, policy in zip(bars, present):
                        padding = number(by_policy[policy], "padding_waste_pct")
                        ax.annotate(
                            f"{padding:.0f}% pad",
                            (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            xytext=(0, 3),
                            textcoords="offset points",
                            ha="center",
                            va="bottom",
                            fontsize=7.5,
                            rotation=90,
                        )
                finish_axis(ax, speedup=speedup)
                if speedup:
                    ax.legend(loc="best")
            fig.suptitle(f"Rank-bucketing {label.lower()} — {band}", y=1.03, fontsize=13)
            fig.tight_layout()
            paths += save_figure(
                fig, outdir, f"ablation_{stem}__{slug(band)}", formats, dpi
            )
    return paths


def plot_file(path: Path, output_root: Path, formats: Sequence[str], dpi: int) -> list[Path]:
    rows = read_rows(path)
    experiments = {row["experiment"] for row in rows}
    if len(experiments) != 1:
        raise ValueError(f"{path}: expected one experiment, got {sorted(experiments)}")
    experiment = experiments.pop()
    outdir = output_root / path.stem
    paths: list[Path] = []
    if experiment in {
        "precision_sweep",
        "workspace_winners",
        "workspace_confirmation",
    }:
        paths += precision_speedup_dashboard(rows, outdir, formats, dpi)
        paths += precision_metric_dashboard(
            rows,
            outdir,
            formats,
            dpi,
            metric=lambda r: number(r, "numeric_median_ms"),
            label="Median numerical time [ms]",
            stem="time",
            logy=True,
        )
        paths += precision_metric_dashboard(
            rows,
            outdir,
            formats,
            dpi,
            metric=ceiling_fraction,
            label="Achieved arithmetic ceiling [%]",
            stem="ceiling_fraction",
        )
        if experiment in {"workspace_winners", "workspace_confirmation"}:
            paths += precision_metric_dashboard(
                rows,
                outdir,
                formats,
                dpi,
                metric=lambda r: number(r, "memory_ratio"),
                label="Selected memory ratio",
                stem="selected_memory_ratio",
            )
    elif experiment == "memory_sweep":
        paths += memory_dashboard(
            rows,
            outdir,
            formats,
            dpi,
            metric=lambda r: number(r, "speedup_median"),
            label="Speedup over dense",
            stem="speedup",
            speedup=True,
        )
        paths += memory_dashboard(
            rows,
            outdir,
            formats,
            dpi,
            metric=lambda r: number(r, "numeric_median_ms"),
            label="Median numerical time [ms]",
            stem="time",
        )
        paths += memory_dashboard(
            rows,
            outdir,
            formats,
            dpi,
            metric=ceiling_fraction,
            label="Achieved arithmetic ceiling [%]",
            stem="ceiling_fraction",
        )
    elif experiment == "rank_bucketing_ablation":
        paths += ablation_figures(rows, outdir, formats, dpi)
    else:
        raise ValueError(f"{path}: unsupported experiment {experiment!r}")
    return paths


def discover_inputs(values: Sequence[Path]) -> list[Path]:
    if values:
        return [path.resolve() for path in values]
    return sorted(DEFAULT_RESULTS.glob("*.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "csv",
        nargs="*",
        type=Path,
        help="input CSVs; defaults to every experiments/results/gemm/*.csv",
    )
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_FIGURES)
    parser.add_argument("--formats", default="png", help="comma-separated: png,pdf,svg")
    parser.add_argument("--dpi", type=int, default=220)
    args = parser.parse_args()

    formats = [value.strip().lower() for value in args.formats.split(",") if value.strip()]
    unsupported = set(formats).difference({"png", "pdf", "svg"})
    if unsupported:
        parser.error(f"unsupported formats: {sorted(unsupported)}")
    inputs = discover_inputs(args.csv)
    if not inputs:
        parser.error("no result CSVs found")

    configure_matplotlib()
    all_paths = []
    for path in inputs:
        generated = plot_file(path, args.output_dir.resolve(), formats, args.dpi)
        all_paths.extend(generated)
        print(f"{path}: {len(generated)} figure files")
    print(f"Generated {len(all_paths)} files under {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
