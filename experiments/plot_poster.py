#!/usr/bin/env python3
"""Create poster figures for the CompressedFTLR dense-output experiments.

The script deliberately uses only the Python standard library and Matplotlib.
It produces:

* ``poster_scaling``: strong scaling plus the rank-ratio reference inset;
* ``poster_workspace``: performance versus normalized workspace;
* ``poster_results_panel``: a ready-to-place composite with one hero plot and
  the two smaller supporting plots.

Examples
--------

    python3 experiments/plot_poster.py

    python3 experiments/plot_poster.py \
        --formats pdf,svg,png \
        --workspace-sizes 4096,16384,65536
"""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.ticker import FuncFormatter, FixedLocator


PRECISION_STYLE = {
    "fp16_fp32": {
        "label": "FP16 / FP32 accumulate",
        "color": "#0072B2",
        "marker": "o",
    },
    "fp32_tf32": {
        "label": "FP32 / TF32",
        "color": "#E69F00",
        "marker": "s",
    },
    "fp32": {
        "label": "FP32",
        "color": "#222222",
        "marker": "^",
    },
}

SIZE_COLORS = {
    4096: "#0072B2",
    16384: "#D55E00",
    65536: "#009E73",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(f"missing benchmark file: {path}")
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def parse_int_list(value: str) -> list[int]:
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "lines.linewidth": 2.1,
            "lines.markersize": 6.5,
            "grid.color": "#D9D9D9",
            "grid.linewidth": 0.7,
            "grid.alpha": 0.75,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def scaling_rows(
    rows: Iterable[dict[str, str]],
    profile: str,
    tile_grid: int,
) -> list[dict[str, str]]:
    selected = []
    for row in rows:
        n = int(row["N"])
        tile_size = int(row["tile_size"])
        if row["profile"] != profile:
            continue
        if n != tile_grid * tile_size:
            continue
        selected.append(row)
    if not selected:
        raise ValueError(
            f"no scaling rows found for profile={profile!r}, tile_grid={tile_grid}"
        )
    return selected


def arithmetic_reference(rows: Iterable[dict[str, str]]) -> tuple[list[int], list[float]]:
    by_size: dict[int, list[float]] = defaultdict(list)
    for row in rows:
        n = int(row["N"])
        executed = float(row["executed_flops"])
        by_size[n].append((2.0 * n**3) / executed)
    sizes = sorted(by_size)
    references = [statistics.median(by_size[n]) for n in sizes]
    return sizes, references


def power_of_two_ticks(values: Iterable[float]) -> list[float]:
    values = list(values)
    low = min(min(values), 1.0)
    high = max(max(values), 1.0)
    first = math.floor(math.log2(low))
    last = math.ceil(math.log2(high))
    return [2.0**power for power in range(first, last + 1)]


def speedup_tick(value: float, _position: float) -> str:
    if value >= 1:
        return f"{value:g}×"
    return f"{value:g}×"


def draw_scaling(
    ax: Axes,
    rows: list[dict[str, str]],
    profile: str,
    tile_grid: int,
    annotate_final: bool = True,
) -> None:
    selected = scaling_rows(rows, profile, tile_grid)
    all_speedups = [
        float(row["dense_min_ms"]) / float(row["analyzed_min_ms"])
        for row in selected
    ]

    for precision, style in PRECISION_STYLE.items():
        precision_rows = sorted(
            (row for row in selected if row["precision"] == precision),
            key=lambda row: int(row["N"]),
        )
        if not precision_rows:
            continue
        sizes = [int(row["N"]) for row in precision_rows]
        speedups = [
            float(row["dense_min_ms"]) / float(row["analyzed_min_ms"])
            for row in precision_rows
        ]
        ax.plot(
            sizes,
            speedups,
            color=style["color"],
            marker=style["marker"],
            label=style["label"],
            zorder=3,
        )
        if annotate_final:
            ax.annotate(
                f"{speedups[-1]:.1f}×",
                xy=(sizes[-1], speedups[-1]),
                xytext=(5, 0),
                textcoords="offset points",
                color=style["color"],
                fontsize=9,
                fontweight="bold",
                va="center",
            )

    roof_sizes, roof = arithmetic_reference(selected)
    all_speedups.extend(roof)
    ax.plot(
        roof_sizes,
        roof,
        color="#666666",
        linestyle=(0, (5, 3)),
        linewidth=1.8,
        label="Equal-throughput arithmetic roof",
        zorder=2,
    )
    ax.axhline(
        1.0,
        color="#999999",
        linestyle=":",
        linewidth=1.4,
        label="Dense break-even",
        zorder=1,
    )

    sizes = sorted({int(row["N"]) for row in selected})
    ax.set_xscale("log", base=2)
    ax.set_yscale("log", base=2)
    ax.set_xticks(sizes, [f"{n // 1024}K" for n in sizes])
    ticks = power_of_two_ticks(all_speedups)
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(speedup_tick))
    ax.grid(True, which="major")
    ax.set_xlabel("Square matrix dimension $N$")
    ax.set_ylabel("Best observed speedup over dense GEMM")
    ax.set_title(
        "Variable-rank CompressedFTLR × CompressedFTLR → Dense",
        loc="left",
        fontweight="bold",
    )
    ax.legend(loc="upper left", frameon=False, ncol=2)


def draw_rank_reference(ax: Axes) -> None:
    ratios = [1 / 32, 1 / 16, 1 / 8]
    labels = [r"$r/b=1/32$", r"$r/b=1/16$", r"$r/b=1/8$"]
    storage_percent = [200.0 * ratio for ratio in ratios]
    memory_compression = [100.0 / value for value in storage_percent]
    roofs = [1.0 / (ratio + 2.0 * ratio**2) for ratio in ratios]
    colors = ["#56B4E9", "#0072B2", "#00517A"]

    bars = ax.bar(
        range(len(ratios)),
        storage_percent,
        width=0.68,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )
    ax.set_xticks(range(len(ratios)), labels)
    ax.set_ylabel("Factor storage / dense storage")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:g}%"))
    ax.set_ylim(0, max(storage_percent) * 1.48)
    ax.grid(True, axis="y")
    ax.set_axisbelow(True)
    ax.set_title("Rank ratio sets the opportunity", loc="left", fontweight="bold")

    for bar, compression, roof in zip(bars, memory_compression, roofs):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.8,
            f"{compression:.0f}× memory\n{roof:.1f}× roof",
            ha="center",
            va="bottom",
            fontsize=8.5,
            linespacing=1.15,
        )


def workspace_rows(
    rows: Iterable[dict[str, str]],
    precision: str,
    sizes: list[int],
) -> dict[int, list[dict[str, str]]]:
    selected: dict[int, list[dict[str, str]]] = {}
    for size in sizes:
        members = sorted(
            (
                row
                for row in rows
                if row["precision"] == precision and int(row["N"]) == size
            ),
            key=lambda row: int(row["rows_per_run"]),
        )
        if not members:
            raise ValueError(
                f"no workspace rows found for N={size}, precision={precision!r}"
            )
        selected[size] = members
    return selected


def draw_workspace(
    ax: Axes,
    rows: list[dict[str, str]],
    precision: str,
    sizes: list[int],
) -> None:
    selected = workspace_rows(rows, precision, sizes)
    fallback_colors = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#666666"]

    all_x = []
    all_y = []
    for index, size in enumerate(sizes):
        members = selected[size]
        minimum_workspace = min(int(row["workspace_bytes"]) for row in members)
        baseline = next(
            float(row["numeric_min_ms"])
            for row in members
            if int(row["workspace_bytes"]) == minimum_workspace
        )
        x = [int(row["workspace_bytes"]) / minimum_workspace for row in members]
        y = [baseline / float(row["numeric_min_ms"]) for row in members]
        all_x.extend(x)
        all_y.extend(y)
        color = SIZE_COLORS.get(size, fallback_colors[index % len(fallback_colors)])
        ax.plot(
            x,
            y,
            marker="o",
            color=color,
            label=f"$N={size // 1024}$K",
        )

    ax.axhline(1.0, color="#999999", linestyle=":", linewidth=1.3)
    ax.set_xlim(min(all_x) - 0.1, max(all_x) + 0.2)
    ax.set_ylim(min(min(all_y) * 0.95, 0.98), max(all_y) * 1.08)
    ax.set_xlabel(r"Workspace / minimum workspace, $W/W_{\min}$")
    ax.set_ylabel(r"Best observed speedup over $W_{\min}$")
    ax.set_title(
        "Workspace exposes concurrency until saturation",
        loc="left",
        fontweight="bold",
    )
    ax.grid(True)
    ax.legend(frameon=False, loc="best")


def save_figure(
    figure: plt.Figure,
    output_dir: Path,
    stem: str,
    formats: list[str],
    dpi: int,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for extension in formats:
        path = output_dir / f"{stem}.{extension}"
        kwargs = {"dpi": dpi} if extension.lower() == "png" else {}
        figure.savefig(path, **kwargs)
        print(f"wrote {path}")


def make_scaling_figure(
    scaling: list[dict[str, str]],
    profile: str,
    tile_grid: int,
) -> plt.Figure:
    figure = plt.figure(figsize=(10.6, 4.8), constrained_layout=True)
    grid = figure.add_gridspec(1, 2, width_ratios=[3.25, 1.25])
    draw_scaling(figure.add_subplot(grid[0, 0]), scaling, profile, tile_grid)
    draw_rank_reference(figure.add_subplot(grid[0, 1]))
    return figure


def make_workspace_figure(
    workspace: list[dict[str, str]],
    precision: str,
    sizes: list[int],
) -> plt.Figure:
    figure, ax = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    draw_workspace(ax, workspace, precision, sizes)
    return figure


def make_composite_figure(
    scaling: list[dict[str, str]],
    workspace: list[dict[str, str]],
    profile: str,
    tile_grid: int,
    workspace_precision: str,
    workspace_sizes: list[int],
) -> plt.Figure:
    figure = plt.figure(figsize=(12.5, 7.0), constrained_layout=True)
    grid = figure.add_gridspec(
        2,
        2,
        width_ratios=[2.25, 1.0],
        height_ratios=[1.0, 1.0],
    )
    scaling_ax = figure.add_subplot(grid[:, 0])
    rank_ax = figure.add_subplot(grid[0, 1])
    workspace_ax = figure.add_subplot(grid[1, 1])
    draw_scaling(scaling_ax, scaling, profile, tile_grid)
    draw_rank_reference(rank_ax)
    draw_workspace(workspace_ax, workspace, workspace_precision, workspace_sizes)
    return figure


def parse_arguments() -> argparse.Namespace:
    repository = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=repository / "experiments" / "results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=repository / "experiments" / "figures",
    )
    parser.add_argument("--scaling-profile", default="skewed_b32_b8")
    parser.add_argument("--tile-grid", type=int, default=16)
    parser.add_argument("--workspace-precision", default="fp16_fp32")
    parser.add_argument(
        "--workspace-sizes",
        type=parse_int_list,
        default=parse_int_list("4096,16384,65536"),
    )
    parser.add_argument(
        "--formats",
        type=lambda value: [item.strip() for item in value.split(",") if item.strip()],
        default=["pdf", "svg", "png"],
    )
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    configure_matplotlib()
    scaling = read_csv(arguments.results_dir / "compressed_dense.csv")
    workspace = read_csv(arguments.results_dir / "rows_per_run.csv")

    scaling_figure = make_scaling_figure(
        scaling,
        arguments.scaling_profile,
        arguments.tile_grid,
    )
    workspace_figure = make_workspace_figure(
        workspace,
        arguments.workspace_precision,
        arguments.workspace_sizes,
    )
    composite_figure = make_composite_figure(
        scaling,
        workspace,
        arguments.scaling_profile,
        arguments.tile_grid,
        arguments.workspace_precision,
        arguments.workspace_sizes,
    )

    save_figure(
        scaling_figure,
        arguments.output_dir,
        "poster_scaling",
        arguments.formats,
        arguments.dpi,
    )
    save_figure(
        workspace_figure,
        arguments.output_dir,
        "poster_workspace",
        arguments.formats,
        arguments.dpi,
    )
    save_figure(
        composite_figure,
        arguments.output_dir,
        "poster_results_panel",
        arguments.formats,
        arguments.dpi,
    )
    plt.close("all")


if __name__ == "__main__":
    main()
