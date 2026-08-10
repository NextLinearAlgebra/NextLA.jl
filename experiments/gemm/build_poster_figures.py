#!/usr/bin/env python3
"""Generate the poster GEMM figures and their numerical highlights."""

from __future__ import annotations

import argparse
import csv
import math
import os
import tempfile
from pathlib import Path
from typing import Iterable, Sequence

_mpl_cache = Path(tempfile.gettempdir()) / "nextla-matplotlib"
_mpl_cache.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(_mpl_cache))

import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.ticker import FixedLocator, FuncFormatter


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RESULTS = ROOT / "experiments" / "results" / "gemm"
DEFAULT_KBLAS = (
    DEFAULT_RESULTS / "kblas" / "constant_rank_fp32_rank_b16_b8.csv"
)
DEFAULT_OUTPUT = ROOT / "experiments" / "figures" / "gemm" / "poster"

# Fixed categorical hue order (dataviz palette default) -- assigned in sequence,
# never cycled or re-ordered per figure.
BLUE = "#2a78d6"
ORANGE = "#eb6834"
RED = "#e34948"
GREEN = "#008300"
NEUTRAL = "#898781"    # reference / dense-baseline lines (not a series identity)
GRID_COLOR = "#e1e0d9"
AXIS_COLOR = "#c3c2b7"
TEXT_PRIMARY = "#0b0b0b"
TEXT_SECONDARY = "#52514e"

FIGSIZE = (6.4, 5.0)

DISTRIBUTION_STYLE = {
    "skewed": (RED, "o", "▇▅▃▂▁  Skewed"),
    "uniform": (GREEN, "s", "▄▄▄▄▄  Uniform"),
}
PRECISION_LABEL = {"bf16": "BF16", "fp16": "FP16", "tf32": "TF32", "fp32": "FP32"}
PRECISION_ORDER = ("bf16", "fp16", "tf32", "fp32")
LAYOUT_STYLE = {
    "compressed_compressed": (BLUE, "o", r"$A_{tlr} \times B_{tlr}$ (NextLA)"),
    "compressed_dense": (ORANGE, "s", r"$A_{tlr} \times B$ (NextLA)"),
}
KBLAS_COLOR = GREEN
KBLAS_MARKER = "o"
KBLAS_LABEL = r"$A_{tlr} \times B_{tlr}$ (KBLAS, padded)"
DIVISOR_STYLE = {8: ("o", "-", "b=N/8"), 16: ("s", (0, (5, 2)), "b=N/16")}
SIZES = (4096, 8192, 16384, 32768, 65536)
MEMORY_RATIO_LABEL = (
    r"$\frac{\mathrm{sizeof}(A_{tlr}+B_{tlr}+\mathrm{workspace})}{\mathrm{sizeof}(A+B)}$"
)


def configure_style() -> None:
    sns.set_theme(style="ticks")
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 15,
            "axes.labelsize": 17,
            "axes.labelcolor": TEXT_PRIMARY,
            "axes.edgecolor": AXIS_COLOR,
            "axes.linewidth": 1.1,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 14,
            "ytick.labelsize": 14,
            "xtick.color": TEXT_PRIMARY,
            "ytick.color": TEXT_PRIMARY,
            "legend.fontsize": 13,
            "legend.labelcolor": TEXT_PRIMARY,
            "lines.linewidth": 2.0,
            "lines.markersize": 8,
            "grid.color": GRID_COLOR,
            "grid.linewidth": 0.8,
            "grid.alpha": 0.6,
            "text.color": TEXT_PRIMARY,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.bbox": "tight",
        }
    )


def finish_axes(ax) -> None:
    sns.despine(ax=ax)
    ax.grid(True, axis="y", which="major", zorder=0)
    ax.set_axisbelow(True)
    ax.tick_params(length=4, width=1.0, color=AXIS_COLOR)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def num(row: dict[str, str], name: str) -> float:
    return float(row[name])


def integer(row: dict[str, str], name: str) -> int:
    return int(row[name])


def compressed(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    return [row for row in rows if row.get("record_kind") == "compressed"]


def arithmetic_ceiling(row: dict[str, str]) -> float:
    return 2.0 * integer(row, "N") ** 3 / num(row, "executed_flops")


def geometric_mean(values: Sequence[float]) -> float:
    if not values or any(value <= 0 for value in values):
        raise ValueError("geometric mean requires positive values")
    return math.exp(sum(math.log(value) for value in values) / len(values))


def save_figure(
    fig,
    output: Path,
    stem: str,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    paths = []
    for extension in formats:
        destination = output / f"{stem}.{extension}"
        fig.savefig(destination, dpi=dpi if extension == "png" else None)
        paths.append(destination)
    plt.close(fig)
    return paths


def format_size(value: float, _position=None) -> str:
    return f"{int(value) // 1024}K"


def format_plain(value: float, _position=None) -> str:
    return f"{value:g}"


def format_percent(value: float, _position=None) -> str:
    return f"{100.0 * value:g}%"


def format_speedup_tick(value: float, _position=None) -> str:
    return f"{value:g}×"


def setup_size_axis(ax) -> None:
    ax.set_xscale("log", base=2)
    ax.set_xticks(SIZES)
    ax.xaxis.set_major_formatter(FuncFormatter(format_size))
    ax.set_xlim(SIZES[0] / 1.08, SIZES[-1] * 1.08)


# --- Figures 1a / 1b: memory-performance Pareto, one tile size per panel ---


def memory_candidates(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    candidates = [
        row for row in compressed(rows)
        if row["operand_layout"] == "compressed_compressed"
        and row["precision"] == "fp16"
        and row["rank_band"] == "b32_b16"
        and integer(row, "N") == 16384
    ]
    expected = 2 * 2 * 7
    if len(candidates) != expected:
        raise ValueError(
            f"memory figure expected {expected} TLR×TLR rows, got {len(candidates)}"
        )
    dense_times = {round(num(row, "dense_median_ms"), 9) for row in candidates}
    if len(dense_times) != 1:
        raise ValueError("memory figure rows do not share one dense baseline")
    return candidates


def memory_metrics(candidates: list[dict[str, str]]) -> dict[str, float | int | str]:
    hero_pool = [
        row for row in candidates
        if row["distribution"] == "skewed" and integer(row, "tile_divisor") == 8
    ]
    hero = min(hero_pool, key=lambda row: num(row, "numeric_median_ms"))
    minimum_memory = min(hero_pool, key=lambda row: num(row, "memory_ratio"))
    return {
        "hero_speedup": num(hero, "speedup_median"),
        "hero_memory_ratio": num(hero, "memory_ratio"),
        "hero_time_ms": num(hero, "numeric_median_ms"),
        "hero_workspace_runs": integer(hero, "workspace_parameter"),
        "minimum_memory_ratio": num(minimum_memory, "memory_ratio"),
        "minimum_memory_speedup": num(minimum_memory, "speedup_median"),
    }


def plot_memory_panel(
    candidates: list[dict[str, str]],
    divisor: int,
    stem: str,
    output: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    fig, ax = plt.subplots(figsize=FIGSIZE)

    plotted: dict[str, list[dict[str, str]]] = {}
    for distribution in ("skewed", "uniform"):
        series = [
            row for row in candidates
            if integer(row, "tile_divisor") == divisor and row["distribution"] == distribution
            and num(row, "memory_ratio") <= 0.315
        ]
        series.sort(key=lambda row: num(row, "memory_ratio"))
        plotted[distribution] = series
        color, marker, label = DISTRIBUTION_STYLE[distribution]
        xs = [num(row, "memory_ratio") for row in series]
        ys = [num(row, "numeric_median_ms") for row in series]
        ax.plot(
            xs, ys, color=color, marker=marker, markersize=7, label=label,
            markeredgecolor="white", markeredgewidth=1.0, zorder=3,
        )

    all_points = plotted["skewed"] + plotted["uniform"]
    x_min = min(num(row, "memory_ratio") for row in all_points)
    x_max = max(num(row, "memory_ratio") for row in all_points)
    y_min = min(num(row, "numeric_median_ms") for row in all_points)
    y_max = max(num(row, "numeric_median_ms") for row in all_points)
    ax.set_xlim(0.085, 0.315)
    ax.set_ylim(y_min / 1.08, y_max * 1.08)
    x_ticks = (0.10, 0.15, 0.20, 0.25, 0.30)
    y_ticks = (2.0, 2.5, 3.0, 3.5, 4.0)
    ax.xaxis.set_major_locator(FixedLocator(x_ticks))
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.xaxis.set_major_formatter(FuncFormatter(format_percent))
    ax.yaxis.set_major_formatter(FuncFormatter(format_plain))
    finish_axes(ax)
    ax.tick_params(axis="x", labelsize=12)
    plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    ax.set_xlabel(MEMORY_RATIO_LABEL, fontsize=19, labelpad=10)
    ax.set_ylabel("Time [ms]")
    ax.legend(
        frameon=False, loc="upper right", fontsize=10,
        handlelength=1.7, handletextpad=0.45, borderaxespad=0.25,
    )
    fig.tight_layout()
    return save_figure(fig, output, stem, formats, dpi)


# --- Figure 2: precision speedup, one subplot per precision ---


def layout_candidates(
    rows: list[dict[str, str]], layout: str, divisor: int = 8
) -> list[dict[str, str]]:
    candidates = [
        row for row in compressed(rows)
        if row["operand_layout"] == layout
        and row["distribution"] == "skewed"
        and row["rank_band"] == "b32_b16"
        and integer(row, "tile_divisor") == divisor
    ]
    expected = len(SIZES) * len(PRECISION_ORDER)
    if len(candidates) != expected:
        raise ValueError(
            f"{layout} figure expected {expected} rows, got {len(candidates)}"
        )
    return candidates


def kblas_fp32_speedup(
    kblas_rows: list[dict[str, str]],
    dense_by_n: dict[int, float],
    divisor: int = 8,
    rank_over_b: float = 1.0 / 16.0,
) -> list[tuple[int, float, float]]:
    rows = [
        row for row in kblas_rows
        if integer(row, "q") == divisor
        and math.isclose(num(row, "rank_over_b"), rank_over_b, rel_tol=0, abs_tol=1e-12)
    ]
    if len(rows) != len(SIZES):
        raise ValueError(f"kblas fp32 speedup expected {len(SIZES)} rows, got {len(rows)}")
    return sorted(
        (
            (
                integer(row, "N"),
                dense_by_n[integer(row, "N")] / num(row, "tlr_median_ms"),
                num(row, "flop_ratio_ceiling"),
            )
            for row in rows
        ),
        key=lambda pair: pair[0],
    )


def precision_metrics(candidates: list[dict[str, str]]) -> dict[str, float | int | str]:
    hero = max(candidates, key=lambda row: num(row, "speedup_median"))
    hero_ceiling = arithmetic_ceiling(hero)
    endpoints = {
        row["precision"]: row
        for row in candidates if integer(row, "N") == SIZES[-1]
    }
    return {
        "hero_N": integer(hero, "N"),
        "hero_precision": hero["precision"],
        "hero_speedup": num(hero, "speedup_median"),
        "hero_ceiling": hero_ceiling,
        "hero_ceiling_fraction": num(hero, "speedup_median") / hero_ceiling,
        "hero_memory_ratio": num(hero, "memory_ratio"),
        "hero_time_ms": num(hero, "numeric_median_ms"),
        "bf16_speedup_N65536": num(endpoints["bf16"], "speedup_median"),
        "bf16_time_ms_N65536": num(endpoints["bf16"], "numeric_median_ms"),
        "fp16_speedup_N65536": num(endpoints["fp16"], "speedup_median"),
        "fp16_time_ms_N65536": num(endpoints["fp16"], "numeric_median_ms"),
        "tf32_speedup_N65536": num(endpoints["tf32"], "speedup_median"),
        "tf32_time_ms_N65536": num(endpoints["tf32"], "numeric_median_ms"),
    }


def plot_precision_grid(
    cc_rows: list[dict[str, str]],
    cd_rows: list[dict[str, str]],
    kblas_points: list[tuple[int, float, float]],
    stem: str,
    output: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    fig, axes = plt.subplots(1, 4, figsize=(19.0, 4.8), sharex=True, sharey=True)

    legend_handles: list = []
    legend_labels: list[str] = []
    seen_layout: set[str] = set()
    for ax, precision in zip(axes, PRECISION_ORDER):
        for layout, rows in (
            ("compressed_compressed", cc_rows),
            ("compressed_dense", cd_rows),
        ):
            color, marker, label = LAYOUT_STYLE[layout]
            series = sorted(
                (row for row in rows if row["precision"] == precision),
                key=lambda row: integer(row, "N"),
            )
            line, = ax.plot(
                [integer(row, "N") for row in series],
                [num(row, "speedup_median") for row in series],
                color=color, marker=marker, markersize=7,
                markeredgecolor="white", markeredgewidth=1.0, zorder=3,
            )
            if layout not in seen_layout:
                legend_handles.append(line)
                legend_labels.append(label)
                seen_layout.add(layout)

        if precision == "fp32":
            line, = ax.plot(
                [n for n, _, _ in kblas_points],
                [speedup for _, speedup, _ in kblas_points],
                color=KBLAS_COLOR, marker=KBLAS_MARKER, markersize=7,
                markeredgecolor="white", markeredgewidth=1.0, zorder=3,
            )
            legend_handles.append(line)
            legend_labels.append(KBLAS_LABEL)

        ax.axhline(1.0, color=AXIS_COLOR, linestyle=":", linewidth=1.2, zorder=1)
        ax.set_xscale("log", base=2)
        ax.set_xticks(SIZES)
        ax.xaxis.set_major_formatter(FuncFormatter(format_size))
        ax.set_xlim(SIZES[0] / 1.22, SIZES[-1] * 1.22)
        ax.set_yscale("log", base=2)
        ax.set_ylim(0.8, 32.0)
        ax.set_yticks((1, 2, 4, 8, 16, 32))
        ax.yaxis.set_major_formatter(FuncFormatter(format_speedup_tick))
        finish_axes(ax)
        ax.set_xlabel("N")
        ax.set_title(
            PRECISION_LABEL[precision], loc="center", fontsize=15,
            color=TEXT_SECONDARY, fontweight="normal", pad=10,
        )

    axes[0].set_ylabel("Speedup")

    fig.legend(
        legend_handles, legend_labels, frameon=False, loc="lower center",
        bbox_to_anchor=(0.5, 1.0), ncol=len(legend_labels),
        columnspacing=1.4, handletextpad=0.45, fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return save_figure(fig, output, stem, formats, dpi)


# --- Figure 5: precision speedup as grouped bars ---


def plot_precision_bar_grid(
    cc_rows: list[dict[str, str]],
    cd_rows: list[dict[str, str]],
    kblas_points: list[tuple[int, float, float]],
    stem: str,
    output: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    """Grouped-bar counterpart of Figure 2, with bars anchored at 1x."""
    fig, axes = plt.subplots(1, 4, figsize=(19.0, 4.8), sharex=True, sharey=True)
    positions = list(range(len(SIZES)))
    kblas_by_n = {n: speedup for n, speedup, _ in kblas_points}

    legend_handles: list = []
    legend_labels: list[str] = []
    for ax, precision in zip(axes, PRECISION_ORDER):
        layouts = [
            ("compressed_compressed", cc_rows),
            ("compressed_dense", cd_rows),
        ]
        include_kblas = precision == "fp32"
        width = 0.25 if include_kblas else 0.34
        offsets = (-width, 0.0, width) if include_kblas else (-width / 2, width / 2)

        for offset, (layout, rows) in zip(offsets, layouts):
            color, _marker, label = LAYOUT_STYLE[layout]
            by_n = {
                integer(row, "N"): num(row, "speedup_median")
                for row in rows if row["precision"] == precision
            }
            values = [by_n[n] for n in SIZES]
            bars = ax.bar(
                [position + offset for position in positions],
                [value - 1.0 for value in values],
                width=width,
                bottom=1.0,
                color=color,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            if precision == PRECISION_ORDER[0]:
                legend_handles.append(bars)
                legend_labels.append(label)

        if include_kblas:
            values = [kblas_by_n[n] for n in SIZES]
            bars = ax.bar(
                [position + offsets[-1] for position in positions],
                [value - 1.0 for value in values],
                width=width,
                bottom=1.0,
                color=KBLAS_COLOR,
                edgecolor="white",
                linewidth=0.8,
                zorder=3,
            )
            legend_handles.append(bars)
            legend_labels.append(KBLAS_LABEL)

        ax.axhline(1.0, color=AXIS_COLOR, linestyle=":", linewidth=1.2, zorder=1)
        ax.set_xticks(positions, [format_size(n) for n in SIZES])
        ax.set_xlim(-0.65, len(SIZES) - 0.35)
        ax.set_yscale("log", base=2)
        ax.set_ylim(0.8, 32.0)
        ax.set_yticks((1, 2, 4, 8, 16, 32))
        ax.yaxis.set_major_formatter(FuncFormatter(format_speedup_tick))
        finish_axes(ax)
        ax.set_xlabel("N")
        ax.set_title(
            PRECISION_LABEL[precision], loc="center", fontsize=15,
            color=TEXT_SECONDARY, fontweight="normal", pad=10,
        )

    axes[0].set_ylabel("Speedup")
    fig.legend(
        legend_handles, legend_labels, frameon=False, loc="lower center",
        bbox_to_anchor=(0.5, 1.0), ncol=len(legend_labels),
        columnspacing=1.4, handletextpad=0.45, fontsize=12,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    return save_figure(fig, output, stem, formats, dpi)


# --- Figures 3a / 3b: NextLA vs. KBLAS, padded and memory-matched panels ---


def kblas_padded_candidates(
    skewed_rows: list[dict[str, str]],
    kblas_rows: list[dict[str, str]],
) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    nextla = [
        row for row in compressed(skewed_rows)
        if row["operand_layout"] == "compressed_compressed"
        and row["precision"] == "fp32"
        and row["distribution"] == "skewed"
        and row["rank_band"] == "b32_b16"
    ]
    if len(nextla) != len(SIZES) * 2:
        raise ValueError(f"padded KBLAS figure expected 10 NextLA rows, got {len(nextla)}")
    padded = [
        row for row in kblas_rows
        if math.isclose(num(row, "rank_over_b"), 1.0 / 16.0, rel_tol=0, abs_tol=1e-12)
    ]
    if len(padded) != len(SIZES) * 2:
        raise ValueError(f"padded KBLAS figure expected 10 KBLAS rows, got {len(padded)}")
    return nextla, padded


def plot_padded_comparison(
    nextla: list[dict[str, str]],
    padded: list[dict[str, str]],
    stem: str,
    output: Path,
    formats: Sequence[str],
    dpi: int,
) -> tuple[list[Path], dict[str, float]]:
    fig, ax = plt.subplots(figsize=FIGSIZE)
    padded_ratios = []
    for divisor in (16,):
        marker, linestyle, divisor_label = DIVISOR_STYLE[divisor]
        nextla_series = sorted(
            (row for row in nextla if integer(row, "tile_divisor") == divisor),
            key=lambda row: integer(row, "N"),
        )
        kblas_series = sorted(
            (row for row in padded if integer(row, "q") == divisor),
            key=lambda row: integer(row, "N"),
        )
        x = [integer(row, "N") for row in nextla_series]
        nextla_time = [num(row, "numeric_median_ms") for row in nextla_series]
        kblas_time = [num(row, "tlr_median_ms") for row in kblas_series]
        padded_ratios.extend(k / n for n, k in zip(nextla_time, kblas_time))
        ax.plot(
            x, nextla_time, color=BLUE, marker=marker, linestyle="-",
            label="NextLA",
            markeredgecolor="white", markeredgewidth=1.2, zorder=3,
        )
        ax.plot(
            x, kblas_time, color=ORANGE, marker=marker, linestyle="-",
            label=r"KBLAS",
            markeredgecolor="white", markeredgewidth=1.2, zorder=3,
        )
    setup_size_axis(ax)
    ax.set_yscale("log")
    finish_axes(ax)
    ax.set_xlabel("N")
    ax.set_ylabel("Time [ms]")
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    fig.tight_layout()
    paths = save_figure(fig, output, stem, formats, dpi)
    return paths, {
        "padded_min_speedup": min(padded_ratios),
        "padded_max_speedup": max(padded_ratios),
    }


def plot_constant_rank_comparison(
    comparison_rows: list[dict[str, str]],
    stem: str,
    output: Path,
    formats: Sequence[str],
    dpi: int,
) -> list[Path]:
    """Controlled b=N/16, r=b/16 comparison using tuned workspace winners."""
    series = sorted(
        (
            row for row in comparison_rows
            if integer(row, "q") == 16
            and math.isclose(
                num(row, "rank_over_b"), 1.0 / 16.0,
                rel_tol=0, abs_tol=1e-12,
            )
        ),
        key=lambda row: integer(row, "N"),
    )
    if len(series) != len(SIZES):
        raise ValueError(
            f"constant-rank figure expected {len(SIZES)} rows, got {len(series)}"
        )
    for row in series:
        if not math.isclose(
            num(row, "nextla_flop_ratio_ceiling"),
            num(row, "kblas_flop_ratio_ceiling"),
            rel_tol=1e-9, abs_tol=1e-9,
        ):
            raise ValueError("constant-rank arithmetic ceilings do not match")

    x = [integer(row, "N") for row in series]
    nextla_time = [num(row, "nextla_median_ms") for row in series]
    kblas_time = [num(row, "kblas_median_ms") for row in series]
    ceiling_time = [
        num(row, "nextla_dense_median_ms") /
        num(row, "nextla_flop_ratio_ceiling")
        for row in series
    ]

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(
        x, nextla_time, color=BLUE, marker="s", linestyle="-",
        label="NextLA",
        markeredgecolor="white", markeredgewidth=1.2, zorder=3,
    )
    ax.plot(
        x, kblas_time, color=KBLAS_COLOR, marker="o", linestyle="-",
        label="KBLAS",
        markeredgecolor="white", markeredgewidth=1.2, zorder=3,
    )
    ax.plot(
        x, ceiling_time, color=NEUTRAL, linestyle=(0, (5, 3)),
        linewidth=1.7, label="Arithmetic ceiling", zorder=2,
    )
    setup_size_axis(ax)
    ax.set_yscale("log")
    finish_axes(ax)
    ax.set_xlabel("N")
    ax.set_ylabel("Time [ms]")
    ax.legend(frameon=False, loc="upper left", fontsize=11)
    fig.tight_layout()
    return save_figure(fig, output, stem, formats, dpi)


def fastest_kblas_summary(rows: list[dict[str, str]]) -> dict[str, float | int]:
    values = [num(row, "kblas_time_over_nextla") for row in rows]
    return {
        "wins": sum(value > 1.0 for value in values),
        "cases": len(values),
        "geomean": geometric_mean(values),
    }


def constant_rank_figure_metrics(
    rows: list[dict[str, str]],
) -> dict[str, float | int]:
    selected = [
        row for row in rows
        if integer(row, "q") == 16
        and math.isclose(
            num(row, "rank_over_b"), 1.0 / 16.0,
            rel_tol=0, abs_tol=1e-12,
        )
    ]
    if len(selected) != len(SIZES):
        raise ValueError(
            f"constant-rank highlights expected {len(SIZES)} rows, got {len(selected)}"
        )
    values = [num(row, "kblas_time_over_nextla") for row in selected]
    return {
        "wins": sum(value > 1.0 for value in values),
        "cases": len(values),
        "geomean": geometric_mean(values),
        "minimum": min(values),
        "maximum": max(values),
    }


def write_highlights(
    output: Path,
    memory: dict[str, float | int | str],
    precision: dict[str, float | int | str],
    kblas: dict[str, float | int | str],
    constant_rank: dict[str, float | int],
    full_constant_rank: dict[str, float | int],
    sources: Sequence[Path],
) -> Path:
    destination = output / "highlights.md"
    lines = [
        "# Poster-ready GEMM highlights",
        "",
        "All NextLA figures use median numerical time from the stored CSV rows.",
        "",
        "## Memory–performance",
        "",
        (
            f"- At N=16384, FP16, skewed TLR×TLR, b=N/8: "
            f"{memory['hero_speedup']:.2f}× speedup at "
            f"{100.0 * memory['hero_memory_ratio']:.1f}% total memory "
            f"({memory['hero_time_ms']:.3f} ms, runs={memory['hero_workspace_runs']})."
        ),
        (
            f"- At the minimum measured memory point ({100.0 * memory['minimum_memory_ratio']:.1f}%), "
            f"NextLA still achieves {memory['minimum_memory_speedup']:.2f}× speedup."
        ),
        "",
        "## Precision scaling",
        "",
        (
            f"- Best relative result: {precision['hero_speedup']:.2f}× FP32 speedup "
            f"at N={precision['hero_N']}, reaching "
            f"{100.0 * precision['hero_ceiling_fraction']:.1f}% of its "
            f"{precision['hero_ceiling']:.2f}× executed-FLOP ceiling while using "
            f"{100.0 * precision['hero_memory_ratio']:.1f}% total memory."
        ),
        (
            f"- At N=65536: FP16 {precision['fp16_speedup_N65536']:.2f}× "
            f"({precision['fp16_time_ms_N65536']:.1f} ms), BF16 "
            f"{precision['bf16_speedup_N65536']:.2f}× "
            f"({precision['bf16_time_ms_N65536']:.1f} ms), and TF32 "
            f"{precision['tf32_speedup_N65536']:.2f}× "
            f"({precision['tf32_time_ms_N65536']:.1f} ms)."
        ),
        "- FP32 has the largest relative speedup; FP16/BF16 have the lowest absolute time.",
        "",
        "## KBLAS comparison",
        "",
        (
            f"- Application/API comparison: NextLA skewed ranks are "
            f"{kblas['padded_min_speedup']:.2f}–{kblas['padded_max_speedup']:.2f}× "
            "faster than KBLAS padded uniformly to r=b/16. This is not a constant-work comparison."
        ),
        (
            f"- Controlled b=N/16, r=b/16 comparison using the fastest tuned "
            f"NextLA workspace: {constant_rank['wins']}/{constant_rank['cases']} wins, "
            f"{constant_rank['geomean']:.2f}× geometric mean "
            f"({constant_rank['minimum']:.2f}–{constant_rank['maximum']:.2f}×)."
        ),
        (
            f"- Across the complete 20-case constant-rank grid, fastest-workspace "
            f"NextLA wins {full_constant_rank['wins']}/{full_constant_rank['cases']} "
            f"with a {full_constant_rank['geomean']:.2f}× geometric mean."
        ),
    ]
    lines += ["", "## Sources", ""]
    lines.extend(
        f"- `{path.relative_to(ROOT) if path.is_relative_to(ROOT) else path}`"
        for path in sources
    )
    destination.write_text("\n".join(lines) + "\n")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS)
    parser.add_argument("--kblas", type=Path, default=DEFAULT_KBLAS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--formats", default="png,pdf,svg")
    parser.add_argument("--dpi", type=int, default=300)
    args = parser.parse_args()

    formats = [item.strip().lower() for item in args.formats.split(",") if item.strip()]
    unsupported = set(formats).difference({"png", "pdf", "svg"})
    if unsupported:
        parser.error(f"unsupported formats: {sorted(unsupported)}")
    configure_style()

    results = args.results_dir.resolve()
    memory_path = (
        results / "nextla" / "memory_pareto_fp16_n16384_rank_b32_b16.csv"
    )
    skewed_path = (
        results / "nextla" /
        "precision_scaling_skewed_rank_b32_b16_best_workspace.csv"
    )
    fastest_path = (
        results / "comparisons" /
        "constant_rank_fp32_best_nextla_vs_kblas.csv"
    )
    kblas_path = args.kblas.resolve()
    sources = [memory_path, skewed_path, kblas_path, fastest_path]
    for path in sources:
        if not path.is_file():
            raise FileNotFoundError(path)

    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []

    memory_rows = memory_candidates(read_csv(memory_path))
    memory_metric_values = memory_metrics(memory_rows)
    paths += plot_memory_panel(
        memory_rows, 8, "figure_1a_memory_pareto_b_n8", output, formats, args.dpi
    )
    paths += plot_memory_panel(
        memory_rows, 16, "figure_1b_memory_pareto_b_n16", output, formats, args.dpi
    )

    skewed_rows = read_csv(skewed_path)
    kblas_rows = read_csv(kblas_path)

    cc_rows = layout_candidates(skewed_rows, "compressed_compressed")
    cd_rows = layout_candidates(skewed_rows, "compressed_dense")
    precision_metric_values = precision_metrics(cc_rows)
    dense_by_n = {
        integer(row, "N"): num(row, "dense_median_ms")
        for row in cc_rows if row["precision"] == "fp32"
    }
    kblas_points = kblas_fp32_speedup(kblas_rows, dense_by_n)
    paths += plot_precision_grid(
        cc_rows, cd_rows, kblas_points,
        "figure_2_precision_scaling_skewed_b_n8", output, formats, args.dpi
    )
    paths += plot_precision_bar_grid(
        cc_rows, cd_rows, kblas_points,
        "figure_5_precision_speedup_bars_skewed_b_n8", output, formats, args.dpi
    )

    nextla_rows, padded_rows = kblas_padded_candidates(skewed_rows, kblas_rows)
    padded_paths, padded_metrics = plot_padded_comparison(
        nextla_rows, padded_rows,
        "figure_3_skewed_nextla_vs_padded_kblas_b_n16",
        output, formats, args.dpi,
    )
    paths += padded_paths
    fastest_rows = read_csv(fastest_path)
    paths += plot_constant_rank_comparison(
        fastest_rows,
        "figure_4_constant_rank_nextla_vs_kblas_b_n16_r_b16",
        output, formats, args.dpi,
    )
    fastest_metrics = fastest_kblas_summary(fastest_rows)
    constant_metrics = constant_rank_figure_metrics(fastest_rows)

    highlights = write_highlights(
        output,
        memory_metric_values,
        precision_metric_values,
        padded_metrics,
        constant_metrics,
        fastest_metrics,
        sources,
    )
    print(f"Generated {len(paths)} figure files under {output}")
    print(f"Highlights: {highlights}")


if __name__ == "__main__":
    main()
