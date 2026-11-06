"""Plot time and gradient error against tolerance for every benchmark problem.

Each point represents one tool and solver at one tolerance. The top row shows
the median time per gradient over parameter draws, with a bar from the
fastest to the slowest draw. The bottom row shows the median relative error
over the draws that didn't fail. AMICI C++ is shown by its own CPU time,
which leaves out the PEtab interface's overhead.

A draw fails if its gradient is NaN or its relative error is above 1, so
that it is worse than a zero gradient. A point where any draw failed is
ringed in red.

Run with `uv run --group benchmark python benchmarks/plot.py [results_dir]`.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter

HERE = Path(__file__).parent
PROBLEMS = {
    "Blasi_CellSystems2016": "Blasi et al. 2016",
    "Zheng_PNAS2012": "Zheng et al. 2012",
    "glycolysis": "Glycolysis and PPP",
}
COLOURS = {"enzax": "#2a78d6", "amici_jax": "#eb6834", "amici_cpp": "#1baf7a"}
TOOL_NAMES = {
    "enzax": "enzax",
    "amici_jax": "AMICI JAX",
    "amici_cpp": "AMICI C++",
}
STYLES = {
    "BDF": ("-", "o"),
    "Kvaerno5": ("--", "s"),
    "adjoint, Newton": ("-", "^"),
    "BDF, hybrid": (":", "o"),
    "adjoint, Newton, hybrid": (":", "^"),
}
SERIES = [
    ("enzax", "BDF"),
    ("enzax", "Kvaerno5"),
    ("enzax", "BDF, hybrid"),
    ("amici_jax", "Kvaerno5"),
    ("amici_cpp", "adjoint, Newton"),
    ("amici_cpp", "adjoint, Newton, hybrid"),
]
MEASURES = {
    "time": "Time per gradient (ms)",
    "error": "Gradient error (relative)",
}
TOLERANCES = [1e-10, 1e-8, 1e-6]
DODGE = 10**0.07
MAX_ERROR = 1.0
FAILURE = "#d1242f"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Summarise time and gradient error per tool, solver and tolerance."""
    results = results.assign(
        time=results["cpu_ms"].where(
            results["tool"] == "amici_cpp",
            results["time_ms"],
        ),
        error=results["gradient_error"].where(
            results["gradient_error"] <= MAX_ERROR,
        ),
    )
    summary = (
        results.groupby(["tool", "solver", "tolerance"])
        .agg(
            time_median=("time", "median"),
            time_min=("time", "min"),
            time_max=("time", "max"),
            error_median=("error", "median"),
            n_failed=("error", lambda error: error.isna().sum()),
        )
        .reset_index()
    )
    return summary


def plot(results_dir: Path) -> plt.Figure:
    plt.rcParams.update(
        {
            "font.size": 8,
            "font.family": "sans-serif",
            "axes.edgecolor": MUTED,
            "axes.labelcolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "axes.spines.top": False,
            "axes.spines.right": False,
        },
    )
    fig, axes = plt.subplots(
        len(MEASURES),
        len(PROBLEMS),
        figsize=(7.5, 4.8),
        sharex=True,
    )
    for column, (problem, title) in zip(
        axes.T,
        PROBLEMS.items(),
        strict=True,
    ):
        summary = summarise(pd.read_csv(results_dir / f"{problem}.csv"))
        for position, (tool, solver) in enumerate(SERIES):
            rows = summary[
                (summary["tool"] == tool) & (summary["solver"] == solver)
            ].sort_values("tolerance")
            tolerance = rows["tolerance"] * DODGE ** (
                position - (len(SERIES) - 1) / 2
            )
            linestyle, marker = STYLES[solver]
            column[0].vlines(
                tolerance,
                rows["time_min"],
                rows["time_max"],
                color=COLOURS[tool],
                linewidth=1,
                alpha=0.5,
            )
            for ax, measure in zip(column, MEASURES, strict=True):
                ax.plot(
                    tolerance,
                    rows[f"{measure}_median"],
                    color=COLOURS[tool],
                    linestyle=linestyle,
                    linewidth=1.5,
                    marker=marker,
                    markersize=5,
                    markeredgecolor="white",
                    markeredgewidth=1,
                    solid_capstyle="round",
                )
                failed = rows["n_failed"] > 0
                ax.plot(
                    tolerance[failed],
                    rows.loc[failed, f"{measure}_median"],
                    linestyle="none",
                    marker="o",
                    markersize=10,
                    markerfacecolor="none",
                    markeredgecolor=FAILURE,
                    markeredgewidth=1.5,
                )
        column[0].set_title(title, loc="left", color=TEXT, fontsize=9)
        column[-1].set_xlabel("Tolerance")
        for ax in column:
            ax.set_xscale("log")
            ax.set_yscale("log")
            ax.set_xticks(TOLERANCES)
            ax.xaxis.set_minor_formatter(NullFormatter())
            ax.grid(color=GRID, linewidth=0.6)
            ax.set_axisbelow(True)
        column[0].yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        column[0].yaxis.set_major_formatter(
            FuncFormatter(lambda v, _: f"{v:g}"),
        )
        column[0].yaxis.set_minor_formatter(NullFormatter())
    for ax, label in zip(axes[:, 0], MEASURES.values(), strict=True):
        ax.set_ylabel(label)
    handles = [
        Line2D(
            [],
            [],
            color=COLOURS[tool],
            linestyle=STYLES[solver][0],
            marker=STYLES[solver][1],
            markeredgecolor="white",
            linewidth=1.5,
            markersize=5,
            label=f"{TOOL_NAMES[tool]}, {solver}",
        )
        for tool, solver in SERIES
    ]
    tools = list(dict.fromkeys(tool for tool, _ in SERIES))
    tool_columns = [
        [h for h, (t, _) in zip(handles, SERIES) if t == tool] for tool in tools
    ]
    tool_columns[-1].append(
        Line2D(
            [],
            [],
            linestyle="none",
            marker="o",
            markersize=10,
            markerfacecolor="none",
            markeredgecolor=FAILURE,
            markeredgewidth=1.5,
            label="Failed on at least one draw",
        ),
    )
    n_rows = max(len(column) for column in tool_columns)
    columns = []
    for column in tool_columns:
        columns += column + [Line2D([], [], alpha=0)] * (n_rows - len(column))
    fig.legend(
        handles=columns,
        loc="lower center",
        ncol=len(tools),
        frameon=False,
        labelcolor=TEXT,
    )
    fig.tight_layout(rect=(0, 0.13, 1, 1))
    return fig


if __name__ == "__main__":
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
    fig = plot(results_dir)
    for suffix in ("pdf", "png"):
        fig.savefig(results_dir / f"work_precision.{suffix}", dpi=300)
    print("wrote", results_dir / "work_precision.pdf")
