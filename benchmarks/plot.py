"""Plot time against gradient error for every benchmark problem.

Each point is one tool and solver at one integrator tolerance: the median
over parameter draws of the time per gradient and of the gradient's error
relative to the reference. AMICI C++ is shown by its own CPU time, which
leaves out the PEtab interface's overhead. Its forward sensitivities at the
reference tolerance are the reference, so they have no error to plot.

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
    "adjoint": ("-", "^"),
    "forward": ("--", "D"),
}
SERIES = [
    ("enzax", "BDF"),
    ("enzax", "Kvaerno5"),
    ("amici_jax", "BDF"),
    ("amici_jax", "Kvaerno5"),
    ("amici_cpp", "adjoint"),
    ("amici_cpp", "forward"),
]
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"


def summarise(results: pd.DataFrame) -> pd.DataFrame:
    """Median time and gradient error per tool, solver and tolerance."""
    results = results.assign(
        time=results["cpu_ms"].where(
            results["tool"] == "amici_cpp",
            results["time_ms"],
        ),
    )
    return (
        results.groupby(["tool", "solver", "tolerance"])
        .agg(time=("time", "median"), error=("gradient_error", "median"))
        .reset_index()
    )


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
    fig, axes = plt.subplots(1, len(PROBLEMS), figsize=(7.5, 2.7))
    for ax, (problem, title) in zip(axes, PROBLEMS.items(), strict=True):
        summary = summarise(pd.read_csv(results_dir / f"{problem}.csv"))
        for tool, solver in SERIES:
            rows = summary[
                (summary["tool"] == tool)
                & (summary["solver"] == solver)
                & (summary["error"] > 0)
            ].sort_values("tolerance")
            linestyle, marker = STYLES[solver]
            ax.plot(
                rows["error"],
                rows["time"],
                color=COLOURS[tool],
                linestyle=linestyle,
                linewidth=1.5,
                marker=marker,
                markersize=5,
                markeredgecolor="white",
                markeredgewidth=1,
                solid_capstyle="round",
            )
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.yaxis.set_major_locator(LogLocator(subs=(1.0, 2.0, 5.0)))
        ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_title(title, loc="left", color=TEXT, fontsize=9)
        ax.set_xlabel("Gradient error (relative)")
        ax.grid(color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Time per gradient (ms)")
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
    fig.legend(
        handles=handles,
        loc="lower center",
        ncol=len(SERIES) // 2,
        frameon=False,
        labelcolor=TEXT,
    )
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    return fig


if __name__ == "__main__":
    results_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "results"
    fig = plot(results_dir)
    for suffix in ("pdf", "png"):
        fig.savefig(results_dir / f"work_precision.{suffix}", dpi=300)
    print("wrote", results_dir / "work_precision.pdf")
