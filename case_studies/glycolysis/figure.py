"""Plot panels B to D of the glycolysis figure.

Panel A, a map of the pathway showing its reactions and PFKM's regulation,
is drawn separately. B shows each measurement's standardised residual, as
the posterior median and interquartile range, on the scale of its
measurement model: log for concentrations, linear for fluxes, and ordered
isometric log-ratio coordinates for MIDs, averaged over replicates. C shows
how much the data narrow each free parameter's prior, as one minus the ratio
of posterior to prior variance. D compares the prior and posterior of
ZeLa's glucose uptake relative to the wild type, as simulated and in
analyse.py's two counterfactual states, against the truth. Saves
results/figure.png and results/figure.pdf.
"""

import jax
import matplotlib.pyplot as plt
import numpy as np
from jax.flatten_util import ravel_pytree
from matplotlib.ticker import NullFormatter

from analyse import DERIVED_FILE
from common import (
    EXPERIMENTS,
    MODEL,
    POOL_SEPARATOR,
    RESULTS_DIR,
    get_labels,
    get_layout,
    get_prior,
    get_split,
    load_measurements,
)
from enzax.measurements import get_ordered_ilr
from fit import N_CHAIN, get_chain_file

COLOURS = {"wild_type": "#2a78d6", "zela": "#eb6834", "high_lactate": "#1baf7a"}
EXPERIMENT_NAMES = {
    "wild_type": "Wild type",
    "zela": "ZeLa",
    "high_lactate": "Wild type, high lactate",
}
PRIOR_COLOUR = "#b5b3ad"
POSTERIOR_COLOUR = "#2a78d6"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
INTERVAL = (0.05, 0.95)
INNER_INTERVAL = (0.25, 0.75)
RESIDUAL_BAND = 2.0
RESIDUAL_KINDS = {
    "conc_measurements": "Concentrations",
    "flux_measurements": "Fluxes",
    "mid_measurements": "MIDs (log-ratio coordinates)",
}
KIND_GAP = 6
PARAMETER_KINDS = {
    "log_kcat": "Turnover numbers",
    "log_enzyme": "Enzyme levels",
    "log_saturation_constant": "Saturation constants",
    "log_tc": "Transfer constants",
    "log_conc_unbalanced": "Clamped concentrations",
}
JITTER = 0.15
JITTER_SEED = 0
UPTAKE_TICKS = (0.33, 0.5, 0.67, 1, 1.5, 2, 3)
COUNTERFACTUALS = {
    "uptake_ratio": "Simulated",
    "uptake_ratio|pep_frozen": (
        "Lactate relief alone\n(PEP site as in wild type)"
    ),
    "uptake_ratio|lac_frozen": (
        "PEP inhibition alone\n(lactate site as in wild type)"
    ),
}


def load_derived():
    derived = np.load(DERIVED_FILE)
    return {
        source: {
            key.split("|", 1)[1]: derived[key]
            for key in derived.files
            if key.startswith(f"{source}|")
        }
        for source in ("truth", "posterior", "prior")
    }


def is_stable(derived):
    return np.all(
        [
            (derived[f"found|{experiment}"] == 1)
            & (derived[f"growth_rate|{experiment}"] < 0)
            for experiment in EXPERIMENTS
        ],
        axis=0,
    )


def get_residuals(posterior):
    measurements = load_measurements()
    residuals = []
    for kind in RESIDUAL_KINDS:
        for m in measurements[kind]:
            if kind == "conc_measurements":
                positions = [MODEL.species.index(s) for s in m.ids]
                predicted = posterior[f"conc|{m.experiment}"][:, positions]
                draws = (np.log(m.values) - np.log(predicted)) / m.errors
            elif kind == "flux_measurements":
                positions = [MODEL.reaction_ids.index(r) for r in m.ids]
                predicted = posterior[f"flux|{m.experiment}"][:, positions]
                draws = (m.values - predicted) / m.errors
            else:
                pool = POOL_SEPARATOR.join(m.ids)
                predicted = posterior[f"mid|{pool}|{m.experiment}"]
                measured = np.mean(
                    [get_ordered_ilr(replicate) for replicate in m.values],
                    axis=0,
                )
                error = m.errors / np.sqrt(len(m.values))
                draws = (
                    measured - jax.vmap(get_ordered_ilr)(predicted)
                ) / error
            for draw in np.asarray(draws).T:
                residuals.append((kind, m.experiment, draw))
    return residuals


def get_contraction():
    layout = get_layout()
    split = get_split(layout)
    prior = get_prior(layout, split)
    loc = {kind: leaf[0] for kind, leaf in prior.items()}
    scale = np.asarray(
        ravel_pytree({kind: leaf[1] for kind, leaf in prior.items()})[0],
    )
    draws = np.concatenate(
        [np.load(get_chain_file(chain))["draws"] for chain in range(N_CHAIN)],
    )
    contraction = 1 - (draws.std(axis=0) / scale) ** 2
    kinds = [label.split("[")[0] for label in get_labels(split, loc)]
    return {
        kind: contraction[[k == kind for k in kinds]]
        for kind in PARAMETER_KINDS
    }


def style(ax):
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def plot_residuals(ax, residuals):
    ax.axhspan(-RESIDUAL_BAND, RESIDUAL_BAND, color=GRID, alpha=0.5, zorder=0)
    ax.axhline(0, color=MUTED, linewidth=0.6, zorder=0)
    position = 0
    centres = []
    for kind in RESIDUAL_KINDS:
        start = position
        for experiment in EXPERIMENTS:
            for k, e, draws in residuals:
                if k != kind or e != experiment:
                    continue
                low, median, high = np.quantile(
                    draws,
                    [INNER_INTERVAL[0], 0.5, INNER_INTERVAL[1]],
                )
                ax.plot(
                    [position, position],
                    [low, high],
                    color=COLOURS[experiment],
                    linewidth=1.2,
                    solid_capstyle="round",
                )
                ax.plot(
                    position,
                    median,
                    "o",
                    markersize=2.5,
                    color=COLOURS[experiment],
                )
                position += 1
        centres.append((start + position - 1) / 2)
        position += KIND_GAP
    ax.set_xticks(centres, list(RESIDUAL_KINDS.values()))
    ax.tick_params(axis="x", length=0)
    ax.set_xlim(-KIND_GAP / 2, position - KIND_GAP / 2)
    ax.set_ylabel("Standardised residual")
    for experiment in EXPERIMENTS:
        ax.plot(
            [],
            [],
            "o",
            markersize=4,
            color=COLOURS[experiment],
            label=EXPERIMENT_NAMES[experiment],
        )
    ax.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1, 1),
        ncols=len(EXPERIMENTS),
    )
    style(ax)
    ax.grid(axis="x", visible=False)


def plot_contraction(ax, contraction):
    rng = np.random.default_rng(JITTER_SEED)
    positions = np.arange(len(contraction))[::-1]
    for position, values in zip(positions, contraction.values(), strict=True):
        ax.scatter(
            values,
            position + rng.uniform(-JITTER, JITTER, len(values)),
            s=10,
            color=POSTERIOR_COLOUR,
            edgecolor="white",
            linewidth=0.5,
        )
    ax.set_yticks(positions, [PARAMETER_KINDS[kind] for kind in contraction])
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.1, 1.05)
    ax.set_xlabel("Posterior contraction")
    style(ax)
    ax.grid(axis="y", visible=False)


def plot_counterfactuals(ax, derived):
    positions = np.arange(len(COUNTERFACTUALS))[::-1]
    for position, key in zip(positions, COUNTERFACTUALS, strict=True):
        prior = derived["prior"][key][is_stable(derived["prior"])]
        posterior = derived["posterior"][key]
        for draws, offset, colour in (
            (prior, 0.15, PRIOR_COLOUR),
            (posterior, -0.15, POSTERIOR_COLOUR),
        ):
            low, high = np.quantile(draws, INTERVAL)
            inner_low, inner_high = np.quantile(draws, INNER_INTERVAL)
            ax.plot(
                [low, high],
                [position + offset] * 2,
                color=colour,
                linewidth=1.5,
                solid_capstyle="round",
            )
            ax.plot(
                [inner_low, inner_high],
                [position + offset] * 2,
                color=colour,
                linewidth=4,
                solid_capstyle="round",
            )
        ax.plot(
            derived["truth"][key],
            position,
            marker="|",
            markersize=14,
            markeredgewidth=1.5,
            color=TEXT,
        )
    ax.axvline(1, color=MUTED, linewidth=0.6, zorder=0)
    ax.set_xscale("log")
    ax.set_xticks(UPTAKE_TICKS, [f"{tick:g}" for tick in UPTAKE_TICKS])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(UPTAKE_TICKS[0], UPTAKE_TICKS[-1])
    ax.set_yticks(positions, list(COUNTERFACTUALS.values()))
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("ZeLa / wild-type glucose uptake (log scale)")
    ax.plot([], [], color=PRIOR_COLOUR, linewidth=4, label="Prior")
    ax.plot([], [], color=POSTERIOR_COLOUR, linewidth=4, label="Posterior")
    ax.plot(
        [],
        [],
        color=TEXT,
        marker="|",
        linestyle="none",
        markersize=10,
        label="Truth",
    )
    ax.legend(
        frameon=False,
        loc="lower right",
        bbox_to_anchor=(1, 1),
        ncols=3,
    )
    style(ax)
    ax.grid(axis="y", visible=False)


def main():
    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.edgecolor": MUTED,
            "axes.labelcolor": TEXT,
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "legend.fontsize": 7,
        },
    )
    derived = load_derived()
    fig = plt.figure(figsize=(7.5, 5.5), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[1, 1.1])
    residual_ax = top.subplots()
    contraction_ax, counterfactual_ax = bottom.subplots(
        1,
        2,
        width_ratios=[1, 1.3],
    )
    plot_residuals(residual_ax, get_residuals(derived["posterior"]))
    plot_contraction(contraction_ax, get_contraction())
    plot_counterfactuals(counterfactual_ax, derived)
    for ax, letter in zip(
        [residual_ax, contraction_ax, counterfactual_ax],
        "BCD",
        strict=True,
    ):
        ax.set_title(letter, loc="left", fontweight="bold")
    for suffix in ("png", "pdf"):
        path = RESULTS_DIR / f"figure.{suffix}"
        fig.savefig(path, dpi=300)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
