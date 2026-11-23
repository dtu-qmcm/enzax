"""Plot the glycolysis regulator-finding figure.

A maps the pathway's reactions, cofactors and known regulation, with PFKM's
candidate inhibitors coloured by hypothesis. B shows sedoheptulose
7-phosphate against external glucose, as predicted by the round-one fits of
the PEP and lactate hypotheses, with the measurements from rounds one and
two. C shows the design screens after rounds
one and two, for the pair of hypotheses each round leaves open. D shows each
hypothesis's cumulative log predictive density of each new round's data, from
the previous round's fits, relative to the best hypothesis. E shows how five
concentrations change under DCA, comparing the PEP and PEP-plus-lactate
hypotheses' predictions from the round-two fits and estimates from the
round-three fits with the measurements. Saves results/figure.png and
results/figure.pdf.
"""

import csv

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

from analyse import get_derived_file, get_pointwise_predictive_density
from common import (
    COLOURS,
    DESIGN,
    HELD_OUT,
    HIGH_GLUCOSE,
    HYPOTHESES,
    MEASUREMENTS_FILE,
    MODEL,
    MODELS,
    PRIOR_SD,
    REFERENCE_EXPERIMENT,
    RESULTS_DIR,
    SERIES,
    get_reference,
)
from design import GLUCOSE_MM, REFERENCE_GLUCOSE_MM, get_design_file
from fit import N_CHAIN, get_chain_file
from network import draw_reactions, plot_network

HYPOTHESIS_NAMES = {
    "none": "No effector",
    "pep": "PEP",
    "lac": "Lactate",
    "both": "PEP and lactate",
}
ROUND_CONTRASTS = {
    "high_glucose": ("pep", "lac"),
    "series": ("pep", "both"),
}
ROUND_NAMES = {
    "high_glucose": "round one",
    "series": "round two",
    "series_dca": "round three",
}
PREDICTED_ROUNDS = {
    "high_glucose": "Round two\nlow glucose",
    "series": "Round three\nDCA",
}
RESPONSE_SPECIES = "s7p_c"
MIN_GLUCOSE_MM = 0.1
RESPONSE_TICKS_MM = (0.1, 0.2, 0.3)
NESTED_HYPOTHESIS = "both"
DCA_CONTRAST = ("pep", "both")
DCA_SPECIES = {
    "f6p_c": "F6P",
    "fdp_c": "FBP",
    "pep_c": "PEP",
    "pyr_c": "Pyruvate",
    "lac_c": "Lactate",
}
DCA_CANDIDATE = "DCA x3"
CONTROL_CANDIDATE = f"Glucose {REFERENCE_GLUCOSE_MM:g} mM"
BEFORE_ALPHA = 0.35
LACTATE_SITE = "dc|PFKM|lac_c"
PHASE_COLOURS = {
    "high_glucose": "#a9c9f0",
    "series": "#5b9be0",
    "series_dca": "#1d5aa6",
}
PRIOR_COLOUR = "#8a8984"
TRAINING_COLOUR = "#efeeeb"
TEXT = "#0b0b0b"
MUTED = "#52514e"
GRID = "#e4e3df"
HEATMAP = "Blues"
N_BIN = 40
DODGE = 0.06
DCA_OFFSET = 0.15
M_TO_MM = 1000


def style(ax):
    ax.grid(color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


def get_glucose_mm(experiment):
    return REFERENCE_GLUCOSE_MM * DESIGN[experiment].get("glc_e", 1.0)


def load_measured(species, experiments):
    with MEASUREMENTS_FILE.open() as f:
        return {
            row["experiment"]: float(row["value"])
            for row in csv.DictReader(f)
            if row["kind"] == "conc"
            and row["id"] == species
            and row["experiment"] in experiments
        }


def plot_glucose_response(ax, design):
    glucose = np.array([mm for mm in GLUCOSE_MM if mm >= MIN_GLUCOSE_MM])
    candidates = list(design["candidates"])
    rows = [candidates.index(f"Glucose {mm:g} mM") for mm in glucose]
    column = list(design["observables"]).index(RESPONSE_SPECIES)
    ax.axvspan(
        min(map(get_glucose_mm, HIGH_GLUCOSE)),
        max(map(get_glucose_mm, HIGH_GLUCOSE)),
        color=TRAINING_COLOUR,
        linewidth=0,
    )
    for hypothesis in ROUND_CONTRASTS["high_glucose"]:
        conc = M_TO_MM * np.exp(
            design[f"predictions|{hypothesis}"][:, rows, column],
        )
        low, median, high = np.nanquantile(conc, [0.05, 0.5, 0.95], axis=0)
        ax.fill_between(
            glucose,
            low,
            high,
            color=COLOURS[hypothesis],
            alpha=0.2,
            linewidth=0,
        )
        ax.plot(
            glucose,
            median,
            color=COLOURS[hypothesis],
            linewidth=2,
            label=f"{HYPOTHESIS_NAMES[hypothesis]}, round-one fit",
        )
    measured = load_measured(RESPONSE_SPECIES, SERIES)
    for experiments, fill, name in (
        (HIGH_GLUCOSE, TEXT, "one"),
        (tuple(e for e in SERIES if e not in HIGH_GLUCOSE), "white", "two"),
    ):
        ax.plot(
            [get_glucose_mm(e) for e in experiments],
            [M_TO_MM * measured[e] for e in experiments],
            "o",
            markersize=5,
            markerfacecolor=fill,
            markeredgecolor=TEXT,
            label=f"Measured, round {name}",
        )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xticks(glucose, [f"{mm:g}" for mm in glucose])
    ax.set_yticks(RESPONSE_TICKS_MM, [f"{y:g}" for y in RESPONSE_TICKS_MM])
    ax.minorticks_off()
    ax.set_xlabel("External glucose (mM)")
    ax.set_ylabel("Sedoheptulose 7-phosphate (mM)")
    ax.legend(
        frameon=False,
        loc="lower right",
        fontsize=6,
        handlelength=1.2,
        borderaxespad=0.2,
    )
    style(ax)


def plot_discrimination(ax, designs):
    candidates = list(designs["high_glucose"]["candidates"])
    columns = []
    for phase, (a, b) in ROUND_CONTRASTS.items():
        design = designs[phase]
        pairs = list(design["pairs"])
        columns.append(design["discrimination"][:, pairs.index(f"{a}-{b}")])
    discrimination = np.stack(columns, axis=1)
    ax.imshow(
        np.log10(1 + discrimination),
        cmap=HEATMAP,
        aspect="auto",
        vmin=0,
    )
    for (i, j), value in np.ndenumerate(discrimination):
        ax.text(
            j,
            i,
            f"{value:.0f}" if value >= 1 else f"{value:.1f}",
            ha="center",
            va="center",
            fontsize=6,
            color="white"
            if np.log10(1 + value) > 0.6 * np.log10(1 + discrimination.max())
            else TEXT,
        )
    ax.set_xticks(
        range(len(ROUND_CONTRASTS)),
        [
            f"After {ROUND_NAMES[phase]}\n"
            f"{HYPOTHESIS_NAMES[a]} vs\n{HYPOTHESIS_NAMES[b]}"
            for phase, (a, b) in ROUND_CONTRASTS.items()
        ],
        fontsize=7,
    )
    ax.xaxis.tick_top()
    ax.set_yticks(range(len(candidates)), candidates)
    ax.tick_params(length=0)
    for side in ax.spines.values():
        side.set_visible(False)


def plot_sequential_score(ax):
    pointwise = {
        h: [
            get_pointwise_predictive_density(
                np.load(get_derived_file(f"{h}_{phase}"))[
                    "held_out_log_likelihood"
                ],
            )[0]
            for phase in PREDICTED_ROUNDS
        ]
        for h in HYPOTHESES
    }
    positions = np.arange(len(PREDICTED_ROUNDS))
    scores = {h: [] for h in HYPOTHESES}
    errors = {h: [] for h in HYPOTHESES}
    for position in positions:
        cumulative = {
            h: np.concatenate(p[: position + 1]) for h, p in pointwise.items()
        }
        best = max(HYPOTHESES, key=lambda h: cumulative[h].sum())
        for hypothesis in HYPOTHESES:
            difference = cumulative[hypothesis] - cumulative[best]
            scores[hypothesis].append(difference.sum())
            errors[hypothesis].append(
                np.sqrt(len(difference) * difference.var()),
            )
    for offset, hypothesis in zip(
        np.linspace(-DODGE, DODGE, len(HYPOTHESES)),
        HYPOTHESES,
        strict=True,
    ):
        ax.errorbar(
            positions + offset,
            scores[hypothesis],
            yerr=errors[hypothesis],
            color=COLOURS[hypothesis],
            linewidth=2,
            marker="o",
            markersize=6,
            capsize=0,
        )
        ax.annotate(
            HYPOTHESIS_NAMES[hypothesis],
            (positions[-1] + offset, scores[hypothesis][-1]),
            xytext=(10, 0),
            textcoords="offset points",
            va="center",
            color=TEXT,
        )
    ax.set_xticks(positions, list(PREDICTED_ROUNDS.values()))
    ax.set_xlim(-0.3, positions[-1] + 0.9)
    ax.set_ylabel("Cumulative log predictive density,\nrelative to the best")
    style(ax)
    ax.grid(axis="x", visible=False)


def load_parameter_draws(fit, column):
    chains = [np.load(get_chain_file(fit, chain)) for chain in range(N_CHAIN)]
    position = [str(x) for x in chains[0]["labels"]].index(column)
    return np.concatenate([c["draws"][:, position] for c in chains])


def get_prior_loc(hypothesis, kind, label):
    labels = list(MODELS[hypothesis].parameter_labelling[kind])
    return float(get_reference(hypothesis)[kind][labels.index(label)])


def plot_marginals(ax, loc, draws, offset):
    to_log10 = np.log10(np.e)
    low = min(loc - 4 * PRIOR_SD, *(d.min() for d, _, _ in draws))
    high = max(loc + 4 * PRIOR_SD, *(d.max() for d, _, _ in draws))
    grid = np.linspace(low, high, 200)
    prior = np.exp(-0.5 * ((grid - loc) / PRIOR_SD) ** 2) / (
        PRIOR_SD * np.sqrt(2 * np.pi)
    )
    ax.plot(
        grid * to_log10 + offset,
        prior / to_log10,
        color=PRIOR_COLOUR,
        linestyle="--",
        linewidth=1.5,
        label="Prior",
    )
    bins = np.linspace(low, high, N_BIN) * to_log10 + offset
    for values, name, colour in draws:
        ax.hist(
            values * to_log10 + offset,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2,
            color=colour,
            label=name,
        )
    ax.set_ylabel("Density")
    style(ax)


def plot_lactate_site(ax):
    column = f"log_saturation_constant[{LACTATE_SITE}]"
    plot_marginals(
        ax,
        get_prior_loc(
            NESTED_HYPOTHESIS,
            "log_saturation_constant",
            LACTATE_SITE,
        ),
        [
            (
                load_parameter_draws(f"{NESTED_HYPOTHESIS}_{phase}", column),
                f"After {ROUND_NAMES[phase]}",
                colour,
            )
            for phase, colour in PHASE_COLOURS.items()
        ],
        offset=3,
    )
    ax.set_xlabel("PFKM lactate dissociation constant\n(log10 mM)")
    ax.legend(frameon=False, loc="upper left")


def get_percent_change(log_ratio):
    return 100 * (np.exp(log_ratio) - 1)


def get_dca_changes(design, hypothesis):
    candidates = list(design["candidates"])
    observables = list(design["observables"])
    predictions = design[f"predictions|{hypothesis}"]
    dca, control = (
        predictions[:, candidates.index(candidate)]
        for candidate in (DCA_CANDIDATE, CONTROL_CANDIDATE)
    )
    derived = np.load(get_derived_file(f"{hypothesis}_series_dca"))
    before, after = {}, {}
    for species in DCA_SPECIES:
        observable = observables.index(species)
        before[species] = get_percent_change(
            dca[:, observable] - control[:, observable],
        )
        position = MODEL.species.index(species)
        after[species] = get_percent_change(
            np.log(derived["conc|dca"][:, position])
            - np.log(derived[f"conc|{REFERENCE_EXPERIMENT}"][:, position]),
        )
    return before, after


def plot_dca_response(ax, design):
    rows = np.arange(len(DCA_SPECIES))[::-1]
    offsets = np.linspace(DCA_OFFSET, -DCA_OFFSET, len(DCA_CONTRAST))
    for offset, hypothesis in zip(offsets, DCA_CONTRAST, strict=True):
        for changes, alpha in zip(
            get_dca_changes(design, hypothesis),
            (BEFORE_ALPHA, 1.0),
            strict=True,
        ):
            for row, species in zip(rows, DCA_SPECIES, strict=True):
                low, median, high = np.quantile(
                    changes[species],
                    [0.05, 0.5, 0.95],
                )
                ax.plot(
                    [low, high],
                    [row + offset] * 2,
                    color=COLOURS[hypothesis],
                    alpha=alpha,
                    linewidth=2,
                    solid_capstyle="round",
                )
                ax.plot(
                    median,
                    row + offset,
                    "o",
                    color=COLOURS[hypothesis],
                    alpha=alpha,
                    markersize=5,
                )
    measured = [
        load_measured(species, (REFERENCE_EXPERIMENT, "dca"))
        for species in DCA_SPECIES
    ]
    ax.plot(
        [
            get_percent_change(np.log(m["dca"] / m[REFERENCE_EXPERIMENT]))
            for m in measured
        ],
        rows,
        "D",
        color=TEXT,
        markersize=4,
        zorder=3,
    )
    handles = [
        Line2D([], [], color=COLOURS[h], marker="o", linewidth=2)
        for h in DCA_CONTRAST
    ] + [
        Line2D([], [], color=MUTED, marker="o", alpha=BEFORE_ALPHA),
        Line2D([], [], color=MUTED, marker="o"),
        Line2D([], [], color=TEXT, marker="D", linestyle="none", markersize=4),
    ]
    ax.legend(
        handles,
        [
            *(HYPOTHESIS_NAMES[h] for h in DCA_CONTRAST),
            "Before DCA (round-two fit)",
            "After DCA (round-three fit)",
            "Measured",
        ],
        frameon=False,
        loc="lower right",
        fontsize=6,
        handlelength=1.2,
        borderaxespad=0.2,
    )
    ax.axvline(0, color=MUTED, linewidth=0.6)
    ax.set_yticks(rows, list(DCA_SPECIES.values()))
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Change under DCA (%)")
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
    designs = {phase: np.load(get_design_file(phase)) for phase in HELD_OUT}
    fig = plt.figure(figsize=(7.5, 8.7), layout="constrained")
    top, bottom = fig.subfigures(2, 1, height_ratios=[5.4, 3.3])
    network, side = top.subfigures(1, 2, width_ratios=[4.5, 3.0])
    network_ax = network.subplots()
    network_texts = plot_network(network_ax)
    response_ax, discrimination_ax = side.subplots(2, 1)
    score_ax, dca_ax = bottom.subplots(1, 2)
    plot_glucose_response(response_ax, designs["high_glucose"])
    plot_discrimination(discrimination_ax, designs)
    plot_sequential_score(score_ax)
    plot_dca_response(dca_ax, designs["series"])
    for ax, letter in zip(
        [network_ax, response_ax, discrimination_ax, score_ax, dca_ax],
        "ABCDE",
        strict=True,
    ):
        ax.set_title(letter, loc="left", fontweight="bold")
    fig.canvas.draw()
    fig.set_layout_engine("none")
    draw_reactions(network_ax, network_texts)
    for suffix in ("png", "pdf"):
        path = RESULTS_DIR / f"figure.{suffix}"
        fig.savefig(path, dpi=300)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
