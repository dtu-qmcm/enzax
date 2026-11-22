import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common import COLOURS, HYPOTHESES, MODEL, REGULATED_REACTION, RESULTS_DIR

sys.path.append(str(Path(__file__).resolve().parents[1]))

from network_drawing import (  # noqa: E402
    CLAMPED_COLOUR,
    KINETIC_COLOUR,
    LABEL_SIZE,
    MUTED,
    draw_cell,
    draw_curve,
    draw_reaction,
    draw_species,
    get_boxes,
)

REGULATION_COLOUR = "#8a8984"
CELL = (5, 2.5, 59.5, 77.5)
XLIM = (0, 66.5)
YLIM = (0, 82)
TARGET_PAD = 3
INHIBITION = "-[,widthB=0.8,lengthB=0"
ACTIVATION = "-|>,head_length=0.8,head_width=0.45"
EXPRESSIONS = {
    "tense_state_expression": INHIBITION,
    "dead_end_states_expression": INHIBITION,
    "relaxed_state_expression": ACTIVATION,
}
SPECIES_NAMES = {
    "glc_e": "Glc",
    "glc_c": "Glc",
    "g6p_c": "G6P",
    "f6p_c": "F6P",
    "fdp_c": "FDP",
    "dhap_c": "DHAP",
    "g3p_c": "GA3P",
    "dpg_c": "1,3-BPG",
    "p3g_c": "3PG",
    "p2g_c": "2PG",
    "pep_c": "PEP",
    "pyr_c": "Pyr",
    "lac_c": "Lac",
    "lac_e": "Lac",
    "pgl6_c": "6PGL",
    "pgc6_c": "6PG",
    "ru5p_c": "Ru5P",
    "r5p_c": "R5P",
    "xu5p_c": "X5P",
    "e4p_c": "E4P",
    "s7p_c": "S7P",
    "atp_c": "ATP",
    "adp_c": "ADP",
    "pi_c": "Pi",
    "nad_c": "NAD⁺",
    "nadh_c": "NADH",
    "nadp_c": "NADP⁺",
    "nadph_c": "NADPH",
    "co2_c": "CO₂",
    "gdp_c": "G1,6BP",
    "f26bp_c": "F2,6BP",
}
DUPLICATES = {"f6p_ta": "f6p_c", "lac_pfkl": "lac_c"}
POSITIONS = {
    "glc_e": (2.5, 77),
    "glc_c": (24, 77),
    "g6p_c": (24, 69),
    "f6p_c": (24, 62),
    "fdp_c": (24, 52),
    "dhap_c": (14, 43),
    "g3p_c": (24, 43),
    "dpg_c": (24, 37),
    "p3g_c": (24, 31.5),
    "p2g_c": (24, 26),
    "pep_c": (24, 20.5),
    "pyr_c": (24, 13.5),
    "lac_c": (24, 6.5),
    "lac_e": (2.5, 6.5),
    "pgl6_c": (38, 69),
    "pgc6_c": (52, 69),
    "ru5p_c": (52, 62),
    "r5p_c": (56.5, 57),
    "xu5p_c": (44, 62),
    "e4p_c": (41, 56),
    "f6p_ta": (47, 54),
    "s7p_c": (49, 46),
    "gdp_c": (29.5, 77.2),
    "f26bp_c": (18.5, 61.8),
    "lac_pfkl": (33, 53.5),
}
REACTIONS = {
    "GLUT4": {"legs": ["glc_e", "glc_c"], "label": (13, 77.4)},
    "HEX1": {
        "legs": ["glc_c", "g6p_c"],
        "offset": -0.8,
        "label": (19.8, 73),
        "ha": "right",
        "va": "center",
        "side": -1,
    },
    "HEX2": {
        "legs": ["glc_c", "g6p_c"],
        "offset": 0.8,
        "label": (28.2, 73),
        "ha": "left",
        "va": "center",
        "side": 1,
    },
    "PGI": {
        "legs": ["g6p_c", "f6p_c"],
        "label": (22.8, 65.5),
        "ha": "right",
        "va": "center",
    },
    "PFKM": {
        "legs": ["f6p_c", "fdp_c"],
        "controls": [(18.5, 60), (18.5, 54)],
        "label": (19.2, 57),
        "ha": "right",
        "va": "center",
        "side": 1,
    },
    "PFKL": {
        "legs": ["f6p_c", "fdp_c"],
        "controls": [(29.5, 60), (29.5, 54)],
        "label": (28.8, 57),
        "ha": "left",
        "va": "center",
        "side": -1,
    },
    "FBA": {
        "legs": ["fdp_c", "g3p_c", "dhap_c"],
        "point": (24, 47.5),
        "direction": (0, -1),
        "leg_controls": {"dhap_c": [(14, 46.5)]},
        "label": (24.8, 47.5),
        "ha": "left",
        "va": "center",
    },
    "TPI": {"legs": ["dhap_c", "g3p_c"], "label": (19, 43.6)},
    "GAPD": {
        "legs": ["g3p_c", "dpg_c"],
        "label": (25.2, 40),
        "ha": "left",
        "va": "center",
        "side": -1,
    },
    "PGK": {
        "legs": ["dpg_c", "p3g_c"],
        "label": (22.8, 34.25),
        "ha": "right",
        "va": "center",
        "side": 1,
    },
    "PGM": {
        "legs": ["p3g_c", "p2g_c"],
        "label": (22.8, 28.75),
        "ha": "right",
        "va": "center",
    },
    "ENO": {
        "legs": ["p2g_c", "pep_c"],
        "label": (22.8, 23.25),
        "ha": "right",
        "va": "center",
    },
    "PKM1": {
        "legs": ["pep_c", "pyr_c"],
        "offset": -0.8,
        "label": (19.8, 17),
        "ha": "right",
        "va": "center",
        "side": -1,
    },
    "PKM2": {
        "legs": ["pep_c", "pyr_c"],
        "offset": 0.8,
        "label": (28.2, 17),
        "ha": "left",
        "va": "center",
        "side": 1,
    },
    "LDHA": {
        "legs": ["pyr_c", "lac_c"],
        "label": (22.8, 10),
        "ha": "right",
        "va": "center",
        "side": 1,
    },
    "pyr_drain": {
        "legs": ["pyr_c"],
        "end": (31, 13.5),
        "label": (31.5, 13.5),
        "ha": "left",
        "va": "center",
    },
    "lac_transport": {
        "legs": ["lac_c", "lac_e"],
        "label": (13, 5.9),
        "va": "top",
    },
    "G6PDH": {
        "legs": ["g6p_c", "pgl6_c"],
        "label": (31, 69.6),
        "side": -1,
    },
    "PGL": {"legs": ["pgl6_c", "pgc6_c"], "label": (45, 69.6)},
    "GND": {
        "legs": ["pgc6_c", "ru5p_c"],
        "label": (50.8, 65.5),
        "ha": "right",
        "va": "center",
        "side": 1,
    },
    "RPI": {
        "legs": ["ru5p_c", "r5p_c"],
        "label": (55.5, 60.5),
        "ha": "left",
    },
    "RPE": {"legs": ["ru5p_c", "xu5p_c"], "label": (48, 62.6)},
    "TKT2": {
        "legs": ["xu5p_c", "e4p_c", "f6p_c", "g3p_c"],
        "point": (37.5, 62),
        "stem": 3,
        "direction": (-1, 0),
        "leg_controls": {"g3p_c": [(36, 50), (29, 45)]},
        "label": (37.5, 62.6),
    },
    "TALA": {
        "legs": ["e4p_c", "f6p_ta", "s7p_c", "g3p_c"],
        "point": (41, 51),
        "direction": (0, -1),
        "label": (41.8, 51),
        "ha": "left",
        "va": "center",
    },
    "TKT1": {
        "legs": ["r5p_c", "xu5p_c", "s7p_c", "g3p_c"],
        "point": (55, 50),
        "direction": (-0.3, -1),
        "leg_controls": {
            "xu5p_c": [(54, 55), (48, 60)],
            "g3p_c": [(52, 37), (30, 43)],
        },
        "label": (55.8, 49.5),
        "ha": "left",
    },
    "r5p_drain": {
        "legs": ["r5p_c"],
        "end": (61, 57),
        "label": (57.6, 56.3),
        "ha": "left",
        "va": "top",
    },
}
REGULATION = {
    ("HEX2", "gdp_c"): {"start": "gdp_c"},
    ("PFKM", "f26bp_c"): {"start": "f26bp_c"},
    ("PFKL", "lac_c"): {"start": "lac_pfkl"},
    ("PFKM", "lac_c"): {"start": "lac_c", "controls": [(3, 6.5), (7, 65)]},
    ("PFKM", "pep_c"): {"start": "pep_c", "controls": [(6, 20.5), (12, 53)]},
}
CANDIDATE_COLOURS = {
    effectors[0]: COLOURS[hypothesis]
    for hypothesis, effectors in HYPOTHESES.items()
    if len(effectors) == 1
}


def get_effectors(reaction):
    effectors = {}
    for attribute, style in EXPRESSIONS.items():
        expression = getattr(reaction, attribute, None)
        if expression is None:
            continue
        for term in expression.terms:
            for factor in term.factors:
                for species, _ in factor.species:
                    if species not in reaction.stoichiometry:
                        effectors[species] = style
    return effectors


def draw_regulation(ax, boxes, label_boxes):
    for reaction_id, reaction in MODEL.reactions.items():
        for species, style in get_effectors(reaction).items():
            spec = REGULATION[reaction_id, species]
            label_box = label_boxes[reaction_id]
            candidate = (
                reaction_id == REGULATED_REACTION
                and species in CANDIDATE_COLOURS
            )
            draw_curve(
                ax,
                [
                    POSITIONS[spec["start"]],
                    *spec.get("controls", []),
                    label_box.get_points().mean(axis=0),
                ],
                (boxes[spec["start"]], label_box),
                CANDIDATE_COLOURS[species] if candidate else REGULATION_COLOUR,
                style,
                "--" if candidate else "-",
                fill=style != ACTIVATION,
            )


def draw_annotations(ax):
    draw_cell(ax, CELL, XLIM, "Medium")
    ax.text(
        34.5,
        19.5,
        "F6P and Lac appear twice",
        fontsize=LABEL_SIZE,
        color=MUTED,
        va="top",
    )
    ax.legend(
        handles=[
            Line2D([], [], color=KINETIC_COLOUR, label="Reaction"),
            Line2D(
                [],
                [],
                color=REGULATION_COLOUR,
                marker=">",
                markersize=4,
                markerfacecolor="white",
                label="Activation",
            ),
            Line2D(
                [],
                [],
                color=REGULATION_COLOUR,
                marker="|",
                markersize=4,
                label="Inhibition",
            ),
            Line2D(
                [],
                [],
                color=CANDIDATE_COLOURS["pep_c"],
                linestyle="--",
                label="Candidate inhibitor: PEP",
            ),
            Line2D(
                [],
                [],
                color=CANDIDATE_COLOURS["lac_c"],
                linestyle="--",
                label="Candidate inhibitor: lactate",
            ),
            Line2D(
                [],
                [],
                color=CLAMPED_COLOUR,
                linestyle="none",
                marker="s",
                markersize=3,
                label="Clamped species",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(34, 33),
        bbox_transform=ax.transData,
        frameon=False,
        fontsize=LABEL_SIZE,
        handlelength=1.5,
    )


def plot_network(ax):
    missing = set(MODEL.reactions) ^ set(REACTIONS)
    if missing:
        raise ValueError(f"Reactions drawn and modelled differ: {missing}")
    regulation = {
        (reaction_id, species)
        for reaction_id, reaction in MODEL.reactions.items()
        for species in get_effectors(reaction)
    }
    missing = regulation ^ set(REGULATION)
    if missing:
        raise ValueError(f"Regulation drawn and modelled differ: {missing}")
    draw_annotations(ax)
    texts = draw_species(
        ax,
        POSITIONS,
        DUPLICATES,
        SPECIES_NAMES,
        set(MODEL.parameter_labelling["log_conc_unbalanced"]),
    )
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.axis("off")
    return texts


def draw_reactions(ax, texts):
    boxes = get_boxes(ax, texts)
    labels = {
        reaction_id: draw_reaction(
            ax,
            reaction_id,
            MODEL.reactions[reaction_id].stoichiometry,
            spec,
            POSITIONS,
            DUPLICATES,
            SPECIES_NAMES,
            boxes,
        )
        for reaction_id, spec in REACTIONS.items()
    }
    draw_regulation(ax, boxes, get_boxes(ax, labels, TARGET_PAD))


def main():
    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(figsize=(4.5, 5.4), layout="constrained")
    texts = plot_network(ax)
    fig.canvas.draw()
    fig.set_layout_engine("none")
    draw_reactions(ax, texts)
    path = RESULTS_DIR / "network.png"
    fig.savefig(path, dpi=300)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
