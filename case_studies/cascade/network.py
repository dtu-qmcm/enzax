import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from common import RESULTS_DIR
from enzax.examples import gdp_fucose

sys.path.append(str(Path(__file__).resolve().parents[1]))

from network_drawing import (  # noqa: E402
    KINETIC_COLOUR,
    LABEL_SIZE,
    MUTED,
    draw_curve,
    draw_reaction,
    draw_species,
    get_boxes,
)

MODEL = gdp_fucose.model
RAPID_COLOUR = "#eb6834"
REGULATION_COLOUR = "#52514e"
INHIBITION = "-[,widthB=0.8,lengthB=0"
TARGET_PAD = 3
XLIM = (0, 96)
YLIM = (16, 64)
SPECIES_NAMES = {
    "fucose": "Fucose",
    "fucose1p": "Fucose-1-P",
    "gdp_fucose": "GDP-fucose",
    "atp": "ATP",
    "adp": "ADP",
    "amp": "AMP",
    "gmp": "GMP",
    "gdp": "GDP",
    "gtp": "GTP",
    "pp": "PPi",
    "p": "Pi",
    "polyp": "polyP",
}
DUPLICATES = {}
POSITIONS = {
    "fucose": (5, 54),
    "fucose1p": (44, 54),
    "gdp_fucose": (88, 54),
    "atp": (10, 38),
    "adp": (38, 38),
    "amp": (19, 31),
    "gmp": (4, 24),
    "gdp": (44, 24),
    "gtp": (56, 40),
    "pp": (76, 40),
}
REACTIONS = {
    "FKP_fk": {
        "legs": ["fucose", "atp", "fucose1p", "adp"],
        "point": (24, 54),
        "direction": (1, 0),
        "label": (24, 55.5),
    },
    "FKP_gt": {
        "legs": ["fucose1p", "gtp", "gdp_fucose", "pp"],
        "point": (66, 54),
        "direction": (1, 0),
        "label": (66, 55.5),
    },
    "PPA": {
        "legs": ["pp"],
        "end": (76, 31),
        "label": (75, 35.5),
        "ha": "right",
        "va": "center",
        "side": 1,
    },
    "PPK3_A": {
        "legs": ["adp", "atp"],
        "controls": [(24, 47)],
        "label": (24, 40.6),
        "va": "center",
        "side": -1,
    },
    "PPK3_G": {
        "legs": ["gdp", "gtp"],
        "label": (48.5, 33),
        "ha": "right",
        "va": "center",
        "side": -1,
    },
    "UNKNOWN_HYDROLASE": {
        "legs": ["atp", "amp", "adp"],
        "point": (25, 38),
        "direction": (1, 0),
        "text": "Unknown",
        "label": (27, 36.5),
        "ha": "left",
        "va": "top",
    },
    "GMPK": {
        "legs": ["gmp", "atp", "gdp", "adp"],
        "point": (24, 24),
        "direction": (1, 0),
        "label": (24, 22.5),
        "va": "top",
    },
}
INHIBITION_CONTROLS = {
    ("FKP_fk", "gdp_fucose"): [(88, 63), (24, 63)],
}


def get_inhibitions():
    return {
        tuple(label.split("|")[1:])
        for label in MODEL.parameter_labelling["log_saturation_constant"]
        if label.startswith("ki|")
    }


def draw_annotations(ax):
    ax.text(
        60,
        19.5,
        "Measured species in bold",
        fontsize=LABEL_SIZE,
        color=MUTED,
        va="top",
    )
    ax.legend(
        handles=[
            Line2D([], [], color=KINETIC_COLOUR, label="Kinetic reaction"),
            Line2D([], [], color=RAPID_COLOUR, label="Rapid equilibrium"),
            Line2D(
                [],
                [],
                color=REGULATION_COLOUR,
                marker="|",
                markersize=4,
                label="Inhibition",
            ),
        ],
        loc="upper left",
        bbox_to_anchor=(59.5, 29),
        bbox_transform=ax.transData,
        frameon=False,
        fontsize=LABEL_SIZE,
        handlelength=1.5,
    )


def plot_network(ax):
    reactions = MODEL.reactions | MODEL.rapid_equilibrium_network.reactions
    missing = set(reactions) ^ set(REACTIONS)
    if missing:
        raise ValueError(f"Reactions drawn and modelled differ: {missing}")
    missing = get_inhibitions() ^ set(INHIBITION_CONTROLS)
    if missing:
        raise ValueError(f"Inhibition drawn and modelled differ: {missing}")
    draw_annotations(ax)
    texts = draw_species(
        ax,
        POSITIONS,
        DUPLICATES,
        SPECIES_NAMES,
        set(MODEL.parameter_labelling["log_conc_unbalanced"]),
    )
    for node, text in texts.items():
        if node in gdp_fucose.MEASURED_SPECIES:
            text.set_fontweight("bold")
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.axis("off")
    return texts


def draw_reactions(ax, texts):
    boxes = get_boxes(ax, texts)
    reactions = MODEL.reactions | MODEL.rapid_equilibrium_network.reactions
    labels = {}
    for reaction_id, spec in REACTIONS.items():
        rapid = reaction_id in MODEL.rapid_equilibrium_network.reactions
        labels[reaction_id] = draw_reaction(
            ax,
            reaction_id,
            reactions[reaction_id].stoichiometry,
            spec,
            POSITIONS,
            DUPLICATES,
            SPECIES_NAMES,
            boxes,
            colour=RAPID_COLOUR if rapid else KINETIC_COLOUR,
            label_colour=RAPID_COLOUR if rapid else MUTED,
            reversible=rapid,
        )
    label_boxes = get_boxes(ax, labels, TARGET_PAD)
    for (reaction_id, species), controls in INHIBITION_CONTROLS.items():
        label_box = label_boxes[reaction_id]
        draw_curve(
            ax,
            [
                POSITIONS[species],
                *controls,
                label_box.get_points().mean(axis=0),
            ],
            (boxes[species], label_box),
            REGULATION_COLOUR,
            INHIBITION,
        )


def main():
    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(figsize=(3.75, 2.0), layout="constrained")
    texts = plot_network(ax)
    fig.canvas.draw()
    fig.set_layout_engine("none")
    draw_reactions(ax, texts)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / "network.png"
    fig.savefig(path, dpi=300)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
