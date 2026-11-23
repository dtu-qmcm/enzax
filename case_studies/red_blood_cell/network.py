"""Draw the red cell network for panel A of the red cell figure.

Species and reactions sit at hand-placed positions, with rapid equilibria in
orange. Drawing a network whose reactions differ from the improved model's
raises an error. Running the module directly saves results/network.png.
"""

import sys
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from enzax.examples import red_blood_cell_volume as rbcv
from ph_sweep import RESULTS_DIR

sys.path.append(str(Path(__file__).resolve().parents[1]))

from network_drawing import (  # noqa: E402
    CLAMPED_COLOUR,
    KINETIC_COLOUR,
    LABEL_SIZE,
    MUTED,
    draw_cell,
    draw_reaction,
    draw_species,
    get_boxes,
)

MODEL = rbcv.improved
RAPID_COLOUR = "#eb6834"
CELL = (4, 2.5, 88, 48.5)
XLIM = (0, 100)
YLIM = (0, 53)
SPECIES_NAMES = {
    "g6p": "G6P",
    "f6p": "F6P",
    "fdp": "FDP",
    "dhap": "DHAP",
    "ga3p": "GA3P",
    "dpg13": "1,3-DPG",
    "pg3": "3PG",
    "pg2": "2PG",
    "pep": "PEP",
    "pyr": "Pyr",
    "lac": "Lac",
    "dpg23": "2,3-DPG",
    "nad": "NAD⁺",
    "nadh": "NADH",
    "ado": "Ado",
    "amp": "AMP",
    "adp": "ADP",
    "atp": "ATP",
    "gl6p": "GL6P",
    "go6p": "GO6P",
    "nadp": "NADP⁺",
    "nadph": "NADPH",
    "gsh": "GSH",
    "gssg": "GSSG",
    "ru5p": "Ru5P",
    "r5p": "R5P",
    "x5p": "X5P",
    "s7p": "S7P",
    "e4p": "E4P",
    "prpp": "PRPP",
    "imp": "IMP",
    "ino": "Ino",
    "hx": "Hx",
    "r1p": "R1P",
    "na": "Na⁺",
    "mg": "Mg²⁺",
    "mgatp": "MgATP",
    "mgadp": "MgADP",
    "mgamp": "MgAMP",
    "pi": "Pi",
    "co2": "CO₂",
    "ade": "Ade",
    "k": "K⁺",
    "cl": "Cl⁻",
    "hco3": "HCO₃⁻",
    "pyr_e": "Pyr",
    "lac_e": "Lac",
    "na_e": "Na⁺",
    "k_e": "K⁺",
    "cl_e": "Cl⁻",
    "hco3_e": "HCO₃⁻",
}

DUPLICATES = {"f6p_ta": "f6p"}
POSITIONS = {
    "g6p": (16, 47),
    "f6p": (16, 42.5),
    "fdp": (16, 38),
    "ga3p": (16, 33),
    "dpg13": (16, 28.5),
    "pg3": (16, 23),
    "pg2": (16, 18.5),
    "pep": (16, 14),
    "pyr": (16, 9.5),
    "lac": (16, 5.5),
    "dhap": (9, 35),
    "dpg23": (9, 25.75),
    "pyr_e": (1.8, 9.5),
    "lac_e": (1.8, 5.5),
    "gl6p": (25, 47),
    "go6p": (34, 47),
    "ru5p": (43, 47),
    "r5p": (51, 43.5),
    "x5p": (35, 43),
    "e4p": (26, 39.5),
    "f6p_ta": (31.5, 40),
    "s7p": (36, 35.5),
    "prpp": (59, 47),
    "ade": (66, 49.5),
    "amp": (73, 45),
    "atp": (87, 45),
    "adp": (80, 39.5),
    "mgamp": (73, 49.5),
    "mgatp": (87, 49.5),
    "mgadp": (80, 35.5),
    "imp": (64, 39),
    "ino": (64, 31),
    "ado": (73, 31),
    "hx": (56, 33),
    "r1p": (54, 28),
    "gsh": (42, 8),
    "gssg": (52, 8),
    "cl": (87, 29),
    "cl_e": (97, 29),
    "hco3": (87, 25),
    "hco3_e": (97, 25),
    "na": (87, 15),
    "k_e": (97, 15),
    "na_e": (97, 8),
    "k": (87, 8),
}
REACTIONS = {
    "HK": {
        "legs": ["g6p"],
        "start": (16, 51),
        "label": (17, 49.8),
        "ha": "left",
        "va": "center",
        "side": -1,
    },
    "PGI": {"legs": ["g6p", "f6p"], "label": (15, 44.75), "ha": "right"},
    "PFK": {
        "legs": ["f6p", "fdp"],
        "label": (17, 40.25),
        "ha": "left",
        "va": "center",
        "side": -1,
    },
    "ALD": {
        "legs": ["fdp", "ga3p", "dhap"],
        "point": (16, 35.9),
        "direction": (0, -1),
        "label": (12.7, 36.8),
    },
    "TPI": {"legs": ["dhap", "ga3p"], "bend": -1.2, "label": (11.5, 32.6)},
    "GAPDH": {
        "legs": ["ga3p", "dpg13"],
        "label": (15, 30.75),
        "ha": "right",
        "va": "center",
        "side": 1,
    },
    "PGK": {
        "legs": ["dpg13", "pg3"],
        "label": (20, 25.75),
        "ha": "left",
        "va": "center",
        "side": 1,
    },
    "DPGM": {"legs": ["dpg13", "dpg23"], "bend": 1.0, "label": (11, 28.6)},
    "DPGase": {
        "legs": ["dpg23", "pg3"],
        "bend": 1.0,
        "label": (10, 23.2),
        "va": "top",
        "side": -1,
    },
    "PGM": {"legs": ["pg3", "pg2"], "label": (17, 20.75), "ha": "left"},
    "EN": {"legs": ["pg2", "pep"], "label": (17, 16.25), "ha": "left"},
    "PK": {
        "legs": ["pep", "pyr"],
        "label": (15, 11.75),
        "ha": "right",
        "va": "center",
        "side": 1,
    },
    "LDH": {
        "legs": ["pyr", "lac"],
        "label": (15, 7.5),
        "ha": "right",
        "va": "center",
        "side": 1,
    },
    "PYR_ex": {"legs": ["pyr", "pyr_e"], "label": (9.5, 10.1)},
    "LAC_ex": {"legs": ["lac", "lac_e"], "label": (9.5, 6.1)},
    "G6PDH": {
        "legs": ["g6p", "gl6p"],
        "label": (20.5, 47.6),
        "side": -1,
    },
    "PGLase": {"legs": ["gl6p", "go6p"], "label": (29.5, 46.2), "va": "top"},
    "GL6PDH": {
        "legs": ["go6p", "ru5p"],
        "label": (38.5, 49.4),
        "side": 1,
    },
    "R5PI": {"legs": ["ru5p", "r5p"], "label": (48, 46)},
    "Xu5PE": {"legs": ["ru5p", "x5p"], "label": (40.5, 44.4), "va": "top"},
    "TKI": {
        "legs": ["r5p", "x5p", "s7p", "ga3p"],
        "point": (44, 38),
        "direction": (-0.4, -1),
        "label": (45, 37.5),
        "ha": "left",
    },
    "TA": {
        "legs": ["s7p", "ga3p", "e4p", "f6p_ta"],
        "point": (29, 37),
        "direction": (0, 1),
        "label": (28, 37.2),
        "ha": "right",
        "va": "center",
    },
    "TKII": {
        "legs": ["x5p", "e4p", "f6p", "ga3p"],
        "point": (23, 41.5),
        "direction": (-1, 0),
        "label": (23.5, 42.3),
    },
    "PRPPsyn": {
        "legs": ["r5p", "prpp"],
        "label": (54, 47.6),
        "side": 1,
    },
    "AdPRT": {
        "legs": ["prpp", "ade", "amp"],
        "point": (68, 46.3),
        "direction": (1, -0.2),
        "label": (68, 45.5),
        "va": "top",
    },
    "HGPRT": {
        "legs": ["prpp", "hx", "imp"],
        "point": (61, 40),
        "direction": (1, 0),
        "label": (60.5, 40.5),
        "ha": "right",
    },
    "AMPDA": {"legs": ["amp", "imp"], "label": (67, 41.6)},
    "IMPase": {
        "legs": ["imp", "ino"],
        "label": (62.8, 35),
        "ha": "right",
        "va": "center",
        "side": 1,
    },
    "AMPase": {
        "legs": ["amp", "ado"],
        "offset": 0.8,
        "label": (76, 38),
        "ha": "left",
        "va": "center",
        "side": 1,
    },
    "AK": {
        "legs": ["ado", "amp"],
        "offset": 0.8,
        "label": (68.8, 38),
        "ha": "right",
        "va": "center",
        "side": 1,
    },
    "ADA": {"legs": ["ado", "ino"], "label": (68.5, 30.4), "va": "top"},
    "PNPase": {
        "legs": ["ino", "hx", "r1p"],
        "point": (60, 31),
        "direction": (-1, 0),
        "label": (60.5, 30.3),
        "va": "top",
        "side": -1,
    },
    "PRM": {"legs": ["r1p", "r5p"], "label": (52.3, 38), "ha": "right"},
    "HX_ex": {"legs": ["hx"], "end": (54.8, 36), "label": (55.6, 36.6)},
    "ApK": {
        "legs": ["adp", "amp", "atp"],
        "point": (80, 42.5),
        "direction": (0, 1),
        "label": (80, 45.8),
    },
    "ATPase": {
        "legs": ["atp", "adp"],
        "label": (84.5, 41.5),
        "ha": "left",
        "side": 1,
    },
    "MgATP": {
        "legs": ["atp", "mgatp"],
        "text": "Mg²⁺",
        "label": (87.8, 47.25),
        "ha": "left",
    },
    "MgAMP": {
        "legs": ["amp", "mgamp"],
        "text": "Mg²⁺",
        "label": (73.8, 47.25),
        "ha": "left",
    },
    "MgADP": {
        "legs": ["adp", "mgadp"],
        "text": "Mg²⁺",
        "label": (80.8, 37.5),
        "ha": "left",
    },
    "GSSGR": {
        "legs": ["gssg", "gsh"],
        "offset": 0.8,
        "label": (47, 4.8),
        "va": "top",
        "side": 1,
    },
    "GSHox": {"legs": ["gsh", "gssg"], "offset": 0.8, "label": (47, 9.1)},
    "pump": {
        "legs": ["na", "k_e", "na_e", "k"],
        "point": (92, 11.5),
        "direction": (0, -1),
        "side": -1,
        "label": (85, 11.5),
        "ha": "right",
        "va": "center",
    },
    "leak_Na": {
        "legs": ["na_e", "na"],
        "controls": [(102, 11), (98, 21.5)],
        "label": (95.5, 20.6),
    },
    "Cl_ex": {
        "legs": ["cl_e", "cl"],
        "label": (84.5, 29),
        "ha": "right",
        "va": "center",
    },
    "HCO3_ex": {
        "legs": ["hco3_e", "hco3"],
        "label": (84.5, 25),
        "ha": "right",
        "va": "center",
    },
}


def draw_annotations(ax):
    draw_cell(ax, CELL, XLIM, "Plasma")
    ax.text(
        25,
        23,
        "Osmotic balance sets the cell volume\n"
        "Electroneutrality sets the membrane potential\n"
        "Moieties: NAD⁺ + NADH, NADP⁺ + NADPH, GSH + 2 GSSG, total Mg²⁺\n"
        "F6P appears twice",
        fontsize=LABEL_SIZE,
        color=MUTED,
        va="top",
        linespacing=1.3,
    )
    ax.legend(
        handles=[
            Line2D([], [], color=KINETIC_COLOUR, label="Kinetic reaction"),
            Line2D([], [], color=RAPID_COLOUR, label="Rapid equilibrium"),
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
        bbox_to_anchor=(24.5, 16),
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
    reactions = MODEL.reactions | MODEL.rapid_equilibrium_network.reactions
    for reaction_id, spec in REACTIONS.items():
        rapid = reaction_id in MODEL.rapid_equilibrium_network.reactions
        draw_reaction(
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


def main():
    plt.rcParams.update({"font.size": 8})
    fig, ax = plt.subplots(figsize=(7.5, 3.6), layout="constrained")
    texts = plot_network(ax)
    fig.canvas.draw()
    fig.set_layout_engine("none")
    draw_reactions(ax, texts)
    path = RESULTS_DIR / "network.png"
    fig.savefig(path, dpi=300)
    print(f"Saved {path}")


if __name__ == "__main__":
    main()
