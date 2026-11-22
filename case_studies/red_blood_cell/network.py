import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path

from enzax.examples import red_blood_cell_volume as rbcv
from ph_sweep import RESULTS_DIR

MODEL = rbcv.improved
KINETIC_COLOUR = "#0b0b0b"
RAPID_COLOUR = "#eb6834"
CLAMPED_COLOUR = "#8a8984"
MEMBRANE_COLOUR = "#2a78d6"
TEXT = "#0b0b0b"
MUTED = "#52514e"
CELL = (4, 2.5, 88, 48.5)
XLIM = (0, 100)
YLIM = (0, 53)
LABEL_SIZE = 5
SPECIES_SIZE = 6
LABEL_PAD = 1.5
FORK_CONTROL = 0.6
STEM = 2.0
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
        "label": (15, 49),
        "ha": "right",
        "inline": True,
    },
    "PGI": {"legs": ["g6p", "f6p"], "label": (15, 44.75), "ha": "right"},
    "PFK": {
        "legs": ["f6p", "fdp"],
        "label": (15, 40.25),
        "ha": "right",
        "inline": True,
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
        "label": (17, 30.6),
        "ha": "left",
        "inline": True,
    },
    "PGK": {
        "legs": ["dpg13", "pg3"],
        "label": (17, 25.75),
        "ha": "left",
        "inline": True,
    },
    "DPGM": {"legs": ["dpg13", "dpg23"], "bend": 1.0, "label": (11, 28.6)},
    "DPGase": {"legs": ["dpg23", "pg3"], "bend": 1.0, "label": (11, 22.4)},
    "PGM": {"legs": ["pg3", "pg2"], "label": (17, 20.75), "ha": "left"},
    "EN": {"legs": ["pg2", "pep"], "label": (17, 16.25), "ha": "left"},
    "PK": {
        "legs": ["pep", "pyr"],
        "label": (17, 11.75),
        "ha": "left",
        "inline": True,
    },
    "LDH": {
        "legs": ["pyr", "lac"],
        "label": (17, 7.5),
        "ha": "left",
        "inline": True,
    },
    "PYR_ex": {"legs": ["pyr", "pyr_e"], "label": (9.5, 10.1)},
    "LAC_ex": {"legs": ["lac", "lac_e"], "label": (9.5, 6.1)},
    "G6PDH": {"legs": ["g6p", "gl6p"], "label": (20.5, 46.2), "va": "top"},
    "PGLase": {"legs": ["gl6p", "go6p"], "label": (29.5, 46.2), "va": "top"},
    "GL6PDH": {"legs": ["go6p", "ru5p"], "label": (38.5, 47.8)},
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
    "PRPPsyn": {"legs": ["r5p", "prpp"], "label": (53.5, 46.5)},
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
    "IMPase": {"legs": ["imp", "ino"], "label": (63.2, 35), "ha": "right"},
    "AMPase": {
        "legs": ["amp", "ado"],
        "offset": 0.8,
        "label": (74.5, 38),
        "ha": "left",
    },
    "AK": {
        "legs": ["ado", "amp"],
        "offset": 0.8,
        "label": (71.5, 36),
        "ha": "right",
    },
    "ADA": {"legs": ["ado", "ino"], "label": (68.5, 30.4), "va": "top"},
    "PNPase": {
        "legs": ["ino", "hx", "r1p"],
        "point": (60, 31),
        "direction": (-1, 0),
        "label": (60.5, 30.3),
        "va": "top",
    },
    "PRM": {"legs": ["r1p", "r5p"], "label": (52.3, 38), "ha": "right"},
    "HX_ex": {"legs": ["hx"], "end": (54.8, 36), "label": (55.6, 36.6)},
    "ApK": {
        "legs": ["adp", "amp", "atp"],
        "point": (80, 42.5),
        "direction": (0, 1),
        "label": (80, 45.8),
    },
    "ATPase": {"legs": ["atp", "adp"], "label": (84.5, 41.5), "ha": "left"},
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
        "label": (47, 6.9),
        "va": "top",
    },
    "GSHox": {"legs": ["gsh", "gssg"], "offset": 0.8, "label": (47, 9.1)},
    "pump": {
        "legs": ["na", "k_e", "na_e", "k"],
        "point": (92, 11.5),
        "direction": (0, -1),
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


def get_species(node):
    return DUPLICATES.get(node, node)


def get_cofactor_text(stoichiometry, legs):
    shown = {get_species(node) for node in legs}
    sides = ([], [])
    for species, coefficient in stoichiometry.items():
        if species in shown:
            continue
        name = SPECIES_NAMES[species]
        if abs(coefficient) != 1:
            name = f"{abs(coefficient):g} {name}"
        sides[coefficient > 0].append(name)
    if not any(sides):
        return ""
    return " + ".join(sides[0]) + " → " + " + ".join(sides[1])


def split(points, t):
    left, right = [points[0]], [points[-1]]
    while len(points) > 1:
        points = [(1 - t) * a + t * b for a, b in zip(points, points[1:])]
        left.append(points[0])
        right.append(points[-1])
    return left, right[::-1]


def evaluate(points, t):
    return split(points, t)[0][-1]


def trim(points, start_box, end_box):
    def last_outside(box, reverse):
        low, high = 0.0, 1.0
        for _ in range(40):
            middle = (low + high) / 2
            t = 1 - middle if reverse else middle
            if box.contains(*evaluate(points, t)):
                high = middle
            else:
                low = middle
        return 1 - low if reverse else low

    a = 0.0 if start_box is None else last_outside(start_box, reverse=True)
    b = 1.0 if end_box is None else last_outside(end_box, reverse=False)
    if b <= a:
        raise ValueError("A curve lies inside a species label")
    left, _ = split(points, b)
    _, middle = split(left, a / b)
    return middle


def draw_curve(ax, points, boxes, colour, style):
    points = [np.asarray(point, dtype=float) for point in points]
    codes = {2: Path.LINETO, 3: Path.CURVE3, 4: Path.CURVE4}[len(points)]
    ax.add_patch(
        FancyArrowPatch(
            path=Path(
                trim(points, *boxes),
                [Path.MOVETO] + [codes] * (len(points) - 1),
            ),
            arrowstyle=style,
            mutation_scale=5,
            linewidth=0.7,
            color=colour,
        ),
    )


def draw_reaction(ax, reaction_id, reaction, spec, rapid, boxes):
    stoichiometry = reaction.stoichiometry
    colour = RAPID_COLOUR if rapid else KINETIC_COLOUR
    legs = spec["legs"]
    substrates = [n for n in legs if stoichiometry[get_species(n)] < 0]
    products = [n for n in legs if stoichiometry[get_species(n)] > 0]
    if "point" in spec:
        point = np.asarray(spec["point"], dtype=float)
        direction = np.asarray(spec["direction"], dtype=float)
        direction /= np.linalg.norm(direction)
        stem = direction * STEM / 2
        draw_curve(ax, [point - stem, point + stem], (None, None), colour, "-")
        for node, sign in [(n, -1) for n in substrates] + [
            (n, 1) for n in products
        ]:
            end = point + sign * stem
            position = np.asarray(POSITIONS[node], dtype=float)
            reach = max(direction @ (position - end) * sign, 0.0)
            control = end + sign * direction * reach * FORK_CONTROL
            style = "-|>" if sign > 0 or rapid else "-"
            draw_curve(
                ax,
                [end, control, position],
                (None, boxes[node]),
                colour,
                style,
            )
    else:
        start = np.asarray(
            spec.get("start", POSITIONS[substrates[0]] if substrates else None),
            dtype=float,
        )
        end = np.asarray(
            spec.get("end", POSITIONS[products[0]] if products else None),
            dtype=float,
        )
        if "offset" in spec:
            along = end - start
            normal = np.array([-along[1], along[0]]) / np.linalg.norm(along)
            start = start + normal * spec["offset"]
            end = end + normal * spec["offset"]
        if "controls" in spec:
            controls = [np.asarray(c, dtype=float) for c in spec["controls"]]
        else:
            along = end - start
            normal = np.array([-along[1], along[0]]) / np.linalg.norm(along)
            controls = [(start + end) / 2 + normal * spec.get("bend", 0.0)]
        start_box = None if "start" in spec else boxes[substrates[0]]
        end_box = None if "end" in spec else boxes[products[0]]
        style = "<|-|>" if rapid else "-|>"
        draw_curve(
            ax,
            [start, *controls, end],
            (start_box, end_box),
            colour,
            style,
        )
    text = spec.get("text", reaction_id)
    cofactors = get_cofactor_text(stoichiometry, legs)
    if cofactors and "text" not in spec:
        text += ("  " if spec.get("inline") else "\n") + cofactors
    ax.text(
        *spec["label"],
        text,
        fontsize=LABEL_SIZE,
        color=colour if rapid else MUTED,
        ha=spec.get("ha", "center"),
        va=spec.get("va", "bottom"),
        linespacing=0.95,
    )


def draw_species(ax):
    clamped = set(MODEL.parameter_labelling["log_conc_unbalanced"])
    texts = {}
    for node, position in POSITIONS.items():
        species = get_species(node)
        texts[node] = ax.text(
            *position,
            SPECIES_NAMES[species],
            fontsize=SPECIES_SIZE,
            color=CLAMPED_COLOUR if species in clamped else TEXT,
            ha="center",
            va="center",
            zorder=3,
        )
    return texts


def get_boxes(ax, texts):
    renderer = ax.figure.canvas.get_renderer()
    to_data = ax.transData.inverted()
    return {
        node: text.get_window_extent(renderer)
        .padded(LABEL_PAD)
        .transformed(to_data)
        for node, text in texts.items()
    }


def draw_annotations(ax):
    x, y, width, height = CELL
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0,rounding_size=2.5",
            facecolor="none",
            edgecolor=MEMBRANE_COLOUR,
            linewidth=1.2,
        ),
    )
    ax.text(
        60,
        y + 1,
        "Cell",
        fontsize=SPECIES_SIZE,
        color=MEMBRANE_COLOUR,
        style="italic",
        va="bottom",
    )
    ax.text(
        99.5,
        1.2,
        "Plasma",
        fontsize=SPECIES_SIZE,
        color=MEMBRANE_COLOUR,
        style="italic",
        ha="right",
        va="center",
    )
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
    texts = draw_species(ax)
    ax.set_xlim(*XLIM)
    ax.set_ylim(*YLIM)
    ax.axis("off")
    return texts


def draw_reactions(ax, texts):
    boxes = get_boxes(ax, texts)
    reactions = MODEL.reactions | MODEL.rapid_equilibrium_network.reactions
    for reaction_id, spec in REACTIONS.items():
        draw_reaction(
            ax,
            reaction_id,
            reactions[reaction_id],
            spec,
            reaction_id in MODEL.rapid_equilibrium_network.reactions,
            boxes,
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
