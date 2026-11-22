import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path

from common import COLOURS, HYPOTHESES, MODEL, REGULATED_REACTION, RESULTS_DIR

KINETIC_COLOUR = "#0b0b0b"
CLAMPED_COLOUR = "#8a8984"
REGULATION_COLOUR = "#8a8984"
MEMBRANE_COLOUR = "#2a78d6"
TEXT = "#0b0b0b"
MUTED = "#52514e"
CELL = (5, 2.5, 59.5, 77.5)
XLIM = (0, 66.5)
YLIM = (0, 82)
LABEL_SIZE = 5
SPECIES_SIZE = 6
LABEL_PAD = 1.5
TARGET_PAD = 3
FORK_CONTROL = 0.6
COFACTOR_SIZE = 4
COFACTOR_LENGTH = 0.9
COFACTOR_WIDTH = 0.9
COFACTOR_PAD = 0.2
STEM = 2.0
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


def get_species(node):
    return DUPLICATES.get(node, node)


def get_cofactors(stoichiometry, legs):
    shown = {get_species(node) for node in legs}
    sides = ([], [])
    for species, coefficient in stoichiometry.items():
        if species in shown:
            continue
        name = SPECIES_NAMES[species]
        if abs(coefficient) != 1:
            name = f"{abs(coefficient):g} {name}"
        sides[coefficient > 0].append(name)
    return sides


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


def draw_curve(
    ax,
    points,
    boxes,
    colour,
    style,
    linestyle="-",
    fill=True,
    linewidth=0.7,
    mutation_scale=5,
):
    points = trim([np.asarray(point, dtype=float) for point in points], *boxes)
    codes = {2: Path.LINETO, 3: Path.CURVE3, 4: Path.CURVE4}[len(points)]
    ax.add_patch(
        FancyArrowPatch(
            path=Path(points, [Path.MOVETO] + [codes] * (len(points) - 1)),
            arrowstyle=style,
            mutation_scale=mutation_scale,
            linewidth=linewidth,
            edgecolor=colour,
            facecolor=colour if fill else "white",
            linestyle=linestyle,
        ),
    )
    return points


def draw_cofactors(ax, curve, cofactors, side):
    middle = evaluate(curve, 0.5)
    along = evaluate(curve, 0.55) - evaluate(curve, 0.45)
    along /= np.linalg.norm(along)
    normal = side * np.array([-along[1], along[0]])
    ends = [
        middle + sign * along * COFACTOR_LENGTH + normal * COFACTOR_WIDTH
        for sign in (-1, 1)
    ]
    draw_curve(
        ax,
        [ends[0], middle - normal * COFACTOR_WIDTH, ends[1]],
        (None, None),
        CLAMPED_COLOUR,
        "-|>",
        linewidth=0.5,
        mutation_scale=3.5,
    )
    vertical = abs(normal[0]) > abs(normal[1])
    if vertical:
        alignments = [("left" if normal[0] > 0 else "right", "center")] * 2
    else:
        va = "bottom" if normal[1] > 0 else "top"
        alignments = [("right", va), ("left", va)]
    for end, names, (ha, va) in zip(ends, cofactors, alignments, strict=True):
        ax.text(
            *(end + normal * COFACTOR_PAD),
            " + ".join(names),
            fontsize=COFACTOR_SIZE,
            color=CLAMPED_COLOUR,
            ha=ha,
            va=va,
        )


def draw_reaction(ax, reaction_id, reaction, spec, boxes):
    stoichiometry = reaction.stoichiometry
    colour = KINETIC_COLOUR
    legs = spec["legs"]
    substrates = [n for n in legs if stoichiometry[get_species(n)] < 0]
    products = [n for n in legs if stoichiometry[get_species(n)] > 0]
    if "point" in spec:
        point = np.asarray(spec["point"], dtype=float)
        direction = np.asarray(spec["direction"], dtype=float)
        direction /= np.linalg.norm(direction)
        stem = direction * spec.get("stem", STEM) / 2
        curve = draw_curve(
            ax,
            [point - stem, point + stem],
            (None, None),
            colour,
            "-",
        )
        for node, sign in [(n, -1) for n in substrates] + [
            (n, 1) for n in products
        ]:
            end = point + sign * stem
            position = np.asarray(POSITIONS[node], dtype=float)
            reach = max(direction @ (position - end) * sign, 0.0)
            control = end + sign * direction * reach * FORK_CONTROL
            controls = spec.get("leg_controls", {}).get(node, [control])
            style = "-|>" if sign > 0 else "-"
            draw_curve(
                ax,
                [end, *controls, position],
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
        style = "-|>"
        curve = draw_curve(
            ax,
            [start, *controls, end],
            (start_box, end_box),
            colour,
            style,
        )
    text = spec.get("text", reaction_id)
    cofactors = get_cofactors(stoichiometry, legs)
    if any(cofactors):
        draw_cofactors(ax, curve, cofactors, spec["side"])
    return ax.text(
        *spec["label"],
        text,
        fontsize=LABEL_SIZE,
        color=MUTED,
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


def get_boxes(ax, texts, pad=LABEL_PAD):
    renderer = ax.figure.canvas.get_renderer()
    to_data = ax.transData.inverted()
    return {
        node: text.get_window_extent(renderer).padded(pad).transformed(to_data)
        for node, text in texts.items()
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
        x + width - 1.5,
        y + 1,
        "Cell",
        fontsize=SPECIES_SIZE,
        color=MEMBRANE_COLOUR,
        style="italic",
        ha="right",
        va="bottom",
    )
    ax.text(
        66.3,
        1.2,
        "Medium",
        fontsize=SPECIES_SIZE,
        color=MEMBRANE_COLOUR,
        style="italic",
        ha="right",
        va="center",
    )
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
    texts = draw_species(ax)
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
            MODEL.reactions[reaction_id],
            spec,
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
