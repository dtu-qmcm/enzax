"""Draw reaction network maps for the case studies' figures.

Each case study's network module places its species and reactions by hand
and calls these functions to draw them. Species appear as text and
reactions as arrows trimmed to stop short of the species' labels. A
reaction that joins or splits several drawn species becomes a fork, and
cofactors sit on a small side arc.
"""

import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path

KINETIC_COLOUR = "#0b0b0b"
CLAMPED_COLOUR = "#8a8984"
MEMBRANE_COLOUR = "#2a78d6"
TEXT = "#0b0b0b"
MUTED = "#52514e"
LABEL_SIZE = 5
SPECIES_SIZE = 6
LABEL_PAD = 1.5
FORK_CONTROL = 0.6
STEM = 2.0
COFACTOR_SIZE = 4
COFACTOR_LENGTH = 0.9
COFACTOR_WIDTH = 0.9
COFACTOR_PAD = 0.2


def get_cofactors(stoichiometry, shown, names):
    sides = ([], [])
    for species, coefficient in stoichiometry.items():
        if species in shown:
            continue
        name = names[species]
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


def draw_cofactors(ax, curve, cofactors, side, scale=1.0):
    middle = evaluate(curve, 0.5)
    along = evaluate(curve, 0.55) - evaluate(curve, 0.45)
    along /= np.linalg.norm(along)
    normal = side * np.array([-along[1], along[0]])
    length, width = scale * COFACTOR_LENGTH, scale * COFACTOR_WIDTH
    ends = [middle + sign * along * length + normal * width for sign in (-1, 1)]
    draw_curve(
        ax,
        [ends[0], middle - normal * width, ends[1]],
        (None, None),
        CLAMPED_COLOUR,
        "-|>",
        linewidth=0.5,
        mutation_scale=3.5 * scale,
    )
    if abs(normal[0]) > abs(normal[1]):
        alignments = [("left" if normal[0] > 0 else "right", "center")] * 2
    else:
        va = "bottom" if normal[1] > 0 else "top"
        alignments = [("right", va), ("left", va)][:: 1 if along[0] > 0 else -1]
    for end, names, (ha, va) in zip(ends, cofactors, alignments, strict=True):
        ax.text(
            *(end + normal * COFACTOR_PAD),
            " + ".join(names),
            fontsize=COFACTOR_SIZE,
            color=CLAMPED_COLOUR,
            ha=ha,
            va=va,
        )


def draw_reaction(
    ax,
    reaction_id,
    stoichiometry,
    spec,
    positions,
    duplicates,
    names,
    boxes,
    colour=KINETIC_COLOUR,
    label_colour=MUTED,
    reversible=False,
    cofactor_scale=1.0,
    label_size=LABEL_SIZE,
):
    legs = spec["legs"]
    shown = {duplicates.get(node, node) for node in legs}
    substrates = [n for n in legs if stoichiometry[duplicates.get(n, n)] < 0]
    products = [n for n in legs if stoichiometry[duplicates.get(n, n)] > 0]
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
            position = np.asarray(positions[node], dtype=float)
            reach = max(direction @ (position - end) * sign, 0.0)
            control = end + sign * direction * reach * FORK_CONTROL
            controls = spec.get("leg_controls", {}).get(node, [control])
            draw_curve(
                ax,
                [end, *controls, position],
                (None, boxes[node]),
                colour,
                "-|>" if sign > 0 or reversible else "-",
            )
    else:
        start = np.asarray(
            spec.get("start", positions[substrates[0]] if substrates else None),
            dtype=float,
        )
        end = np.asarray(
            spec.get("end", positions[products[0]] if products else None),
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
        curve = draw_curve(
            ax,
            [start, *controls, end],
            (start_box, end_box),
            colour,
            "<|-|>" if reversible else "-|>",
        )
    cofactors = get_cofactors(stoichiometry, shown, names)
    if any(cofactors) and "text" not in spec:
        draw_cofactors(ax, curve, cofactors, spec["side"], cofactor_scale)
    return ax.text(
        *spec["label"],
        spec.get("text", reaction_id),
        fontsize=label_size,
        color=label_colour,
        ha=spec.get("ha", "center"),
        va=spec.get("va", "bottom"),
        linespacing=0.95,
    )


def draw_species(ax, positions, duplicates, names, clamped):
    texts = {}
    for node, position in positions.items():
        species = duplicates.get(node, node)
        texts[node] = ax.text(
            *position,
            names[species],
            fontsize=SPECIES_SIZE,
            color=CLAMPED_COLOUR if species in clamped else TEXT,
            ha="center",
            va="center",
            zorder=3,
        )
    return texts


def get_boxes(ax, texts, pad=LABEL_PAD):
    """Find each text's padded box in data coordinates.

    Label sizes are only known once the figure has its layout, so call this
    after fixing the layout.
    """
    renderer = ax.figure.canvas.get_renderer()
    to_data = ax.transData.inverted()
    return {
        node: text.get_window_extent(renderer).padded(pad).transformed(to_data)
        for node, text in texts.items()
    }


def draw_cell(ax, cell, xlim, outside):
    x, y, width, height = cell
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
    for position, text, va in (
        ((x + width - 1.5, y + 1), "Cell", "bottom"),
        ((xlim[1] - 0.2, 1.2), outside, "center"),
    ):
        ax.text(
            *position,
            text,
            fontsize=SPECIES_SIZE,
            color=MEMBRANE_COLOUR,
            style="italic",
            ha="right",
            va=va,
        )
