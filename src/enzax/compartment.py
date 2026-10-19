"""Provides class `Compartment` for representing compartments."""

from collections.abc import Mapping, Sequence

from equinox import Module

from enzax.parameters import SEP, check_id_has_no_separator


class Compartment(Module):
    """A volume containing some of the model's species.

    A compartment has either a fixed `volume` or, with `variable_volume=True`,
    a volume that is the algebraic variable `volume|<id>`, which some algebraic
    constraint must own. Volumes can be in any unit, but values near 1 suit
    the solvers. Each species' concentration is per unit of its compartment's
    volume.

    `species` lists the species in the compartment. If a model has
    compartments, every species must be in exactly one.
    """

    species: list[str]
    volume: float | None = None
    variable_volume: bool = False


def get_volume_variable(compartment_id: str) -> str:
    """Get the name of the algebraic variable that is a variable compartment's
    volume."""
    return f"volume{SEP}{compartment_id}"


def get_volume_variables(
    compartments: Mapping[str, Compartment] | None,
) -> dict[str, str]:
    """Map the volume variable of each compartment with a variable volume to
    that compartment."""
    return {
        get_volume_variable(c): c
        for c, compartment in (compartments or {}).items()
        if compartment.variable_volume
    }


def get_species_to_compartment(
    compartments: Mapping[str, Compartment],
    species: Sequence[str],
) -> dict[str, str]:
    """Map each species to its compartment, raising a ValueError unless every
    species is in exactly one compartment."""
    for compartment_id in compartments:
        check_id_has_no_separator(compartment_id, "Compartment")
    not_one = [
        c
        for c, comp in compartments.items()
        if (comp.volume is None) != comp.variable_volume
    ]
    if not_one:
        msg = (
            f"Compartments {not_one} need either a `volume` or "
            "`variable_volume=True`, but not both."
        )
        raise ValueError(msg)
    not_positive = [
        c
        for c, comp in compartments.items()
        if comp.volume is not None and comp.volume <= 0
    ]
    if not_positive:
        msg = f"Compartments {not_positive} need positive volumes."
        raise ValueError(msg)
    species_to_compartment: dict[str, str] = {}
    in_several: dict[str, list[str]] = {}
    for compartment_id, compartment in compartments.items():
        for s in compartment.species:
            if s in species_to_compartment:
                in_several.setdefault(s, [species_to_compartment[s]])
                in_several[s].append(compartment_id)
            else:
                species_to_compartment[s] = compartment_id
    if in_several:
        msg = (
            "A species can only be in one compartment, but these are in "
            f"several: {in_several}."
        )
        raise ValueError(msg)
    unknown = [s for s in species_to_compartment if s not in species]
    if unknown:
        msg = f"Compartments name species {unknown}, which the model lacks."
        raise ValueError(msg)
    unassigned = [s for s in species if s not in species_to_compartment]
    if unassigned:
        msg = (
            "When a model declares compartments, every species must be in "
            f"one, but these are in none: {unassigned}."
        )
        raise ValueError(msg)
    return {s: species_to_compartment[s] for s in species}


def get_reaction_compartment(
    reaction_id: str,
    stoichiometry: Mapping[str, float],
    stated: str | None,
    species_to_compartment: Mapping[str, str],
) -> str:
    """Work out which compartment's volume a reaction's rate is per, raising a
    ValueError if the reaction's stated compartment is missing or wrong."""
    spanned = list(
        dict.fromkeys(species_to_compartment[s] for s in stoichiometry),
    )
    if stated is not None and stated not in spanned:
        msg = (
            f"Reaction {reaction_id} says it is in compartment {stated!r}, "
            f"but its species are in {spanned}."
        )
        raise ValueError(msg)
    if stated is not None:
        return stated
    if len(spanned) > 1:
        msg = (
            f"Reaction {reaction_id} moves species between compartments "
            f"{spanned}, so it must say which compartment's volume its rate "
            "is per, with `compartment`."
        )
        raise ValueError(msg)
    return spanned[0]


def check_volume_variables(
    compartments: Mapping[str, Compartment] | None,
    algebraic_variables: Sequence[str],
) -> None:
    """Raise a ValueError unless each variable compartment's volume is an
    algebraic variable and each algebraic variable named like a volume is a
    variable compartment's."""
    volume_variables = get_volume_variables(compartments)
    unowned = [v for v in volume_variables if v not in algebraic_variables]
    if unowned:
        msg = (
            f"No algebraic constraint owns the volume variables {unowned}. "
            "A compartment with a variable volume needs a constraint that "
            "lists its volume variable in `variables`."
        )
        raise ValueError(msg)
    prefix = get_volume_variable("")
    not_volumes = [
        v
        for v in algebraic_variables
        if v.startswith(prefix) and v not in volume_variables
    ]
    if not_volumes:
        msg = (
            f"Algebraic variables {not_volumes} are named like compartment "
            "volumes, but the model has no such variable compartments."
        )
        raise ValueError(msg)


def check_fast_moiety_compartments(
    compartments: Mapping[str, Compartment] | None,
    species_to_compartment: Mapping[str, str],
    fast_moiety_coefficients: Mapping[str, Mapping[str, float]],
) -> None:
    """Raise a ValueError if a fast moiety spans a variable compartment and
    another compartment."""
    if not compartments:
        return
    spanning = {}
    for label, coefficients in fast_moiety_coefficients.items():
        spanned = list(
            dict.fromkeys(species_to_compartment[s] for s in coefficients),
        )
        if len(spanned) > 1 and any(
            compartments[c].variable_volume for c in spanned
        ):
            spanning[label] = spanned
    if spanning:
        msg = (
            "A fast moiety can't span a compartment with a variable volume "
            f"and another compartment, but these do: {spanning}."
        )
        raise ValueError(msg)
