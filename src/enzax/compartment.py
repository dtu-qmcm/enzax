"""Provides class `Compartment` for representing compartments."""

from collections.abc import Mapping, Sequence

from equinox import Module


class Compartment(Module):
    """A fixed volume containing some of the model's species.

    `volume` can be in any unit. Each species' concentration is per unit of its
    compartment's volume.

    `species` lists the species in the compartment. If a model has
    compartments, every species must be in exactly one.
    """

    volume: float
    species: list[str]


def get_species_to_compartment(
    compartments: Mapping[str, Compartment],
    species: Sequence[str],
) -> dict[str, str]:
    """Map each species to its compartment, raising a ValueError unless every
    species is in exactly one compartment."""
    not_positive = [c for c, comp in compartments.items() if comp.volume <= 0]
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
