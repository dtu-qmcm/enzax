"""Thermodynamic quantities calculated from formation energies.

enzax gives every compound a standard formation energy `dgf`, in kJ/mol, and
calculates each reaction's standard Gibbs energy change from it:

    dgr_std = reactant_stoichiometry @ dgf + water_stoichiometry * water_dgf

Water is added separately because it is not one of the model's species. The
functions here turn `dgr_std` into what rate laws need, so every reversible
rate law in a model agrees about where each reaction's equilibrium is.
"""

from typing import TYPE_CHECKING

import equinox as eqx
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Scalar

from enzax.array_types import (
    ConcArray,
    ParamDict,
    ReactantArr,
    StaticReactantArr,
)

if TYPE_CHECKING:
    from enzax.kinetic_model import KineticModel

# The gas constant in kJ/mol/K, the units formation energies are given in.
GAS_CONSTANT = 0.008314


def get_reversibility(
    reactant_conc: ReactantArr,
    dgf: ReactantArr,
    temperature: Scalar,
    reactant_stoichiometry: StaticReactantArr,
    water_stoichiometry: float,
    water_dgf: float,
) -> Scalar:
    """Get the reversibility of a reaction.

    The equation is

      1 - exp(((dgr + (RT * quotient)) / RT))

    but it's implemented a bit differently so as to be more numerically stable.
    Reactant concentrations are clipped below at 1e-9 and the exponent to
    between -100 and 100, so the result can differ slightly from the formula
    in extreme cases, and a NaN result raises an error.
    """
    RT = temperature * GAS_CONSTANT
    conc_clipped = jnp.clip(reactant_conc, min=1e-9)
    dgr_std = (
        reactant_stoichiometry.T @ dgf + water_stoichiometry * water_dgf
    ).flatten()
    quotient = (reactant_stoichiometry.T @ jnp.log(conc_clipped)).flatten()
    expand = jnp.clip((dgr_std / RT) + quotient, min=-1e2, max=1e2)
    out = -jnp.expm1(expand)[0]
    return eqx.error_if(out, jnp.isnan(out), "Reversibility is nan!")


def get_keq(
    dgf: Array,
    temperature: Scalar,
    reactant_stoichiometry: np.ndarray,
    water_stoichiometry: float,
    water_dgf: float,
) -> Scalar:
    """Get a reaction's equilibrium constant from its formation energies.

    The equation is

        K = exp(-dgr_std / RT)

    The standard state is a concentration of 1 in the model's concentration
    units, so K has those units raised to the reaction's net stoichiometry:
    for example mM^-1 for a binding reaction A + B -> AB in a model whose
    concentrations are in mM.
    """
    RT = temperature * GAS_CONSTANT
    dgr_std = reactant_stoichiometry @ dgf + water_stoichiometry * water_dgf
    return jnp.exp(-dgr_std / RT)


def get_flux_at_equilibrium(
    model: "KineticModel",
    reaction_id: str,
    conc: ConcArray,
    parameters: ParamDict,
) -> Scalar:
    """Get a reaction's flux at a point where it is at equilibrium.

    The point is `conc` with the reaction's first product changed so that the
    mass action ratio equals the equilibrium constant. A thermodynamically
    consistent rate law gives zero flux there, so this checks one that is not
    consistent by construction, such as a `SymbolicReaction` with a
    hand-written equilibrium constant. Irreversible rate laws fail the check,
    as they should.

    `conc` holds the concentrations of all the model's species, in the model's
    order, as a rate equation receives them.
    """
    position = model.reaction_ids.index(reaction_id)
    rate_equation = model.reactions[reaction_id]
    ix = model.reaction_ix[position]
    stoichiometry = model.S[:, position]
    products = np.flatnonzero(stoichiometry > 0.0)
    if len(products) == 0:
        msg = (
            f"Reaction {reaction_id} has no products, so it has no "
            "equilibrium to evaluate its flux at."
        )
        raise ValueError(msg)
    ix_product = products[0]
    water_stoichiometry = getattr(rate_equation, "water_stoichiometry", 0.0)
    water_dgf = getattr(rate_equation, "water_dgf", -150.9)
    ix_reactant = np.flatnonzero(stoichiometry != 0.0)
    keq = get_keq(
        parameters["dgf"][model.species_to_dgf_ix[ix_reactant]],
        parameters["temperature"],
        stoichiometry[ix_reactant],
        water_stoichiometry,
        water_dgf,
    )
    log_q_without_product = sum(
        stoichiometry[i] * jnp.log(conc[i])
        for i in ix_reactant
        if i != ix_product
    )
    conc_product = jnp.exp(
        (jnp.log(keq) - log_q_without_product) / stoichiometry[ix_product]
    )
    conc_eq = conc.at[ix_product].set(conc_product)
    return rate_equation(conc_eq, rate_equation.get_input(parameters, ix))
