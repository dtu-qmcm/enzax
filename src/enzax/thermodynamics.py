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
    from enzax.kinetic_model import RateEquationModel

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
    """  # noqa: E501
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
    RT = temperature * GAS_CONSTANT
    dgr_std = reactant_stoichiometry @ dgf + water_stoichiometry * water_dgf
    return jnp.exp(-dgr_std / RT)


def get_flux_at_equilibrium(
    model: "RateEquationModel",
    reaction_id: str,
    conc: ConcArray,
    parameters: ParamDict,
) -> Scalar:
    position = model.reactions.index(reaction_id)
    rate_equation = model.rate_equations[reaction_id]
    ix = model.rate_equation_ix[position]
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
