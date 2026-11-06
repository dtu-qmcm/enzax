"""Thermodynamic quantities calculated from formation energies.

enzax gives every compound a standard formation energy `dgf`, in kJ/mol, and
calculates each reaction's standard Gibbs energy change from it:

    dgr_std = reactant_stoichiometry @ dgf + water_stoichiometry * water_dgf

Water is added separately because it is not one of the model's species. The
functions here turn `dgr_std` into what rate laws need, so every reversible
rate law in a model agrees about where each reaction's equilibrium is.
"""

from collections.abc import Sequence
from typing import TYPE_CHECKING

import equinox as eqx
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Float, Int, Scalar

from enzax.array_types import (
    BalancedConcArr,
    BatchFlux,
    ConcArray,
    Flux,
    LocalSpeciesIx,
    ParamDict,
    ReactantArr,
    StaticReactantArr,
)

from enzax.reaction import get_slots
from enzax.utils import pad_rows

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
    Reactant concentrations are clipped below at 1e-12 and the exponent to
    between -100 and 100, so the result can differ slightly from the formula
    in extreme cases, and a NaN result raises an error.
    """
    RT = temperature * GAS_CONSTANT
    conc_clipped = jnp.clip(reactant_conc, min=1e-12)
    dgr_std = (
        reactant_stoichiometry.T @ dgf + water_stoichiometry * water_dgf
    ).flatten()
    quotient = (reactant_stoichiometry.T @ jnp.log(conc_clipped)).flatten()
    expand = jnp.clip((dgr_std / RT) + quotient, min=-1e2, max=1e2)
    out = -jnp.expm1(expand)[0]
    return eqx.error_if(out, jnp.isnan(out), "Reversibility is nan!")


def get_reverse_term(
    reactant_conc: ReactantArr,
    forward_order: ReactantArr,
    dgf: ReactantArr,
    temperature: Scalar,
    reactant_stoichiometry: StaticReactantArr,
    water_stoichiometry: float,
    water_dgf: float,
) -> Scalar:
    """Get a reaction's forward term multiplied by Q/K.

    The forward term is the product of each reactant's concentration raised
    to its forward order, so a rate of the form forward * (1 - Q/K) is
    forward - reverse. Each reactant's exponents are added before any
    concentration is raised to them, so a substrate whose forward order
    matches its stoichiometric coefficient drops out, and the result stays
    finite when that substrate's concentration is zero. A reactant left with a
    negative exponent has its concentration clipped below at 1e-12.
    """
    RT = temperature * GAS_CONSTANT
    dgr_std = jnp.ravel(
        reactant_stoichiometry.T @ dgf + water_stoichiometry * water_dgf,
    )[0]
    exponent = jnp.ravel(forward_order + reactant_stoichiometry)
    conc = jnp.ravel(reactant_conc)
    base = jnp.where(
        exponent > 0,
        jnp.maximum(conc, 0.0),
        jnp.maximum(conc, 1e-12),
    )
    return jnp.exp(dgr_std / RT) * jnp.prod(base**exponent)


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
    ix_reactant = np.flatnonzero(stoichiometry != 0.0)
    keq = get_keq(
        parameters["dgf"][model.species_to_dgf_ix[ix_reactant]],
        parameters["temperature"],
        stoichiometry[ix_reactant],
        rate_equation.water_stoichiometry,
        model.water_dgf,
    )
    log_q_without_product = sum(
        stoichiometry[i] * jnp.log(conc[i])
        for i in ix_reactant
        if i != ix_product
    )
    conc_product = jnp.exp(
        (jnp.log(keq) - log_q_without_product) / stoichiometry[ix_product],
    )
    conc_eq = conc.at[ix_product].set(conc_product)
    return rate_equation(conc_eq, rate_equation.get_input(parameters, ix))


def get_one_way_fluxes(
    model: "KineticModel",
    conc_balanced: BalancedConcArr,
    parameters: ParamDict,
) -> tuple[Flux, Flux]:
    """Get each reaction's forward and reverse fluxes.

    A reaction's one-way fluxes satisfy `reverse / forward = Q/K`, the
    flux-force relation, so `forward = v / (1 - Q/K)` for net flux `v`. An
    irreversible reaction has no reverse flux. At exactly `Q = K` the net flux
    says nothing about the one-way fluxes, and the result is NaN.

    :param conc_balanced: the balanced species' concentrations, as
        `KineticModel.flux` takes them.
    """
    flux = model.flux(conc_balanced, parameters)
    conc = model.get_conc(
        conc_balanced,
        model.get_log_conc_unbalanced(parameters),
    )
    reactions = [model.reactions[r] for r in model.reaction_ids]
    if not all(hasattr(r, "reversible") for r in reactions):
        msg = "Every reaction must say whether it is reversible."
        raise NotImplementedError(msg)
    reversible = np.array([r.reversible for r in reactions])
    water_stoichiometry = np.array([r.water_stoichiometry for r in reactions])
    S = np.asarray(model.S)
    dgf = parameters["dgf"][model.species_to_dgf_ix]
    dgr_std = S.T @ dgf + water_stoichiometry * model.water_dgf
    RT = parameters["temperature"] * GAS_CONSTANT
    log_q = S.T @ jnp.log(jnp.clip(conc, min=1e-12))
    log_qk = jnp.where(reversible, dgr_std / RT + log_q, -jnp.inf)
    forward = flux / -jnp.expm1(log_qk)
    return forward, forward - flux


class ReverseTermIx(eqx.Module):
    """Where a batch of reactions reads its reverse terms' inputs.

    Slots index a local concentration block with a column of ones appended,
    and are padded with the position of that column. `ix_dgf` is padded with
    -1, which reads an appended formation energy of 0.
    """

    reactant_slot: Int[np.ndarray, "n_batch max_reactants"]
    reactant_exponent: Float[np.ndarray, "n_batch max_reactants"]
    reactant_stoichiometry: Float[np.ndarray, "n_batch max_reactants"]
    ix_dgf: Int[np.ndarray, "n_batch max_reactants"]
    water_stoichiometry: Float[np.ndarray, " n_batch"]
    water_dgf: Float[np.ndarray, " n_batch"]


def get_reverse_term_indexes(
    local_species: LocalSpeciesIx,
    ix_reactant: Sequence[np.ndarray],
    forward_order: Sequence[np.ndarray],
    reactant_stoichiometry: Sequence[np.ndarray],
    ix_dgf: Sequence[np.ndarray],
    water_stoichiometry: Sequence[float],
    water_dgf: Sequence[float],
) -> ReverseTermIx:
    """Combine reactions' reverse term inputs into one batch bundle.

    Each reaction's exponents are its forward orders plus its stoichiometry,
    as in `get_reverse_term`.
    """
    n_slot = local_species.shape[1]
    return ReverseTermIx(
        reactant_slot=pad_rows(
            [
                get_slots(local_species, row, species)
                for row, species in enumerate(ix_reactant)
            ],
            n_slot,
        ),
        reactant_exponent=pad_rows(
            [
                order + stoichiometry
                for order, stoichiometry in zip(
                    forward_order,
                    reactant_stoichiometry,
                )
            ],
            0.0,
            float,
        ),
        reactant_stoichiometry=pad_rows(reactant_stoichiometry, 0.0, float),
        ix_dgf=pad_rows([ix.tolist() for ix in ix_dgf], -1),
        water_stoichiometry=np.array(water_stoichiometry, dtype=float),
        water_dgf=np.array(water_dgf, dtype=float),
    )


def get_reverse_terms(
    conc: Float[Array, "n_batch n_slot_and_one"],
    parameters: ParamDict,
    ix: ReverseTermIx,
) -> BatchFlux:
    """Get a batch of reactions' reverse terms, as `get_reverse_term` gets
    one."""
    RT = parameters["temperature"] * GAS_CONSTANT
    dgf = jnp.append(parameters["dgf"], 0.0)[ix.ix_dgf]
    dgr_std = (
        jnp.einsum("ij,ij->i", ix.reactant_stoichiometry, dgf)
        + ix.water_stoichiometry * ix.water_dgf
    )
    rows = np.arange(conc.shape[0])[:, None]
    reactant_conc = conc[rows, ix.reactant_slot]
    base = jnp.where(
        ix.reactant_exponent > 0,
        jnp.maximum(reactant_conc, 0.0),
        jnp.maximum(reactant_conc, 1e-12),
    )
    return jnp.exp(dgr_std / RT) * jnp.prod(
        base**ix.reactant_exponent,
        axis=1,
    )
