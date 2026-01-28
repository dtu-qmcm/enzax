import equinox as eqx
from jax import numpy as jnp
import numpy as np
from numpy.typing import NDArray
from jaxtyping import PyTree, Scalar, Float, Array, ScalarLike

from enzax.rate_equation import ConcArray, RateEquation
from enzax.rate_equations.thermodynamics import get_keq


class MassActionInput(eqx.Module):
    kf: ScalarLike
    dgf: Float[Array, " _"]
    temperature: ScalarLike
    ix_substrate: NDArray
    ix_product: NDArray


class MassAction(RateEquation):
    """A reaction with first order mass action kinetics."""

    water_stoichiometry: float

    def get_input(
        self,
        parameters: PyTree,
        reaction_id: str,
        reaction_stoichiometry: NDArray[np.float64],
        species_to_dgf_ix: NDArray[np.int16],
    ):
        ix_reactant = np.argwhere(reaction_stoichiometry != 0.0).flatten()
        ix_substrate = np.argwhere(reaction_stoichiometry < 0.0).flatten()
        ix_product = np.argwhere(reaction_stoichiometry > 0.0).flatten()
        return MassActionInput(
            kf=jnp.exp(parameters["log_kf"][reaction_id]),
            ix_substrate=ix_substrate,
            ix_product=ix_product,
            dgf=parameters["dgf"][ix_reactant],
            temperature=parameters["temperature"],
        )

    def __call__(self, conc: ConcArray, ma_input: PyTree) -> Scalar:
        """Get the flux of a drain reaction."""

        keq = get_keq(
            ma_input.reaction_stoichiometry,
            ma_input.dgf,
            ma_input.temperature,
            self.water_stoichiometry,
        )
        kr = ma_input.kf / keq
        return ma_input.kf * jnp.prod(conc[self.ix_substrate]) - kr * jnp.prod(
            conc[self.ix_product]
        )
