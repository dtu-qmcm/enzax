"""Module containing enzax's definition of a kinetic model."""

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree, ScalarLike, jaxtyped
from typeguard import typechecked

from enzax.rate_equation import RateEquation


@jaxtyped(typechecker=typechecked)
class KineticModelStructure(eqx.Module):
    """Structural information about a kinetic model."""

    S: Float[Array, " s r"] = eqx.field(static=True)
    balanced_species: Int[Array, " n_balanced"] = eqx.field(static=True)
    unbalanced_species: Int[Array, " n_unbalanced"] = eqx.field(static=True)


class UnparameterisedKineticModel(eqx.Module):
    """A kinetic model without parameter values."""

    structure: KineticModelStructure
    rate_equations: list[RateEquation] | None = None


class KineticModel(eqx.Module):
    """A parameterised kinetic model."""

    parameters: PyTree
    structure: KineticModelStructure
    rate_equations: list[RateEquation] | None = None

    def __init__(self, parameters, unparameterised_model):
        self.parameters = parameters
        self.structure = unparameterised_model.structure
        self.rate_equations = unparameterised_model.rate_equations

    def flux(
        self,
        conc_balanced: Float[Array, " n_balanced"],
    ) -> Float[Array, " n"]:
        """Get fluxes from balanced species concentrations.

        :param conc_balanced: a one dimensional array of positive floats representing concentrations of balanced species. Must have same size as self.structure.ix_balanced

        :return: a one dimensional array of (possibly negative) floats representing reaction fluxes. Has same size as number of columns of self.structure.S.

        """  # Noqa: E501
        conc = jnp.zeros(self.structure.S.shape[0])
        conc = conc.at[self.structure.balanced_species].set(conc_balanced)
        conc = conc.at[self.structure.unbalanced_species].set(
            jnp.exp(self.parameters.log_conc_unbalanced)
        )
        t = [f(conc, self.parameters) for f in self.rate_equations]
        out = jnp.array(t)
        return out

    def dcdt(
        self, t: ScalarLike, conc: Float[Array, " n_balanced"], args=None
    ) -> Float[Array, " n_balanced"]:
        """Get the rate of change of balanced species concentrations.

        Note that the signature is as required for a Diffrax vector field function, hence the redundant variable t and the weird name "args".

        :param t: redundant variable representing time.

        :param conc: a one dimensional array of positive floats representing concentrations of balanced species. Must have same size as self.structure.ix_balanced

        """  # Noqa: E501
        sv = self.structure.S @ self.flux(conc)
        return sv[self.structure.balanced_species]
