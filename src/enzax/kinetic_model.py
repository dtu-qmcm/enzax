"""Module containing enzax's definition of a kinetic model."""

from abc import ABC, abstractmethod

import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, PyTree, ScalarLike, jaxtyped
from typeguard import typechecked

from enzax.rate_equation import RateEquation


@jaxtyped(typechecker=typechecked)
class KineticModelStructure(eqx.Module):
    """Structural information about a kinetic model."""

    S: Float[Array, " s r"]
    balanced_species: Int[Array, " n_balanced"]
    unbalanced_species: Int[Array, " n_unbalanced"]


class KineticModel(eqx.Module, ABC):
    """Abstract base class for kinetic models."""

    parameters: PyTree
    structure: KineticModelStructure

    @abstractmethod
    def flux(
        self,
        conc_balanced: Float[Array, " n_balanced"],
    ) -> Float[Array, " n"]: ...

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


class RateEquationModel(KineticModel):
    """A kinetic model that specifies its fluxes using RateEquation objects."""

    rate_equations: list[RateEquation] = eqx.field(default_factory=list)

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


class KineticModelSbml(KineticModel):
    balanced_ids: PyTree
    sym_module: any

    
    def flux(
        self,
        conc_balanced: Float[Array, " n_balanced"],
    ) -> Float[Array, " n"]:
        flux = jnp.array(
            self.sym_module(
                **self.parameters, **dict(zip(self.balanced_ids, conc_balanced))
            )
        )
        return flux
