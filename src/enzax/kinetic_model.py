from abc import abstractmethod, ABC
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, Float, Int, Scalar, ScalarLike


class KineticModelParameters(eqx.Module):
    """Parameters for a kinetic model."""

    log_kcat: Float[Array, " n"]
    log_enzyme: Float[Array, " n"]
    dgf: Float[Array, " n_metabolite"]
    log_km: Float[Array, " n_km"]
    log_ki: Float[Array, " n_ki"]
    log_conc_unbalanced: Float[Array, " n_unbalanced"]
    temperature: Scalar
    log_transfer_constant: Float[Array, " n_allosteric_enzyme"]
    log_dissociation_constant: Float[Array, " n_allosteric_effector"]


class KineticModelStructure(eqx.Module):
    """Structural information about a kinetic model."""

    S: Float[Array, " m n"]
    ix_balanced: Int[Array, " n_balanced"]
    ix_unbalanced: Int[Array, " n_unbalanced"]
    ix_substrate: Int[Array, " n _"]
    ix_product: Int[Array, " n _"]
    ix_reactant: Int[Array, " n _"]
    ix_rate_to_km: Int[Array, " n _"]
    ix_rate_to_ki: list[list[int]]
    ix_mic_to_metabolite: Int[Array, " m"]
    stoich_by_rate: Float[Array, "n _"]
    ix_ki_species: Int[Array, " n_competitive_inhibitor"]
    ix_rate_to_tc: list[list[int]]
    ix_rate_to_dc_inhibition: list[list[int]]
    ix_rate_to_dc_activation: list[list[int]]
    ix_dc_species: Int[Array, " n_allosteric_interaction"]
    subunits: Int[Array, " n"]


class RateEquation(ABC, eqx.Module):
    """Class representing an abstract rate equation."""

    @abstractmethod
    def __init__(
        self,
        parameters: KineticModelParameters,
        structure: KineticModelStructure,
        ix: int,
    ):
        """Signature for the __init__ method of a rate equation.

        A rate equation is initialised from a set of parameters, a structure and a positive integer index.
        """
        ...

    @abstractmethod
    def __call__(self, conc: Float[Array, " n"]) -> Scalar:
        """Signature for the __call__ method of a rate equation.

        A rate equation takes in a one dimensional array of positive float-valued concentrations and returns a scalar flux.
        """
        ...


class UnparameterisedKineticModel(eqx.Module):
    """A kinetic model without parameter values."""

    structure: KineticModelStructure
    rate_equation_classes: list[type[RateEquation]]


class KineticModel(eqx.Module):
    """A parameterised kinetic model."""

    parameters: KineticModelParameters
    structure: KineticModelStructure
    rate_equations: list[RateEquation]

    def __init__(self, parameters, unparameterised_model):
        self.parameters = parameters
        self.structure = unparameterised_model.structure
        self.rate_equations = [
            cls(self.parameters, self.structure, ix)
            for ix, cls in enumerate(
                unparameterised_model.rate_equation_classes
            )
        ]

    def __call__(
        self, conc_balanced: Float[Array, " n_balanced"]
    ) -> Float[Array, " n"]:
        """Get fluxes from balanced species concentrations.

        :param conc_balanced: a one dimensional array of positive floats representing concentrations of balanced species. Must have same size as self.structure.ix_balanced

        :return: a one dimensional array of (possibly negative) floats representing reaction fluxes. Has same size as number of columns of self.structure.S.

        """
        conc = jnp.zeros(self.structure.S.shape[0])
        conc = conc.at[self.structure.ix_balanced].set(conc_balanced)
        conc = conc.at[self.structure.ix_unbalanced].set(
            jnp.exp(self.parameters.log_conc_unbalanced)
        )
        return jnp.array(
            [
                f(conc[self.structure.ix_reactant[ix]])
                for ix, f in enumerate(self.rate_equations)
            ]
        )


def dcdt(
    t: ScalarLike, conc: Float[Array, " n_balanced"], args: KineticModel
) -> Float[Array, " n_balanced"]:
    """Get the rate of change of balanced species concentrations.

    Note that the signature is as required for a Diffrax vector field function, hence the redundant variable t and the weird name "args".

    """
    model = args
    return (model.structure.S @ model(conc))[model.structure.ix_balanced]
