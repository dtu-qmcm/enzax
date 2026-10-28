"""Keeping concentrations non-negative while integrating a kinetic model.

Two things work together. clamp_at_zero stops any ODE state at or below zero
from decreasing further, and NonNegativeSolver rejects any step that leaves a
balanced species' concentration more than a small tolerance below zero, so the
step size controller retries it with a shorter step. The tolerance is not zero
because species whose concentrations are found from conserved moiety totals
pick up rounding error.
"""

from collections.abc import Callable

import diffrax
import equinox as eqx
import jax
from jax import numpy as jnp
from jaxtyping import Array, Bool, PyTree

from enzax.array_types import BalancedConcArr, OdeStateArr
from enzax.kinetic_model import KineticModel

DEFAULT_NEGATIVE_TOLERANCE = 1e-12


def clamp_at_zero(
    state: OdeStateArr,
    rate: OdeStateArr,
) -> OdeStateArr:
    """Stop any state at or below zero from decreasing further."""
    return jnp.where(state <= 0, jnp.maximum(rate, 0.0), rate)


def reject_error(y_error: PyTree, reject: Bool[Array, ""]) -> PyTree:
    """Make a step's error estimate infinite if it is to be rejected.

    diffrax-bdf's BDF returns its error together with its order, as an object
    with an error field; only the error is changed.
    """
    if y_error is None:
        return None
    if hasattr(y_error, "error") and hasattr(y_error, "order"):
        return eqx.tree_at(
            lambda e: e.error,
            y_error,
            replace_fn=lambda error: reject_error(error, reject),
        )
    return jax.tree.map(lambda e: jnp.where(reject, jnp.inf, e), y_error)


def get_dae_balanced_conc(
    model: KineticModel,
    y: tuple[OdeStateArr, dict[str, Array]],
    parameters: PyTree,
) -> BalancedConcArr:
    """Get the balanced species' concentrations from a DAE state."""
    return model._get_algebraic_conc(y[0], y[1], parameters)[1]


def clamp_dae_rate(model: KineticModel, t, y, args):
    """Get the DAE vector field, with the ODE state's rates clamped at zero."""
    rate, residuals = model.dae_vector_field(t, y, args)
    return clamp_at_zero(y[0], rate), residuals


class NonNegativeSolver(diffrax.AbstractWrappedSolver):
    """A solver that rejects steps leaving a concentration below zero.

    get_conc gets the balanced species' concentrations from the solver's state
    and the parameters. A step whose concentrations go below -tolerance gets an
    infinite error estimate, so the step size controller rejects it and tries
    a shorter one.
    """

    solver: diffrax.AbstractSolver
    get_conc: Callable = eqx.field(static=True)
    tolerance: float = eqx.field(static=True)

    @property
    def term_structure(self):
        return self.solver.term_structure

    @property
    def interpolation_cls(self):
        return self.solver.interpolation_cls

    @property
    def root_finder(self):
        return self.solver.root_finder

    def order(self, terms):
        return self.solver.order(terms)

    def error_order(self, terms):
        return self.solver.error_order(terms)

    def init(self, terms, t0, t1, y0, args):
        return self.solver.init(terms, t0, t1, y0, args)

    def step(self, terms, t0, t1, y0, args, solver_state, made_jump):
        y1, y_error, dense_info, solver_state, result = self.solver.step(
            terms,
            t0,
            t1,
            y0,
            args,
            solver_state,
            made_jump,
        )
        reject = (self.get_conc(y1, args) < -self.tolerance).any()
        return (
            y1,
            reject_error(y_error, reject),
            dense_info,
            solver_state,
            result,
        )

    def func(self, terms, t0, y0, args):
        return self.solver.func(terms, t0, y0, args)
