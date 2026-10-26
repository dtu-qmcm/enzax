"""Time courses: solving a kinetic model from a given initial state.

A time course starts from the amounts of every balanced species, which are the
parameter log_amount_initial. get_initial_state_labelling gives that
parameter's labelling, to merge into the model's. The initial amounts also fix
the conserved moiety totals, so a time course ignores the moiety_totals
parameter.

There are two solves with the same outputs.
get_measurables_time_course_single_experiment integrates dndt with a stiff ODE
solver, solving any rapid equilibria and algebraic constraints at every step.
get_measurables_time_course_single_experiment_dae integrates the model as a
differential algebraic equation with diffrax-bdf's BDF solver, which needs
diffrax-bdf installed.
"""

import diffrax
import jax
from jax import numpy as jnp
from jaxtyping import Array, Bool, Float, PyTree

from enzax.array_types import (
    AlgebraicVariableArr,
    BalancedConcArr,
    ConcArray,
    Flux,
    MoietyTotalsArr,
    OdeStateArr,
    ParamLabelling,
)
from enzax.kinetic_model import KineticModel
from enzax.steady_state import DEFAULT_MAX_STEPS

INITIAL_STATE_PARAMETER = "log_amount_initial"


def get_initial_state_labelling(model: KineticModel) -> ParamLabelling:
    """Get the labelling of the initial state parameter, log_amount_initial.

    Merge it into the model's labelling, as in
    model.parameter_labelling | get_initial_state_labelling(model), to make a
    labelling for time courses.
    """
    return {INITIAL_STATE_PARAMETER: tuple(model.balanced_species)}


def get_initial_ode_state(
    model: KineticModel,
    amount_initial: BalancedConcArr,
) -> tuple[OdeStateArr, MoietyTotalsArr]:
    """Get the ODE state and the conserved moiety totals from initial amounts.

    amount_initial holds every balanced species' amount, in the model's
    balanced species order. Species in a rapid equilibrium need not be at
    equilibrium: only their fast moieties' totals matter, and the solve
    equilibrates them.
    """
    totals = model.fast_moieties.fast_moiety_matrix @ amount_initial
    ode_state = totals[model.fast_moiety_rows(model.ode_state_species)]
    dependent = totals[model.fast_moiety_rows(model.moiety_label_species)]
    return ode_state, dependent - model.L0 @ ode_state


def with_moiety_totals(
    model: KineticModel,
    parameters: PyTree,
    moiety_totals: MoietyTotalsArr,
) -> PyTree:
    """Replace the parameters' moiety totals, if the model has any."""
    if not model.moiety_label_species:
        return parameters
    return parameters | {"moiety_totals": moiety_totals}


def get_measurables_at(
    model: KineticModel,
    parameters: PyTree,
    ode_states: Float[Array, "n_time n_ode_state"],
) -> tuple[
    Float[Array, "n_time n_species"],
    Float[Array, "n_time n_reaction"],
    Float[Array, "n_time n_algebraic_variable"],
]:
    """Get concentrations, fluxes and algebraic variables at some ODE states."""

    def at(
        ode_state: OdeStateArr,
    ) -> tuple[ConcArray, Flux, AlgebraicVariableArr]:
        algebraic = model._solve_algebraic(ode_state, parameters)
        conc_balanced = model._get_algebraic_conc(
            ode_state,
            algebraic,
            parameters,
        )[1]
        conc = model.get_conc(
            conc_balanced,
            model.get_log_conc_unbalanced(parameters),
        )
        variables = algebraic["variables"]
        return conc, model.flux(conc_balanced, parameters, variables), variables

    return jax.vmap(at)(ode_states)


def finish(
    model: KineticModel,
    parameters: PyTree,
    sol: diffrax.Solution,
    ode_states: Float[Array, "n_time n_ode_state"],
    ode_state_initial: OdeStateArr,
) -> tuple[
    Bool[Array, ""],
    Float[Array, "n_time n_species"],
    Float[Array, "n_time n_reaction"],
    Float[Array, "n_time n_algebraic_variable"],
]:
    """Get whether a solve succeeded and the measurables at its saved states.

    If it failed, the measurables are computed at the initial state instead,
    since the rate laws raise on NaN concentrations: they are then finite but
    meaningless.
    """
    found = (sol.result == diffrax.RESULTS.successful) & jnp.isfinite(
        ode_states,
    ).all()
    ode_states = jnp.where(found, ode_states, ode_state_initial)
    return found, *get_measurables_at(model, parameters, ode_states)


def get_measurables_time_course_single_experiment(
    model: KineticModel,
    parameters: PyTree,
    amount_initial: BalancedConcArr,
    ts: Float[Array, " n_time"],
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = DEFAULT_MAX_STEPS,
    solver: diffrax.AbstractSolver | None = None,
    stepsize_controller: diffrax.AbstractStepSizeController | None = None,
) -> tuple[
    Bool[Array, ""],
    Float[Array, "n_time n_species"],
    Float[Array, "n_time n_reaction"],
    Float[Array, "n_time n_algebraic_variable"],
]:
    """Solve one experiment's time course and get the measurables over time.

    The solve starts at time 0 from amount_initial and saves at each of ts,
    which must increase and end after 0. Gradients go through the solver's
    steps, with diffrax's RecursiveCheckpointAdjoint.

    :param rtol: relative tolerance of the default step size controller.

    :param atol: absolute tolerance of the default step size controller.

    :param max_steps: how many steps the solve may take before it is called a
        failure.

    :param solver: a diffrax ODE solver. Defaults to Kvaerno5.

    :param stepsize_controller: a diffrax step size controller. Defaults to a
        PIDController with rtol and atol.

    :return: whether the solve succeeded, then every species' concentration,
        every reaction's flux and every algebraic variable at each of ts.
    """
    ode_state_initial, moiety_totals = get_initial_ode_state(
        model,
        amount_initial,
    )
    parameters = with_moiety_totals(model, parameters, moiety_totals)
    if solver is None:
        solver = diffrax.Kvaerno5()
    if stepsize_controller is None:
        stepsize_controller = diffrax.PIDController(
            pcoeff=0.1,
            icoeff=0.3,
            rtol=rtol,
            atol=atol,
        )
    sol = diffrax.diffeqsolve(
        terms=diffrax.ODETerm(lambda t, y, args: model.dndt(y, args)),
        solver=solver,
        t0=0.0,
        t1=ts[-1],
        dt0=1e-6,
        y0=ode_state_initial,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=stepsize_controller,
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        args=parameters,
        max_steps=max_steps,
        throw=False,
    )
    return finish(model, parameters, sol, sol.ys, ode_state_initial)


def get_measurables_time_course_single_experiment_dae(
    model: KineticModel,
    parameters: PyTree,
    amount_initial: BalancedConcArr,
    ts: Float[Array, " n_time"],
    rtol: float = 1e-8,
    atol: float = 1e-10,
    max_steps: int = DEFAULT_MAX_STEPS,
    suppress_algebraic_error: bool = True,
) -> tuple[
    Bool[Array, ""],
    Float[Array, "n_time n_species"],
    Float[Array, "n_time n_reaction"],
    Float[Array, "n_time n_algebraic_variable"],
]:
    """Solve one experiment's time course as a differential algebraic equation.

    Takes the same arguments as get_measurables_time_course_single_experiment,
    except solver and stepsize_controller, and returns the same things. Needs
    diffrax-bdf.

    :param suppress_algebraic_error: whether to leave the log concentrations
        out of the integrator's error estimate.
    """
    from diffrax_bdf import BDF, BDFController, SemiExplicitDAETerm

    ode_state_initial, moiety_totals = get_initial_ode_state(
        model,
        amount_initial,
    )
    parameters = with_moiety_totals(model, parameters, moiety_totals)
    y0 = model.get_dae_state(ode_state_initial, parameters)
    sol = diffrax.diffeqsolve(
        terms=SemiExplicitDAETerm(
            model.dae_vector_field,
            (False, jax.tree.map(lambda _: True, y0[1])),
        ),
        solver=BDF(suppress_algebraic_error=suppress_algebraic_error),
        t0=0.0,
        t1=ts[-1],
        dt0=1e-6,
        y0=y0,
        saveat=diffrax.SaveAt(ts=ts),
        stepsize_controller=BDFController(rtol=rtol, atol=atol),
        adjoint=diffrax.RecursiveCheckpointAdjoint(),
        args=parameters,
        max_steps=max_steps,
        throw=False,
    )
    return finish(model, parameters, sol, sol.ys[0], ode_state_initial)
