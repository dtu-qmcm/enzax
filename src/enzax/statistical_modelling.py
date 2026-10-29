import operator
from collections.abc import Callable, Mapping, Sequence
from functools import partial

import diffrax
import jax
from jax import numpy as jnp
from jax.scipy.stats import multivariate_normal, norm
from jaxtyping import Array, Bool, Float, PyTree, Scalar

from enzax.array_types import ConcArray, Flux, OdeStateArr, ParamDict
from enzax.experiments import (
    ExperimentLayout,
    gather_parameters_by_experiment,
    get_experiment_layout,
    pick_experiment,
)
from enzax.kinetic_model import KineticModel
from enzax.measurements import (
    Measurement,
    pick_concs,
    pick_concs_by_time,
    pick_enzymes,
    pick_fluxes,
    pick_fluxes_by_time,
)
from enzax.parameter_split import ParameterSplit, combine_parameters
from enzax.parameters import SEP
from enzax.steady_state import get_steady_state_hybrid
from enzax.time_course import (
    INITIAL_STATE_PARAMETER,
    get_initial_state_labelling,
    get_measurables_time_course_single_experiment,
)


def prior_from_truth(
    true_params: ParamDict,
    sd: float,
    is_multivariate: PyTree[bool] | None = None,
) -> PyTree:
    """Get a PyTree of priors from a PyTree of true values and an sd.

    Params:
    -------
    true_params: PyTree
        A PyTree of true parameter values.
    sd: float
        The standard deviation of the prior.
    is_multivariate: PyTree[bool] | None
        An optional PyTree of booleans indicating whether each parameter is
        multivariate. If None, all parameters are assumed to be univariate.

    Returns:
    --------
    PyTree
        A PyTree of priors, where each leaf is a tuple of the prior location
        and the prior scale (i.e. the argument sd repeated as appropriate).

    """

    def get_scale_for_leaf(true_leaf: jax.Array, is_mv: bool) -> jax.Array:
        sd_arr = jnp.full_like(true_leaf, sd)
        if is_mv:
            sd_arr = jnp.diag(sd_arr)
        return sd_arr

    if is_multivariate is None:
        is_multivariate = jax.tree.map(lambda leaf: False, true_params)
    scale = jax.tree.map(get_scale_for_leaf, true_params, is_multivariate)
    return pack_locs_and_scales(loc=true_params, scale=scale)


def pack_locs_and_scales(loc: PyTree, scale: PyTree) -> PyTree:
    """Get a Pytree whose leaves are tuples of loc and scale.

    Note that loc and scale must have the same tree structure.

    Params:
    -------
    loc: PyTree
        A PyTree representing location parameters.
    scale: PyTree
        A PyTree representing scale parameters.

    Returns:
    --------
    PyTree
        A PyTree whose leaves are tuples of loc and scale.

    Examples:
    ---------
    >>> loc = {"dgf": jnp.array([1.0, -1.0]), "temperature": jnp.array(310.0)}
    >>> scale = {"dgf": jnp.array([[1.0, 2.0], [3.0, 4.0]]), "temperature": jnp.array(5.0)}
    >>> pack_locs_and_scales(loc, scale)
    {'dgf': (Array([ 1., -1.], dtype=float64),
    Array([[1., 2.],
           [3., 4.]], dtype=float64)),
    'temperature': (Array(310., dtype=float64),
     Array(5., dtype=float64))}

    """  # noqa: E501
    if not jax.tree.structure(loc) == jax.tree.structure(scale):
        msg = "Prior mean and prior sd must have the same tree structure"
        raise ValueError(msg)
    return jax.tree.transpose(
        outer_treedef=jax.tree.structure(("*", "*")),
        inner_treedef=jax.tree.structure(loc),
        pytree_to_transpose=(loc, scale),
    )


def match_prior_leaf(node):
    """Match a Pytree node representing a prior, i.e. a 2-tuple of jax arrays.

    This makes it possible to use jax.tree.map with a function that takes
    a prior leaf as an argument.

    Params:
    -------
    node: Any
        A Pytree node.

    Returns:
    --------
    bool
        True if the node is a 2-tuple of jax arrays
    """
    return (
        (isinstance(node, tuple))
        and (isinstance(node[0], jax.Array))
        and (len(node) == 2)
    )


def match_measurement(node) -> bool:
    """Match a measurement, so jax.tree.map can treat it as a leaf."""
    return isinstance(node, Measurement)


def enzax_prior_logdensity(parameters: PyTree, prior: PyTree) -> Scalar:
    """Get the prior log density for the parameters given the prior.

    The distribution used depends on the shape of each prior location and scale
    parameter: if they are the same shape, a normal distribution is used; if
    the location is one dimension lower than the scale, a multivariate normal
    distribution is used.

    Params:
    -------
    parameters: PyTree
        A PyTree representing the parameters.
    prior: PyTree
        A PyTree representing the prior. Each leaf is a tuple of the prior
        location and the prior scale (i.e. the argument sd repeated as
        appropriate

    """

    def prior_logdensity_leaf(param_leaf, prior_leaf):
        loc, scale = prior_leaf
        if loc.shape == scale.shape:
            return norm.logpdf(param_leaf, loc, scale).sum()
        elif len(loc.shape) == len(scale.shape) - 1:
            return multivariate_normal.logpdf(param_leaf, loc, scale).sum()
        else:
            msg = "Prior loc and scale have incompatible shape"
            raise ValueError(msg)

    prior_logdensities = jax.tree.map(
        prior_logdensity_leaf,
        parameters,
        prior,
        is_leaf=match_prior_leaf,
    )
    return jax.tree.reduce(operator.add, prior_logdensities)


DEFAULT_EXPERIMENT = "exp1"

_STATIC_SOLVE_ARGUMENTS = (
    "ivp_rtol",
    "ivp_atol",
    "steady_state_rtol",
    "steady_state_atol",
    "solver",
    "stepsize_controller",
)


def get_measurables_single_experiment(
    model: KineticModel,
    parameters: ParamDict,
    guess: OdeStateArr,
    ivp_rtol: float = 1e-9,
    ivp_atol: float = 1e-9,
    steady_state_rtol: float = 1e-12,
    steady_state_atol: float = 1e-12,
    solver: diffrax.AbstractSolver | None = None,
    stepsize_controller: diffrax.AbstractStepSizeController | None = None,
) -> tuple[OdeStateArr, Bool[Array, ""], ConcArray, Flux]:
    """Get one experiment's steady state and the quantities measured at it.

    parameters has the model's labelling, and guess is where the steady state
    solve starts. The other arguments are passed on to
    get_steady_state_hybrid.

    Non-finite entries of guess are replaced with 0.01, the default guess.
    Under grapevine, a failed solve hands the next draw a NaN guess, and
    nothing can start from one: the solve would return NaN again, and the rate
    laws raise on a NaN concentration rather than propagate it. Replacing it
    lets the next draw recover instead of inheriting the failure.

    :return: the steady state, whether it was found, every species'
        concentration and every reaction's flux. get_steady_state_hybrid
        returns NaN when it finds no steady state, and flux raises on a NaN
        concentration, so in that case the concentrations and fluxes are
        computed at the guess instead: they are finite but meaningless, and
        found says to throw them away. Balanced species' concentrations are
        clipped below at zero, since a solve can leave a species that has run
        out a little below zero.
    """
    guess = jnp.where(jnp.isfinite(guess), guess, 0.01)
    steady = get_steady_state_hybrid(
        model,
        guess,
        parameters,
        ivp_rtol=ivp_rtol,
        ivp_atol=ivp_atol,
        steady_state_rtol=steady_state_rtol,
        steady_state_atol=steady_state_atol,
        solver=solver,
        stepsize_controller=stepsize_controller,
    )
    found = jnp.isfinite(steady).all()
    conc_balanced = jnp.maximum(
        model.get_balanced_conc(jnp.where(found, steady, guess), parameters),
        0.0,
    )
    conc = model.get_conc(
        conc_balanced,
        model.get_log_conc_unbalanced(parameters),
    )
    flux = model.flux(conc_balanced, parameters)
    return steady, found, conc, flux


def _log_density_and_steady_states(
    free_parameters: PyTree,
    model: KineticModel,
    prior: PyTree,
    layout: ExperimentLayout | None = None,
    split: ParameterSplit | None = None,
    guess: Float[Array, "n_experiment n_state"] | None = None,
    conc_measurements: Sequence[Measurement] = (),
    enzyme_measurements: Sequence[Measurement] = (),
    flux_measurements: Sequence[Measurement] = (),
    ivp_rtol: float = 1e-9,
    ivp_atol: float = 1e-9,
    steady_state_rtol: float = 1e-12,
    steady_state_atol: float = 1e-12,
    solver: diffrax.AbstractSolver | None = None,
    stepsize_controller: diffrax.AbstractStepSizeController | None = None,
) -> tuple[Scalar, Float[Array, "n_experiment n_state"]]:
    """Get the log posterior density of the parameters, and the steady states.

    :param free_parameters: the parameters being inferred. With a split, this
        is the short form that get_free_parameters returns; without one, it is
        a complete expanded parameter set.

    :param prior: the prior, with a (location, scale) pair for each leaf of
        free_parameters, as prior_from_truth or pack_locs_and_scales returns.

    :param layout: which parameter values each experiment has. Defaults to a
        single experiment, exp1, with nothing experiment-specific.

    :param split: which parameters are free, and the values of the rest. Leave
        it out to infer every parameter.

    :param guess: where each experiment's steady state solve starts, with a
        leading experiment axis in the layout's order. Defaults to 0.01
        everywhere.

    :param conc_measurements: measurements of species concentrations.

    :param enzyme_measurements: measurements of enzyme concentrations.

    :param flux_measurements: measurements of fluxes.

    :param ivp_rtol: relative tolerance of the steady state solve's initial
        value problem, passed on to get_steady_state_hybrid. Static, since a
        traced tolerance would reach diffrax as an array rather than a float.

    :param ivp_atol: absolute tolerance of the initial value problem.

    :param steady_state_rtol: relative tolerance of the terminating event.

    :param steady_state_atol: absolute tolerance of the terminating event.

    :param solver: a diffrax ODE solver for get_steady_state_hybrid to use.

    :param stepsize_controller: a diffrax step size controller for the ODE
        solver to use.

    :return: the log density, which is NaN if any experiment's steady state
        was not found, and every experiment's steady state. A sampler reads a
        NaN density as an infinite energy change, and so as a divergence to
        reject. The steady states are returned as they are, NaN included, so
        they show which solve failed.

    The solve is get_steady_state_hybrid rather than get_steady_state, which
    matters most here: with grapevine each draw's guess is the previous draw's
    steady state, so the Newton path usually succeeds and the integration
    usually takes no steps.
    """
    if layout is None:
        layout = get_experiment_layout(
            model.parameter_labelling,
            [DEFAULT_EXPERIMENT],
            {},
        )
    if guess is None:
        guess = jnp.full(
            (len(layout.experiments), len(model.ode_state_species_ix)),
            0.01,
        )
    if split is not None:
        params = combine_parameters(split, free_parameters)
    else:
        params = free_parameters
    parameters_by_experiment = gather_parameters_by_experiment(layout, params)
    steady, found, conc, flux = jax.vmap(
        partial(
            get_measurables_single_experiment,
            model,
            ivp_rtol=ivp_rtol,
            ivp_atol=ivp_atol,
            steady_state_rtol=steady_state_rtol,
            steady_state_atol=steady_state_atol,
            solver=solver,
            stepsize_controller=stepsize_controller,
        ),
    )(parameters_by_experiment, guess)
    conc_hat, enz_hat, flux_hat = (
        jax.tree.map(
            lambda m: pick(
                model,
                pick_experiment(layout, predicted, m.experiment),
                m.ids,
            ),
            measurements,
            is_leaf=match_measurement,
        )
        for measurements, pick, predicted in (
            (conc_measurements, pick_concs, conc),
            (enzyme_measurements, pick_enzymes, parameters_by_experiment),
            (flux_measurements, pick_fluxes, flux),
        )
    )
    llik_conc, llik_enz, llik_flux = (
        jax.tree.map(
            lambda m, yhat: m.log_likelihood(yhat).sum(),
            measurements,
            yhats,
            is_leaf=match_measurement,
        )
        for measurements, yhats in (
            (conc_measurements, conc_hat),
            (enzyme_measurements, enz_hat),
            (flux_measurements, flux_hat),
        )
    )
    llik = jax.tree.reduce(operator.add, (llik_conc, llik_enz, llik_flux), 0.0)
    log_prior = enzax_prior_logdensity(free_parameters, prior)
    log_density = jnp.where(found.all(), log_prior + llik, jnp.nan)
    return log_density, steady


log_density_grapevine = partial(
    jax.jit,
    static_argnames=_STATIC_SOLVE_ARGUMENTS,
)(_log_density_and_steady_states)


@partial(jax.jit, static_argnames=_STATIC_SOLVE_ARGUMENTS)
def log_density(
    free_parameters: PyTree,
    model: KineticModel,
    prior: PyTree,
    layout: ExperimentLayout | None = None,
    split: ParameterSplit | None = None,
    guess: Float[Array, "n_experiment n_state"] | None = None,
    conc_measurements: Sequence[Measurement] = (),
    enzyme_measurements: Sequence[Measurement] = (),
    flux_measurements: Sequence[Measurement] = (),
    ivp_rtol: float = 1e-9,
    ivp_atol: float = 1e-9,
    steady_state_rtol: float = 1e-12,
    steady_state_atol: float = 1e-12,
    solver: diffrax.AbstractSolver | None = None,
    stepsize_controller: diffrax.AbstractStepSizeController | None = None,
) -> Scalar:
    """Get the log posterior density of a kinetic model's parameters.

    Takes the same arguments as log_density_grapevine, which documents them,
    but returns only the log density.
    """
    density, _ = _log_density_and_steady_states(
        free_parameters,
        model,
        prior,
        layout,
        split,
        guess,
        conc_measurements,
        enzyme_measurements,
        flux_measurements,
        ivp_rtol,
        ivp_atol,
        steady_state_rtol,
        steady_state_atol,
        solver,
        stepsize_controller,
    )
    return density


TimeGrids = tuple[tuple[tuple[str, ...], tuple[float, ...]], ...]


def get_time_grids(
    layout: ExperimentLayout,
    times: Mapping[str, Mapping[str, float]],
) -> TimeGrids:
    """Check each experiment's time grid and freeze the grids in layout order.

    Each experiment in the layout needs a grid, and no other experiments may
    have one. A grid maps time labels to times, which must increase in the
    order given, start at 0 or later and end after 0. Time labels must be
    non-empty and must not contain the separator.
    """
    missing = [e for e in layout.experiments if e not in times]
    extra = [e for e in times if e not in layout.experiments]
    if missing or extra:
        msg = (
            "times needs a grid for each experiment in the layout, "
            f"{list(layout.experiments)}, and no others. Missing: {missing}. "
            f"Extra: {extra}."
        )
        raise ValueError(msg)
    grids = []
    for experiment in layout.experiments:
        labels = tuple(times[experiment])
        values = tuple(float(times[experiment][label]) for label in labels)
        bad_labels = [label for label in labels if not label or SEP in label]
        if bad_labels:
            msg = (
                f"Time labels must be non-empty and must not contain {SEP!r}, "
                f"but {experiment!r} has {bad_labels}."
            )
            raise ValueError(msg)
        increasing = all(a < b for a, b in zip(values, values[1:]))
        if not values or values[0] < 0 or values[-1] <= 0 or not increasing:
            msg = (
                f"The times of {experiment!r} must increase, start at 0 or "
                f"later and end after 0, but they are {list(values)}."
            )
            raise ValueError(msg)
        grids.append((labels, values))
    return tuple(grids)


def check_time_course_parameters(
    model: KineticModel,
    layout: ExperimentLayout,
    split: ParameterSplit | None,
) -> None:
    """Raise unless the parameters can describe a time course."""
    if INITIAL_STATE_PARAMETER not in layout.labelling:
        msg = (
            f"A time course needs the {INITIAL_STATE_PARAMETER} parameter. "
            "Merge get_initial_state_labelling(model) into the labelling the "
            "layout is made from."
        )
        raise ValueError(msg)
    if not model.moiety_label_species:
        return
    if split is None or split.free_positions.get("moiety_totals"):
        msg = (
            "A time course takes the conserved moiety totals from the initial "
            "amounts and ignores the moiety_totals parameter, so it must be "
            "fixed with a split."
        )
        raise ValueError(msg)


@partial(jax.jit, static_argnames=("grids", "get_measurables"))
def _log_density_time_course(
    free_parameters: PyTree,
    model: KineticModel,
    prior: PyTree,
    grids: TimeGrids,
    layout: ExperimentLayout,
    split: ParameterSplit | None,
    conc_measurements: Sequence[Measurement],
    enzyme_measurements: Sequence[Measurement],
    flux_measurements: Sequence[Measurement],
    get_measurables: Callable,
) -> Scalar:
    if split is not None:
        params = combine_parameters(split, free_parameters)
    else:
        params = free_parameters
    parameters_by_experiment = gather_parameters_by_experiment(layout, params)
    amount_initial = jnp.exp(
        parameters_by_experiment[INITIAL_STATE_PARAMETER],
    )
    n_time = max(len(values) for _, values in grids)
    ts = jnp.array(
        [values + values[-1:] * (n_time - len(values)) for _, values in grids],
    )
    found, conc, flux, _ = jax.vmap(
        lambda parameters, amounts, ts: get_measurables(
            model,
            parameters,
            amounts,
            ts,
        ),
    )(parameters_by_experiment, amount_initial, ts)
    time_labels = dict(
        zip(layout.experiments, (labels for labels, _ in grids)),
    )
    conc_hat, enz_hat, flux_hat = (
        jax.tree.map(pick, measurements, is_leaf=match_measurement)
        for measurements, pick in (
            (
                conc_measurements,
                lambda m: pick_concs_by_time(
                    model,
                    pick_experiment(layout, conc, m.experiment),
                    time_labels[m.experiment],
                    m.ids,
                ),
            ),
            (
                enzyme_measurements,
                lambda m: pick_enzymes(
                    model,
                    pick_experiment(
                        layout,
                        parameters_by_experiment,
                        m.experiment,
                    ),
                    m.ids,
                ),
            ),
            (
                flux_measurements,
                lambda m: pick_fluxes_by_time(
                    model,
                    pick_experiment(layout, flux, m.experiment),
                    time_labels[m.experiment],
                    m.ids,
                ),
            ),
        )
    )
    llik_conc, llik_enz, llik_flux = (
        jax.tree.map(
            lambda m, yhat: m.log_likelihood(yhat).sum(),
            measurements,
            yhats,
            is_leaf=match_measurement,
        )
        for measurements, yhats in (
            (conc_measurements, conc_hat),
            (enzyme_measurements, enz_hat),
            (flux_measurements, flux_hat),
        )
    )
    llik = jax.tree.reduce(operator.add, (llik_conc, llik_enz, llik_flux), 0.0)
    log_prior = enzax_prior_logdensity(free_parameters, prior)
    return jnp.where(found.all(), log_prior + llik, jnp.nan)


def log_density_time_course(
    free_parameters: PyTree,
    model: KineticModel,
    prior: PyTree,
    times: Mapping[str, Mapping[str, float]],
    layout: ExperimentLayout | None = None,
    split: ParameterSplit | None = None,
    conc_measurements: Sequence[Measurement] = (),
    enzyme_measurements: Sequence[Measurement] = (),
    flux_measurements: Sequence[Measurement] = (),
    get_measurables: Callable = get_measurables_time_course_single_experiment,
) -> Scalar:
    """Get the log posterior density of the parameters, given time courses.

    Each experiment starts at time 0 from its initial amounts, the parameter
    log_amount_initial, and is measured at the time points in its grid.

    :param free_parameters: as for log_density_grapevine.

    :param prior: as for log_density_grapevine.

    :param times: each experiment's time grid, mapping time labels to times,
        e.g. {"exp1": {"0min": 0.0, "5min": 0.083}}. The times must increase
        in the order given. Measurement ids name a time point by its label,
        as {species}|{time label} or {reaction}|{time label}.

    :param layout: which parameter values each experiment has. Its labelling
        must include log_amount_initial: make it from the model's labelling
        merged with get_initial_state_labelling(model). Defaults to a single
        experiment, exp1, with nothing experiment-specific.

    :param split: which parameters are free, and the values of the rest. A
        model with conserved moieties needs one that fixes moiety_totals,
        which a time course ignores.

    :param conc_measurements: measurements of species concentrations, with
        time labels in their ids.

    :param enzyme_measurements: measurements of enzyme concentrations, which
        do not change over time, so their ids have no time labels.

    :param flux_measurements: measurements of fluxes, with time labels in
        their ids.

    :param get_measurables: the function that solves one experiment's time
        course: get_measurables_time_course_single_experiment, the default,
        or get_measurables_time_course_single_experiment_dae, optionally
        wrapped with functools.partial to change its solver options. Create
        it once and reuse it, since a new one means a new compilation.

    :return: the log density, which is NaN if any experiment's solve failed.
    """
    if layout is None:
        layout = get_experiment_layout(
            model.parameter_labelling | get_initial_state_labelling(model),
            [DEFAULT_EXPERIMENT],
            {},
        )
    check_time_course_parameters(model, layout, split)
    return _log_density_time_course(
        free_parameters,
        model,
        prior,
        get_time_grids(layout, times),
        layout,
        split,
        conc_measurements,
        enzyme_measurements,
        flux_measurements,
        get_measurables,
    )
