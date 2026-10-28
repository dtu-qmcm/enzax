import jax
import pytest
from jax import numpy as jnp

from enzax.array_types import ParamLeaf
from enzax.examples import (
    conserved_moiety,
    gdp_fucose,
    glycolysis,
    linear,
    methionine,
)
from enzax.experiments import gather_parameters_by_experiment, pick_experiment
from enzax.steady_state import get_steady_state
from enzax.time_course import (
    get_initial_ode_state,
    get_measurables_time_course_single_experiment,
)


@pytest.mark.parametrize(
    ["model", "steady_state", "parameters"],
    [
        (methionine.model, methionine.steady_state, methionine.parameters),
        (linear.model, linear.steady_state, linear.parameters),
        (
            conserved_moiety.model,
            conserved_moiety.steady_state,
            conserved_moiety.parameters,
        ),
        (
            glycolysis.model,
            glycolysis.steady_state,
            glycolysis.parameters,
        ),
    ],
)
def test_dndt(model, steady_state, parameters):
    """Test for near-zero dndt at a known steady state."""

    dndt = model.dndt(steady_state, parameters)
    zero = jnp.full((len(steady_state),), 0.0)
    assert jnp.isclose(dndt, zero).all()


def test_conserved_moiety_is_conserved():
    pool: ParamLeaf = conserved_moiety.parameters["moiety_totals"]
    log_unbalanced: ParamLeaf = conserved_moiety.parameters[
        "log_conc_unbalanced"
    ]

    def get_conc(ind):
        balanced = conserved_moiety.model.get_balanced_conc(
            ind,
            conserved_moiety.parameters,
        )
        return conserved_moiety.model.get_conc(balanced, log_unbalanced)

    guess = jnp.full(conserved_moiety.steady_state.shape, 1e-3)
    steady = get_steady_state(
        conserved_moiety.model,
        guess,
        conserved_moiety.parameters,
    )
    species = conserved_moiety.model.species
    ix_conserved = jnp.array([species.index("X1_c"), species.index("X2_c")])
    conc_steady = get_conc(steady)
    conserved_sum_steady = conc_steady[ix_conserved].sum()
    assert jnp.isclose(conserved_sum_steady, pool[0]).all()


def test_the_cascade_data_load_for_every_experiment():
    experiments = (
        gdp_fucose.FITTING_EXPERIMENTS + gdp_fucose.VALIDATION_EXPERIMENTS
    )
    times, measurements = gdp_fucose.load_measurements(experiments)
    assert list(times) == experiments
    for measurement in measurements:
        assert len(measurement.ids) == len(measurement.values)
        assert (measurement.values > 0).all()
        labels = {i.split("|")[1] for i in measurement.ids}
        assert labels <= set(times[measurement.experiment])


def get_cascade_parameters(experiment):
    by_experiment = gather_parameters_by_experiment(
        gdp_fucose.layout,
        gdp_fucose.parameters,
    )
    return pick_experiment(gdp_fucose.layout, by_experiment, experiment)


def test_the_cascade_solves_and_conserves_its_moieties():
    parameters = get_cascade_parameters("FE13")
    times, _ = gdp_fucose.load_measurements(["FE13"])
    found, conc, _, _ = get_measurables_time_course_single_experiment(
        gdp_fucose.model,
        parameters,
        jnp.exp(parameters["log_amount_initial"]),
        jnp.array(list(times["FE13"].values())),
    )
    assert found
    species = gdp_fucose.model.species
    adenine = conc[:, [species.index(s) for s in ("atp", "adp", "amp")]]
    assert jnp.allclose(adenine.sum(axis=1), 6.0)


def test_the_cascade_conserves_its_moieties_when_polyphosphate_runs_out():
    model = gdp_fucose.model
    parameters = get_cascade_parameters("FE18_1FKP")
    amount_initial = (
        jnp.exp(parameters["log_amount_initial"])
        .at[model.balanced_species.index("polyp")]
        .set(15.0)
    )
    found, conc, _, _ = get_measurables_time_course_single_experiment(
        model,
        parameters,
        amount_initial,
        jnp.linspace(0.01, 24.0, 100),
    )
    balanced = [model.species.index(s) for s in model.balanced_species]
    _, totals_initial = get_initial_ode_state(model, amount_initial)
    totals = jax.vmap(lambda amounts: get_initial_ode_state(model, amounts)[1])(
        conc[:, balanced],
    )
    assert found
    assert conc.min() >= 0
    assert jnp.allclose(totals, totals_initial, rtol=0.0, atol=1e-9)
