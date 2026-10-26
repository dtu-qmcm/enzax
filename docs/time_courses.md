# Time courses

Most of enzax is about steady states. Sometimes, though, a system is followed over time instead: an enzyme cascade in a closed pot, say, sampled every hour or so as it turns its substrates into product. Enzax can solve and fit these initial value problems as well, differentiably, with the same models, measurements and experiment layouts as steady states.

This page uses the cell-free cascade in `enzax.examples.gdp_fucose`, which makes GDP-fucose from GMP, fucose and ATP. It was fitted to five experiments by Huber et al. (2023), each with its own enzyme loadings and starting amounts.

## The initial state

A time course starts from the amount of every balanced species. These amounts are a parameter, `log_amount_initial`, so known starting amounts can be fixed with a split and unknown ones inferred. A model's own labelling leaves this parameter out, since a steady-state model has no use for it; `get_initial_state_labelling` gives its labelling, to merge into the model's:

```python
from jax import numpy as jnp

from enzax.examples import gdp_fucose
from enzax.time_course import get_initial_state_labelling

model = gdp_fucose.model
labelling = model.parameter_labelling | get_initial_state_labelling(model)
labelling["log_amount_initial"]
```

```
('pp', 'p', 'atp', 'adp', 'amp', 'fucose', 'fucose1p', 'gdp_fucose', 'gtp', 'gdp', 'gmp', 'polyp')
```

Three things to know about this parameter:

- **It holds amounts, not concentrations.** In a compartment of fixed volume these differ only by the volume. In a compartment whose volume is itself solved for, only the amounts fix the state.
- **A species that starts at zero has a fixed value of `-inf`**, the logarithm of zero. Most of the cascade's species start at zero. A free initial amount must be positive.
- **The initial amounts fix the conserved moiety totals**, so a time course ignores the `moiety_totals` parameter. Species in a rapid equilibrium need not start at equilibrium; the solve equilibrates them.

`gdp_fucose.get_parameters` packs a parameter set with this labelling for one experiment:

```python
parameters = gdp_fucose.get_parameters("FE13")
jnp.exp(parameters["log_amount_initial"])
```

```
Array([ 0. ,  0. ,  6. ,  0. ,  0. , 21.5,  0. ,  0. ,  0. ,  0. , 21.5,
       30. ], dtype=float64)
```

## Solving a time course

`get_measurables_time_course_single_experiment` solves one experiment from time 0 and returns whether the solve succeeded, then every species' concentration, every reaction's flux and every algebraic variable at each of the times it is given:

```python
from enzax.time_course import get_measurables_time_course_single_experiment

found, conc, flux, algebraic_variables = (
    get_measurables_time_course_single_experiment(
        model,
        parameters,
        jnp.exp(parameters["log_amount_initial"]),
        jnp.array([1.0, 5.0, 24.0]),
        rtol=1e-6,
        atol=1e-6,
    )
)
conc[:, model.species.index("gdp_fucose")]
```

The solve integrates the model with diffrax's stiff solver `Kvaerno5`, solving any rapid equilibria and algebraic constraints at every step. Gradients go through the solver's steps. The cascade needs looser tolerances than the defaults to get through its fast reactions, which is why `gdp_fucose.get_measurables` is this function with `rtol` and `atol` set to `1e-6`. If a solve fails, `found` is `False` and the other outputs are computed at the initial state, so they are finite but meaningless.

`get_measurables_time_course_single_experiment_dae` takes the same arguments and returns the same things, but integrates the model as a differential algebraic equation with the BDF solver from [diffrax-bdf](https://github.com/dtu-qmcm/diffrax-bdf), which needs installing. For models with rapid equilibria or algebraic constraints it takes fewer steps than solving them at every step.

## Measurements over time

Each experiment has a grid of named time points, mapping a label to a time. Time points must increase in the order given:

```python
times = {"FE13": {"5min": 0.083, "50min": 0.83, "145min": 2.42}}
```

A measurement names a time point after a `|` in each id, so the existing measurement classes hold time courses as they are:

```python
from enzax.measurements import LognormalMeasurement

gdp_fucose_measurement = LognormalMeasurement(
    experiment="FE13",
    ids=["gdp_fucose|50min", "gdp_fucose|145min"],
    values=jnp.array([1.33, 2.71]),
    errors=jnp.array([0.1, 0.1]),
)
```

One measurement can cover several species and several time points, and anything that was not measured is simply left out. `gdp_fucose.load_measurements` builds the grids and the measurements for the cascade's data, labelling each time point by its minute. For statistical models of your own, `pick_concs_by_time` and `pick_fluxes_by_time` pick a measurement's predicted values out of a time course by id.

## Fitting time courses

`log_density_time_course` is the time-course counterpart of `log_density`. Besides the usual arguments it takes the time grids. Here the five fitting experiments share their kinetic parameters, while each has its own enzyme loadings and initial amounts, and only the turnover numbers are inferred:

```python
from enzax.experiments import (
    get_experiment_layout,
    merge_experiment_parameters,
)
from enzax.parameter_split import (
    get_free_parameters,
    split_parameters_by_freeing,
)
from enzax.statistical_modelling import (
    log_density_time_course,
    prior_from_truth,
)

experiments = gdp_fucose.FITTING_EXPERIMENTS
layout = get_experiment_layout(
    labelling,
    experiments,
    {"log_enzyme": None, "log_amount_initial": None},
)
expanded_parameters = merge_experiment_parameters(
    layout,
    [gdp_fucose.get_parameters(e) for e in experiments],
)
split = split_parameters_by_freeing(
    layout.labelling,
    expanded_parameters,
    {"log_kcat": None},
)
free_parameters = get_free_parameters(split, expanded_parameters)
times, conc_measurements = gdp_fucose.load_measurements(experiments)
log_density_time_course(
    free_parameters,
    model,
    prior_from_truth(free_parameters, sd=1.0),
    times,
    layout=layout,
    split=split,
    conc_measurements=conc_measurements,
    get_measurables=gdp_fucose.get_measurables,
)
```

The layout must be made from a labelling that includes `log_amount_initial`, and a model with conserved moieties needs a split that fixes `moiety_totals`; the density raises otherwise. Concentration and flux measurements name time points in their ids; enzyme measurements do not, since enzyme levels do not change over time.

`get_measurables` chooses the solve: the default, `get_measurables_time_course_single_experiment`, or `get_measurables_time_course_single_experiment_dae`, either of them wrapped with `functools.partial` to change its options. Create it once and reuse it, since each new one is compiled anew. As with `log_density`, the density is NaN if any experiment's solve fails, and experiments are solved together with `jax.vmap`.

`scripts/mcmc_demo.py` shows how to sample from a log density like this one. `scripts/cascade_demo.py` fits the cascade's turnover numbers to the five experiments by maximum a posteriori optimisation and compares the fit with two validation experiments.
