# Time courses

As well as finding and differentiating a kinetic model's steady state, enzax can also solve and differentiate initial value problems. This page demonstrates how, using the cell-free cascade in `enzax.examples.gdp_fucose`, which makes GDP-fucose from GMP, fucose and ATP. It was fitted to five experiments by Huber et al. (2023), each with its own enzyme loadings and starting amounts.

## The initial state

To run a time course enzax needs the initial amount of every balanced species. It looks for these amounts in a special parameter called `log_amount_initial`. To add the labels for `log_amount_initial` to a model's parameter labelling, use the function `get_initial_state_labelling` as follows:

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

A species that starts at zero should have a fixed value of `-inf` in `log_amount_initial`, i.e. the logarithm of zero.

Time courses ignore the `moiety_totals` parameter as moiety totals are fixed by the initial amounts. Species in a rapid equilibrium need not start at equilibrium: the solve equilibrates them.

The initial amounts and enzyme loadings differ between the cascade's experiments, so `gdp_fucose.parameters` holds every experiment's values, with the labelling of `gdp_fucose.layout` (see [statistical modelling](statistical_modelling.md) for how experiments share parameters). `pick_experiment` gets one experiment's values, as required by the kinetic model:

```python
from enzax.experiments import (
    gather_parameters_by_experiment,
    pick_experiment,
)

params_by_experiment = gather_parameters_by_experiment(
    gdp_fucose.layout,
    gdp_fucose.parameters,
)
fe13_parameters = pick_experiment(
    gdp_fucose.layout,
    params_by_experiment,
    "FE13",
)
jnp.exp(fe13_parameters["log_amount_initial"])
```

```
Array([  0. ,   0. ,   6. ,   0. ,   0. ,  21.5,   0. ,   0. ,   0. ,
         0. ,  21.5, 100. ], dtype=float64)
```

## Solving a time course

`get_measurables_time_course_single_experiment` solves one experiment from time 0 and returns whether the solve succeeded, then every species' concentration, every reaction's flux and every algebraic variable at each of the times it is given:

```python
from enzax.time_course import get_measurables_time_course_single_experiment

found, conc, flux, algebraic_variables = (
    get_measurables_time_course_single_experiment(
        model,
        fe13_parameters,
        jnp.exp(fe13_parameters["log_amount_initial"]),
        jnp.array([1.0, 5.0, 24.0]),
    )
)
conc[:, model.species.index("gdp_fucose")]
```

The solve integrates the model with diffrax's stiff solver `Kvaerno5`, solving any rapid equilibria and algebraic constraints at every step. Gradients go through the solver's steps. If a solve fails, `found` is `False` and the other outputs are computed at the initial state, so they are finite but meaningless.

`get_measurables_time_course_single_experiment_dae` takes the same arguments and returns the same things, but integrates the model as a differential algebraic equation with the BDF solver from [diffrax-bdf](https://github.com/dtu-qmcm/diffrax-bdf), a Python package which needs installing separately. For models with rapid equilibria or algebraic constraints it takes fewer steps than solving them at every step.

## Measurements over time

To represent time course measurements, first make a grid of named time points for each experiment, in the form of a map from labels to floating point times. Time points must increase in the order given:

```python
times = {"FE13": {"5min": 0.083, "50min": 0.83, "145min": 2.42}}
```

A measurement object can name these time points after a `|` character in its `ids`:

```python
from enzax.measurements import LognormalMeasurement

gdp_fucose_measurement = LognormalMeasurement(
    experiment="FE13",
    ids=["gdp_fucose|50min", "gdp_fucose|145min"],
    values=jnp.array([1.33, 2.71]),
    errors=jnp.array([0.1, 0.1]),
)
```

One measurement can cover several species and several time points. `gdp_fucose.load_measurements` builds the grids and the measurements for the cascade's data, labelling each time point by its minute. For statistical models of your own, `pick_concs_by_time` and `pick_fluxes_by_time` pick a measurement's predicted values out of a time course by id.

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
    [
        pick_experiment(gdp_fucose.layout, params_by_experiment, e)
        for e in experiments
    ],
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
)
```

The layout must be made from a labelling that includes `log_amount_initial`, and a model with conserved moieties needs a split that fixes `moiety_totals`; the density raises otherwise. Concentration and flux measurements name time points in their ids; enzyme measurements do not, since enzyme levels do not change over time.

`get_measurables` chooses the solve: the default, `get_measurables_time_course_single_experiment`, or `get_measurables_time_course_single_experiment_dae`, either of them wrapped with `functools.partial` to change its options. Create it once and reuse it, since each new one is compiled anew. As with `log_density`, the density is NaN if any experiment's solve fails, and experiments are solved together with `jax.vmap`.

`scripts/mcmc_demo.py` shows how to sample from a log density like this one. `scripts/cascade_demo.py` fits the cascade's free parameters to the five experiments by maximum a posteriori optimisation and compares the fit with two validation experiments.
