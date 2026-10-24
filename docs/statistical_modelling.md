# Statistical modelling

One of the main reasons to make a differentiable kinetic model with enzax is to embed it inside a statistical model. This makes it possible to infer kinetic parameters from quantitative measurements and background information, opening a wide range of possible uses.

Enzax aims to support applications beyond statistical modelling, such as optimisation and machine learning, so it focuses on providing kinetic modelling primitives, leaving it somewhat up to the user to implement an enzax-friendly statistical model using their favourite modelling framework.

However, enzax does provide some utility functionality specifically focused on statistical modelling, as well as plenty of worked examples.

Users are highly encouraged to post more examples to enzax's wiki: <https://github.com/dtu-qmcm/enzax/wiki>.

## Fixing parameters

Quite often when statistical modelling with kinetic models, you are only interested in uncertainty related to certain parameters and want to treat  all other parameters as if they were known exactly.

The simplest way to do this is to not include the known parameters in the kinetic model in the first place. However, it can get tedious to rewrite the model every time you want to change which parameters are fixed. More conveniently, enzax can split an existing set of parameters into free and fixed ones. Here's how to do it.

In this example, we fix some parameters of the `methionine` model provided by enzax, a medium-to-small sized model that describes the mammalian methionine cycle. We can load this model and its parameters as follows:

```python
from enzax.examples.methionine import model, parameters as true_parameters
true_parameters
```

The parameters are a dictionary with one flat array per parameter. Which label sits at which position is recorded in `model.parameter_labelling`:

```python
model.parameter_labelling["log_kcat"]
```

```
('MAT1', 'MAT3', 'METH-Gen', 'GNMT1', 'AHC1', 'MS1', 'BHMT1', 'CBS1',
 'MTHFR1', 'PROT1')
```

Suppose we want a statistical model where everything is fixed except MAT1's $k_{cat}$, the temperature and the formation energies. We say so by label:

```python
from enzax.parameter_split import (
    combine_parameters,
    count_free_parameters,
    get_free_labels,
    get_free_parameters,
    split_parameters_by_freeing,
)

split = split_parameters_by_freeing(
    model.parameter_labelling,
    true_parameters,
    {"log_kcat": ["MAT1"], "temperature": None, "dgf": None},
)
count_free_parameters(split)
```

```
21
```

A parameter mapped to a list of labels frees exactly those values; a parameter mapped to `None` frees the whole thing. `temperature` has no labels, so `None` is the only way to free it. Anything not mentioned is fixed, and its value is taken from the parameter set you passed in. There is a `split_parameters_by_fixing` for when it is more convenient to say which parameters are *not* free.

`get_free_parameters` then pulls the free parameters out:

```python
free_parameters = get_free_parameters(split, true_parameters)
{k: v.shape for k, v in free_parameters.items()}
```

```
{'log_kcat': (1,), 'dgf': (19,), 'temperature': ()}
```

Note that `free_parameters["log_kcat"]` has one element, not ten. The free parameters are *gathered*, not masked, so a fixed parameter is genuinely absent rather than present-but-ignored. That matters for inference: a masked coordinate would still be part of the sampler's state space and would still be explored, and a prior built from the free parameters would be a prior on parameters that are not being inferred. `get_free_labels` says which value each position holds:

```python
get_free_labels(split, "log_kcat")
```

```
('MAT1',)
```

We can use `free_parameters` when we want to do uncertainty-related things, like for example applying some random perturbations:

```python
import jax

key = jax.random.key(1234)
leaves, treedef = jax.tree.flatten(free_parameters)
keys = jax.tree.unflatten(treedef, jax.random.split(key, num=len(leaves)))
new_free_parameters = jax.tree.map(
    lambda leaf, k: leaf + jax.random.normal(k, shape=leaf.shape) * 0.1,
    free_parameters,
    keys
)
new_free_parameters
```

When we want the fixed parameters back in, `combine` scatters the free values and the fixed ones into full-size arrays:

```python
new_parameters = combine_parameters(split, new_free_parameters)
new_parameters
```

`log_density` does this combining itself, so a statistical model over a subset of the parameters only needs the split. The next three sections describe the other ingredients: measurements, the log density that brings everything together, and how to fit several experiments at once.

## Measurements

A measurement holds some measured values from one experiment, together with a method that scores them against the model's predictions. Enzax provides two kinds of measurement:

- `LognormalMeasurement`, for positive quantities such as concentrations and enzyme levels. The logarithms of the values are normally distributed, and `errors` are standard deviations on the log scale.
- `NormalMeasurement`, for quantities that can be negative, such as fluxes. The values are normally distributed, and `errors` are standard deviations.

Each measurement names its experiment and the ids of its values: species ids for concentrations, reaction ids for fluxes, and labels from `model.parameter_labelling["log_enzyme"]` for enzymes.

```python
from jax import numpy as jnp

from enzax.measurements import LognormalMeasurement, NormalMeasurement

conc_measurement = LognormalMeasurement(
    experiment="exp1",
    ids=["met-L", "atp", "amet"],
    values=jnp.array([4.0e-5, 1.2e-3, 2.9e-5]),
    errors=jnp.array([0.1, 0.1, 0.1]),
)
flux_measurement = NormalMeasurement(
    experiment="exp1",
    ids=["MAT1"],
    values=jnp.array([0.04]),
    errors=jnp.array([0.005]),
)
```

Since a measurement names its values, it can cover any subset of them in any order, and a value that was not measured can simply be left out.

Other kinds of measurement are subclasses of `Measurement` with their own `log_likelihood` method. The method receives the model's predicted values for the measurement's ids, in the same order, and returns one log density per independent observation. For example, a compositional measurement of the proportions of some species, such as the isotopologues of one metabolite, could use a Dirichlet distribution:

```python
import jax
from jax.scipy.stats import dirichlet

from enzax.measurements import Measurement


class DirichletMeasurement(Measurement):
    proportions: jax.Array
    concentration: float

    def log_likelihood(self, predicted):
        alpha = self.concentration * predicted / predicted.sum()
        return jnp.atleast_1d(dirichlet.logpdf(x=self.proportions, alpha=alpha))
```

## The log density

`log_density` brings the pieces together. It combines the free parameters with the fixed ones, solves for the steady state, and adds the log prior to the log likelihood of every measurement. With several experiments it also needs an experiment layout, which the next section describes. Without one it assumes a single experiment called `exp1`, which is why the measurements above name that experiment:

```python
import functools

from enzax.statistical_modelling import log_density, prior_from_truth

posterior_log_density = functools.partial(
    log_density,
    model=model,
    split=split,
    prior=prior_from_truth(free_parameters, sd=0.1),
    conc_measurements=[conc_measurement],
    flux_measurements=[flux_measurement],
)
posterior_log_density(free_parameters)
```

Measurements come in three lists, by what they measure: `conc_measurements`, `enzyme_measurements` and `flux_measurements`. Any `Measurement` subclass can go in the list for the quantity its ids refer to. Leave out `split` to infer every parameter, in which case the first argument is a complete parameter set rather than a gathered one.

The steady state solve starts from `guess`, an array with one row per experiment and one column per state variable, which defaults to 0.01 everywhere. If no steady state is found, the log density is NaN, which a sampler treats as a divergence and rejects.

### Writing your own statistical model

`log_density` covers concentrations, enzymes and fluxes with known measurement errors. For anything else, such as measured algebraic variables or inferred errors, you can build a statistical model from the same parts. `get_measurables_single_experiment` solves one experiment's steady state and returns it, whether it was found, and every species' concentration and every reaction's flux. `pick_concs`, `pick_fluxes`, `pick_enzymes` and `pick_algebraic_variables` then pick a measurement's predicted values out of these by id:

```python
from enzax.measurements import pick_concs
from enzax.statistical_modelling import get_measurables_single_experiment

steady, found, conc, flux = get_measurables_single_experiment(
    model,
    true_parameters,
    jnp.full(len(model.ode_state_species_ix), 0.01),
)
conc_measurement.log_likelihood(pick_concs(model, conc, conc_measurement.ids))
```

## Multiple experiments

Experiments often share a model but not all its parameter values: enzyme levels or boundary concentrations usually differ between experiments, while kinetic constants do not. An experiment layout says which values are experiment-specific, using the same `{parameter: labels}` selections as a split:

```python
from enzax.experiments import get_experiment_layout

layout = get_experiment_layout(
    model.parameter_labelling,
    ["exp1", "exp2"],
    {"log_enzyme": None},
)
layout.labelling["log_enzyme"][:4]
```

```
('MAT1|exp1', 'MAT1|exp2', 'MAT3|exp1', 'MAT3|exp2')
```

The layout's labelling expands the model's: each experiment-specific label appears once per experiment, as `{label}|{experiment}`, and every other label once. A parameter set with this labelling holds every experiment's values, and it is an ordinary parameter set, so splits and priors work on it as before. Parameters without labels, such as `temperature`, can only be shared.

The easiest way to make such a parameter set is from one parameter set per experiment, in the layout's order. `merge_experiment_parameters` raises if the experiments disagree about a shared value:

```python
from enzax.experiments import merge_experiment_parameters

more_mat1 = true_parameters | {
    "log_enzyme": true_parameters["log_enzyme"].at[0].add(1.0),
}
expanded_parameters = merge_experiment_parameters(
    layout,
    [true_parameters, more_mat1],
)
split = split_parameters_by_freeing(
    layout.labelling,
    expanded_parameters,
    {"log_kcat": None},
)
free_parameters = get_free_parameters(split, expanded_parameters)
```

Note that the split uses `layout.labelling`, not the model's. Here every experiment's enzyme levels are fixed at their own values, and the turnover numbers are free and shared.

Each measurement names the experiment it comes from, and `log_density` scores it against that experiment's predictions. The guess needs one row per experiment:

```python
posterior_log_density = functools.partial(
    log_density,
    model=model,
    layout=layout,
    split=split,
    prior=prior_from_truth(free_parameters, sd=0.1),
    guess=jnp.full((2, len(model.ode_state_species_ix)), 0.01),
    conc_measurements=[
        conc_measurement,
        LognormalMeasurement(
            experiment="exp2",
            ids=["met-L", "atp", "amet"],
            values=jnp.array([3.1e-5, 1.0e-3, 4.4e-5]),
            errors=jnp.array([0.1, 0.1, 0.1]),
        ),
    ],
)
posterior_log_density(free_parameters)
```

`log_density` solves all the experiments' steady states together, with `jax.vmap`. When the experiments need similar amounts of solver work this is much cheaper than solving them one at a time. When one experiment needs much more work than the others, all of them wait for it, which in our tests cost up to about 1.5 times as much as solving them one at a time. For your own statistical models, `gather_parameters_by_experiment` turns an expanded parameter set into each experiment's parameters, stacked along a leading axis ready for `jax.vmap`.

## Posterior sampling

Enzax does not ship a sampler: it should work with any JAX-compatible MCMC
library.

The samplers in [blackjax](https://blackjax-devs.github.io/blackjax/) are a
good fit, since they accept arbitrary differentiable log densities.
[blackjax-utils](https://github.com/teddygroves/blackjax-utils) wraps blackjax's
NUTS sampler behind a single function that handles warmup, multiple chains and
the sampling loop. It is not on PyPI, so install it from GitHub:

```bash
uv add git+https://github.com/teddygroves/blackjax-utils.git
```

Sampling then looks like this:

```python
import blackjax
import jax
from blackjax_utils import run_nuts

with blackjax.progress_bar("enzax NUTS"):
    states, info = run_nuts(
        key=jax.random.key(1234),
        log_posterior=posterior_log_density,
        init_params=free_parameters,
        init_sd=0.01,
        n_chain=4,
        n_warmup=200,
        n_sample=200,
        warmup_options=dict(initial_step_size=0.01),
    )
```

`init_params` gives the shape of the parameter set to sample, and `init_sd`
jitters each chain's starting point away from it. `states.position` is a PyTree
with the same structure as `free_parameters`, whose leaves have shape
`(n_chain, n_sample, ...)`, and `info.is_divergent` has shape
`(n_chain, n_sample)`, so `info.is_divergent.any()` tells you whether the
sampler hit a divergence.

`warmup_options` is passed to blackjax's `window_adaptation` and nothing else,
which is where `initial_step_size` has to go. `target_acceptance_rate` and
`is_mass_matrix_diagonal` belong there too. Any further keyword argument
reaches both `window_adaptation` and the NUTS kernel, so it has to be one that
both accept, such as `max_num_doublings`; use `sampling_options` for arguments
meant for the sampling stage alone.

Setting the initial step size is necessary for many kinetic models: if it is too high the sampler may fail at its first attempt to traverse the parameter space.

## Watching a sampler run

blackjax has a progress bar, which the example above wraps around the call. It
works by patching `jax.lax.scan` for the duration of the `with` block, so it
has to enclose the call that *traces* the sampler rather than merely one that
runs an already-compiled one, and it reports once per step for all chains
together. Warmup and sampling are separate scans, so the bar fills up twice. See [here](https://blackjax-devs.github.io/blackjax/examples/howto_progress_bar.html) for more about blackjax's progress bar.

The progress bar needs blackjax's `progress` optional extra, which you can install with `pip install 'blackjax[progress]'`.

`scripts/mcmc_demo.py` in the enzax repository is a complete worked example: it simulates measurements from three methionine experiments that differ in their enzyme levels, then samples the shared turnover numbers with grapevine (see below). Run it with
```bash
uv run --group mcmc python scripts/mcmc_demo.py
```

## Optimised Hamiltonian Monte Carlo with grapevine

Every leapfrog step above costs one steady state solve, and every one of them
starts from the same fixed guess. The
[grapevine](https://github.com/dtu-qmcm/grapevine) method instead starts each
solve from the steady state found at the previous point on the same Hamiltonian
trajectory. Adjacent points have nearly identical steady states, so the solver
starts essentially converged. The guess resets to the default at the end of
every MCMC iteration, so the reuse is within a trajectory only.

grapevine is not on PyPI, so install it from GitHub:

```bash
uv add git+https://github.com/dtu-qmcm/grapevine.git
```

The solution needs a way out of the log density, so `log_density_grapevine` takes the same arguments as `log_density` and returns `(log_density, steady_states)`. Do not bind its `guess` argument, since the sampler supplies it. The default guess has one row per experiment:

```python
from blackjax_utils import run_sampler
from grapevine import grapenuts

from enzax.statistical_modelling import log_density_grapevine

posterior_log_density = functools.partial(
    log_density_grapevine,
    model=model,
    layout=layout,
    split=split,
    prior=prior,
    conc_measurements=conc_measurements,
)
default_guess = jnp.full((len(layout.experiments), len(model.ode_state_species_ix)), 0.01)
states, info = run_sampler(
    key=jax.random.key(1234),
    log_posterior=posterior_log_density,
    init_params=free_parameters,
    init_sd=0.01,
    n_chain=4,
    n_warmup=200,
    n_sample=200,
    warmup_options=dict(initial_step_size=0.01),
    sampler=grapenuts(default_guess),
)
```

`grapenuts` supplies the warmup and the sampling kernel; everything else is as it was above. It also takes a `guess_fn`, which decides how the previous solution becomes the next guess. The default, `guess_previous`, reuses it as it is. `guess_implicit` instead takes an Euler step from it, which needs the residual whose root the steady state is. `guess_implicit` works with a vector, so with several experiments the guess is flattened around it:

```python
from grapevine import guess_implicit

from enzax.experiments import gather_parameters_by_experiment
from enzax.parameter_split import combine_parameters

n_experiment = len(layout.experiments)


def target_function(conc_ind, position):
    parameters = gather_parameters_by_experiment(
        layout,
        combine_parameters(split, unflatten(position)),
    )
    conc_ind = conc_ind.reshape(n_experiment, -1)
    return jax.vmap(model.dndt)(conc_ind, parameters).ravel()


def guess_fn(guess_inputs, position):
    shape = guess_inputs.solution.shape
    flat_inputs = guess_inputs._replace(solution=guess_inputs.solution.ravel())
    return guess_implicit(flat_inputs, position, target_function).reshape(shape)


sampler = grapenuts(default_guess, guess_fn=guess_fn)
```

The position a `guess_fn` receives is the sampler's, which under
blackjax-utils' default `flatten=True` is `init_params` ravelled into a single
array; `unflatten` above is the second return value of
`jax.flatten_util.ravel_pytree(free_parameters)`.

### A caveat about tolerances

grapevine is exactly valid when the solver reaches the same answer whatever
guess it starts from. The integration stops once
`norm(dndt) < steady_state_atol + steady_state_rtol * norm(conc)`, so its
terminal state does depend on where it started, and a guess that already meets
that bound is returned unchanged.

`log_density` uses `get_steady_state_hybrid`, which narrows this rather
than widening it: when the guess is close enough for its Newton step to be
accepted, the answer is a root converged to machine precision and so does not
depend on the guess at all. What is left is the gap between that root and the
point the integration stops at, which on the methionine model is about `1e-7`
in the log density.

This makes the event tolerances matter more than they otherwise would, and they
have to be tight relative to the concentrations. Methionine's are of order
`1e-5`, so at an atol of `1e-9` the solve stops at a residual only `1e-4`
relative to the state, and the log density moves by around `1e-3` depending on
the guess. At the default `1e-12` it moves by around `1e-7`. Tightening the
event is cheap, because it decides only when to stop rather than how finely to
integrate; tightening `ivp_atol` and `ivp_rtol` is expensive and does not help
here. `tests/test_lp_grad.py::test_log_density_is_guess_invariant` measures
this on the methionine model.
