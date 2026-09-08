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

This is what `enzax_log_density` does internally, so a Bayesian model over a subset of the parameters is as follows. The `measurements` argument is described in the next section:

```python
import functools
from enzax.statistical_modelling import enzax_log_density, prior_from_truth

posterior_log_density = functools.partial(
    enzax_log_density,
    model=model,
    split=split,
    measurements=measurements,
    prior=prior_from_truth(free_parameters, sd=0.1),
)
```

Leave out `split` to infer every parameter, in which case the first argument is a complete parameter set rather than a gathered one.

## Measurement order

`enzax_log_density` compares three kinds of measurement against the model's predictions, and each has an order you have to match:

- concentrations are in the model's `species` order;
- fluxes are in its `reactions` order;
- enzyme concentrations are in `model.parameter_labelling["log_enzyme"]` order, i.e. the order the model's rate equations first label their enzymes in.

The last of these is not the same as the reaction order whenever a reaction has no enzyme, as with methionine's drain reaction, or whenever two reactions share one.

## Posterior sampling

Enzax does not ship a sampler. Its job ends at the log density: once you have a
callable that maps parameters to a log probability, any JAX-compatible MCMC
library can sample from it.

The samplers in [blackjax](https://blackjax-devs.github.io/blackjax/) are a
good fit, since they work with arbitrary differentiable log densities.
[blackjax-utils](https://github.com/teddygroves/blackjax-utils) wraps blackjax's
NUTS sampler behind a single function that handles warmup, multiple chains and
the sampling loop, which is roughly what you want for a kinetic model.

It is not on PyPI, so install it from GitHub:

```bash
uv add git+https://github.com/teddygroves/blackjax-utils.git
```

Enzax declares it in an optional dependency group called `mcmc`, so if you are
working in a clone of the enzax repository you can instead run

```bash
uv sync --group mcmc
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

Setting the initial step size is not optional for a kinetic model, and it is
the setting that decides how long sampling takes. Every leapfrog step costs one
steady state solve, so the bill is the number of leapfrog steps times the cost
of a solve, and the step size decides the first factor. Too small, and NUTS
never triggers its U-turn criterion: each trajectory runs to the
`2 ** max_num_doublings - 1` ceiling, buying thousands of solves per draw. Too
large, and the first proposal leaves the region where the model has a steady
state, at which point enzax raises "Binding polynomial is not positive!"
instead of reporting a log density of -inf. On the methionine example 0.01
works and 1.0 (blackjax's default) fails; going from 1e-4 to 1e-2 took
`scripts/mcmc_demo.py` from 7 minutes 19 seconds to 38 seconds, with better
posterior spread and no divergences.

The other factor, the cost of one solve, is set by the tolerance argument of
`enzax.steady_state.get_steady_state`. See its docstring for the trade-off.

## Watching a sampler run

blackjax has a progress bar, which the example above wraps around the call. It
works by patching `jax.lax.scan` for the duration of the `with` block, so it
has to enclose the call that *traces* the sampler rather than merely one that
runs an already-compiled one, and it reports once per step for all chains
together. Warmup and sampling are separate scans, so the bar fills up twice.

It needs blackjax's `progress` optional extra, which enzax's `mcmc` dependency
group already asks for. Outside that group, install it with
`pip install 'blackjax[progress]'`.

`scripts/mcmc_demo.py` in the enzax repository is a complete worked example:
it fixes all but a few of the methionine model's parameters, simulates
measurements from the true model, builds the posterior log density and samples
from it. Run it with

```bash
uv run --group mcmc python scripts/mcmc_demo.py
```

## Optimised Hamiltonian Monte Carlo with grapevine
