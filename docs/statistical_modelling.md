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

## Optimised Hamiltonian Monte Carlo with grapevine
