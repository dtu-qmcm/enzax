# Statistical modelling

One of the main reasons to make a differentiable kinetic model with enzax is to embed it inside a statistical model. This makes it possible to infer kinetic parameters from quantitative measurements and background information, opening a wide range of possible uses.

Enzax aims to support applications beyond statistical modelling, such as optimisation and machine learning, so it focuses on providing kinetic modelling primitives, leaving it somewhat up to the user to implement an enzax-friendly statistical model using their favourite modelling framework.

However, enzax does provide some utility functionality specifically focused on statistical modelling, as well as plenty of worked examples.

Users are highly encouraged to post more examples to enzax's wiki: <https://github.com/dtu-qmcm/enzax/wiki>.

## Fixing parameters

Quite often when statistical modelling with kinetic models, you are only interested in uncertainty related to certain parameters and want to treat  all other parameters as if they were known exactly.

The simplest way to do this is to not include the known parameters in the kinetic model in the first place. However, it can get tedious to rewrite the model every time you want to change which parameters are fixed. More conveniently, it is possible to mask an existing set of parameters. Here's how to do it.

In this example, we fix some parameters of the `methionine` model provided by enzax, a medium-to-small sized model that describes the mammalian methionine cycle. We can load this model and its parameters as follows:

```python
from enzax.examples.methionine import parameters as true_parameters
true_parameters
```

The parameters are a pretty large dictionary. Suppose we want to make a statistical model where all parameters are fixed except for the $k_{cat}$ parameter of enzyme MAT1 and the temperature parameter. The first step is to define a function taking in the whole set of parameters and returning a tuple of the parameters we want to leave free:

```python
def get_free_params(params):
    return params["log_kcat"]["MAT1"], params["temperature"]
```

Next, we use some functions from the library [equinox](https://github.com/patrick-kidger/equinox)---`tree_at` and `partition`---as well as `jax.tree.map`to create dictionaries of free and fixed parameters based on our function.

```python
import jax
import equinox as eqx

is_free = eqx.tree_at(
    get_free_params,
    jax.tree.map(lambda _: False, true_parameters),
    replace_fn=lambda _: True,
)
free_parameters, fixed_parameters = eqx.partition(true_parameters, is_free)

free_parameters
```

`free_parameters` is the same as `true_parameters`, but all arrays apart from the ones we want to be free have been replaced by `None`. Similarly, `fixed_parameters` is the same as `true_parameters`, but with `None`s in place of the free parameters.

This is what we want! We can use `free_parameters` when we want to uncertainty-related things, like for example applying some random perturbations:

```python
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

When we want to add the fixed parameters back in, we can use equinox's `combine` function:

```python
new_parameters = eqx.combine(new_free_parameters, fixed_parameters)
new_parameters
```

Note that this method of splitting fixed and free parameters works for arbitrary pytrees, and can easily be adjusted so that the fixed parameters rather than the free ones are user-specified.

## Posterior sampling

## Optimised Hamiltonian Monte Carlo with grapevine
