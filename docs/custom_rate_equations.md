# How to make your own rate equation

This page assumes you have read [Getting started](getting_started.md), and in particular that you have seen how to define a `RateEquationModel`, how enzax labels model parameters and how two reactions can share a parameter value by using the same label.

A rate equation says how one reaction's flux depends on the concentrations of the model's species and on the model's parameters. Enzax gives you three levels of control, in increasing order of effort:

1. **[Use a built-in rate equation](#use-a-built-in-rate-equation)**, configured with its fields. Covers most reactions.
2. **[Write your own binding polynomial](#write-your-own-binding-polynomial)** with `SaturableRateEquation`, keeping Michaelis-Menten's thermodynamics and turnover. Covers reactions whose enzyme has states that the stoichiometry does not imply, such as abortive complexes.
3. **[Write a rate equation from scratch](#write-a-rate-equation-from-scratch)** by subclassing `RateEquation`. Covers everything else.

## Use a built-in rate equation

Enzax provides two "ready-to-go" rate equations:

- `MichaelisMenten` covers reversible and irreversible reactions, competitive inhibition and allosteric regulation.
- `Drain` gives a reaction a constant flux, for representing a process at the edge of your model whose kinetics you do not want to describe.

Enzax's other built-in rate equation, `SaturableRateEquation`, is mostly the same as `MichaelisMenten`, but with the enzyme's states written out by hand, allowing for more customisation. It is the subject of the [next section](#write-your-own-binding-polynomial).

Every enzyme-catalysed rate equation in enzax has the same shape:

```
v = enzyme * kcat * numerator / Z * reversibility * allosteric factor
```

- `enzyme` and `kcat` are the enzyme's concentration and turnover number, one of each per reaction.
- `numerator` is the product of each substrate's concentration over its Michaelis constant.
- `Z` is the reaction's [binding polynomial](api/binding.md), so `1 / Z` is the fraction of the enzyme that is unbound.
- `reversibility` is the thermodynamic driving force, present unless you pass `reversible=False`.
- The `allosteric factor` is present only if the rate equation declares something about the enzyme's tense and relaxed states.

Each of the fields below either adds a state to `Z`, switches one of the optional factors on, or says which parameter value a quantity should use. The examples all extend the linear pathway model from [Getting started](getting_started.md):

```python
from enzax.kinetic_model import RateEquationModel
from enzax.rate_equations import MichaelisMenten

stoichiometry = {
    "r1": {"m1e": -1.0, "m1c": 1.0},
    "r2": {"m1c": -1.0, "m2c": 1.0},
    "r3": {"m2c": -1.0, "m2e": 1.0},
}
```

### Reversible Michaelis-Menten kinetics

`MichaelisMenten()` with nothing declared is a reversible reaction whose substrates bind in one random-order complex and whose products bind in another:

```python
model = RateEquationModel(
    stoichiometry=stoichiometry,
    balanced_species=["m1c", "m2c"],
    rate_equations={
        "r1": MichaelisMenten(),
        "r2": MichaelisMenten(),
        "r3": MichaelisMenten(),
    },
)
model.parameter_labelling["log_saturation_constant"]
```

```
('km|r1|m1e', 'km|r1|m1c', 'km|r2|m1c', 'km|r2|m2c', 'km|r3|m2c', 'km|r3|m2e')
```

Every reactant gets a Michaelis constant, because a reversible reaction's products bind the enzyme just as its substrates do. Each reaction also gets one `log_kcat` and one `log_enzyme` entry labelled with the reaction's id, and the driving force is computed from the model's `dgf` and `temperature` parameters, so a reversible flux can come out negative.

### Irreversible reactions

`reversible=False` drops the driving force, which also means the products no longer bind:

```python
"r2": MichaelisMenten(reversible=False),
...
model.parameter_labelling["log_saturation_constant"]
```

```
('km|r1|m1e', 'km|r1|m1c', 'km|r2|m1c', 'km|r3|m2c', 'km|r3|m2e')
```

`km|r2|m2c` is gone: an irreversible reaction has a Michaelis constant for each substrate only. Its flux always has the sign its stoichiometry implies.

This is also the simplest way to write a saturating drain, since one-substrate irreversible Michaelis-Menten kinetics is `v * conc / (conc + eps)` with `kcat * enzyme` as `v` and the Michaelis constant as `eps`. The `r5p_drain` and `pyr_drain` reactions in `enzax.examples.glycolysis` are declared this way.

### Competitive inhibition

A competitive inhibitor binds the free enzyme and stops it working, adding a state to `Z` but no new pathway to product:

```python
"r2": MichaelisMenten(competitive_inhibitors=["m2c"]),
...
model.parameter_labelling["log_saturation_constant"]
```

```
('km|r1|m1e', 'km|r1|m1c', 'km|r2|m1c', 'km|r2|m2c', 'ki|r2|m2c', 'km|r3|m2c', 'km|r3|m2e')
```

The new `ki|r2|m2c` is the inhibition constant. Declaring several inhibitors gives each its own state, and each its own constant.

An inhibitor does not have to be one of the reaction's reactants, or take part in any reaction at all. A species that enzax meets for the first time here joins the model as an unbalanced species, so it gets a constant concentration to declare and a formation energy of its own:

```python
"r2": MichaelisMenten(competitive_inhibitors=["atp_c"]),
...
model.parameter_labelling
```

```
...
log_saturation_constant: (..., 'ki|r2|atp_c', ...)
log_conc_unbalanced: ('m1e', 'm2e', 'atp_c')
dgf: ('m1e', 'm1c', 'm2c', 'm2e', 'atp_c')
...
```

### Allosteric regulation

An allosteric effector does not compete with the substrate; it shifts the enzyme's balance between a tense state that works poorly and a relaxed state that works well. Enzax uses the generalised Monod-Wyman-Changeux model, which multiplies the rate by

```
1 / (1 + tc * (tense / relaxed) ** subunits)
```

where `tc` is the transfer constant, and `tense` and `relaxed` are binding polynomials for the enzyme's two states.

To build a rate equation following this model, you can simply declare the effectors:

```python
"r1": MichaelisMenten(allosteric_activators=["m2c"], subunits=4),
model.parameter_labelling
```

```
...
log_saturation_constant: ('km|r1|m1e', 'km|r1|m1c', 'dc|r1|m2c', ...)
log_tc: ('r1',)
...
```

The effector's dissociation constant is labelled with the `dc` prefix, and the reaction gains a `log_tc` entry. `subunits` is the exponent in the formula above: it does nothing at all unless the reaction is allosteric.

Enzax considers a rate equation allosteric if it declares an allosteric inhibitor, an allosteric activator or a `tc` label. Note that the last option, with a `tc` declaration and no effectors, is a legitimate option: with no effector states to place, the tense state is the unbound enzyme and the relaxed state is the whole catalytic polynomial, so the factor becomes `1 / (1 + tc * (1 / Z) ** subunits)`. Such an enzyme relaxes as its own substrates and products saturate it, and `tc` says how much of it is tense when nothing is bound.

An allosteric constant can be made equal to a catalytic constant by giving it a `km` label:

```python
"r1": MichaelisMenten(allosteric_activators={"m1c": "km|r1|m1c"}),
```

This adds no new position to `log_saturation_constant`: `m1c`'s Michaelis constant now does double duty as its allosteric dissociation constant.

### Sharing values between reactions

Three fields say which parameter value a quantity should use rather than adding anything to the enzyme's states. `kcat` and `enzyme` each default to the reaction's id, and `michaelis_constants` maps species ids to labels, defaulting to `km|{reaction}|{species}`. Reactions that use the same label use the same value: one position in one array, one thing to infer.

As an example, consider the two transketolase reactions in `enzax.examples.glycolysis`. One enzyme catalyses both reactions, with one set of Michaelis constants and a turnover number each:

```python
"TKT1": MichaelisMenten(
    enzyme="TKT",
    michaelis_constants={
        "r5p_c": "km|TKT|r5p_c",
        "xu5p_c": "km|TKT|xu5p_c",
        "s7p_c": "km|TKT|s7p_c",
        "g3p_c": "km|TKT|g3p_c",
    },
),
"TKT2": MichaelisMenten(
    enzyme="TKT",
    michaelis_constants={
        "e4p_c": "km|TKT|e4p_c",
        "xu5p_c": "km|TKT|xu5p_c",
        "f6p_c": "km|TKT|f6p_c",
        "g3p_c": "km|TKT|g3p_c",
    },
),
```

`xu5p_c` and `g3p_c` take part in both reactions, so their constants are declared once and gathered twice. Gradients with respect to a shared value accumulate contributions from every reaction that uses it.

`michaelis_constants` is partial, so mention only the species whose label you want to change. Its keys have to be species that the reaction actually has a Michaelis constant for. In particular, note that mentioning the substrate of an irreversible reaction will cause an error:

```python
"r2": MichaelisMenten(reversible=False, michaelis_constants={"m2c": "km|r2|m2c"}),
```

```
ValueError: Reaction r2's michaelis_constants declaration names ['m2c'], which are not among its substrates ['m1c'].
```

Note also that `competitive_inhibitors`, `allosteric_inhibitors` and `allosteric_activators` accept either a list of species ids, which get default labels, or a `{species: label}` mapping.

### Water and formation energies

A reversible reaction's driving force comes from the formation energies of its reactants, which the model works out from its compounds. Two fields handle water, which is not considered a species:

- `water_stoichiometry`: how much water the reaction consumes or produces. It defaults to zero and only matters to a reversible reaction.
- `water_dgf`: water's formation energy. The default is [equilibrator's](http://equilibrator.weizmann.ac.il/metabolite?compoundId=C00001) value. It is a property of the model rather than of the reaction, so give every reaction that touches water the same value.

You can also modify a reaction's thermodynamics using the field `dgf_species`, a `{species: compound}` mapping that overrides which formation energy a reactant contributes. This is an escape hatch for reproducing a published model that says something the model's compounds do not; ideally your model's reaction thermodynamics should agree with it's reactants' formation energies!

### Checking what you declared

Since every field either creates a label or points at one, the quickest check on a declaration is the model's own labelling:

```python
model.parameter_labelling
```

By inspecting the parameter labelling you can easily see if a constant you expected to be shared appears twice, or one you expected to exist is missing. Enzax also raises errors at model construction time for declarations that cannot be satisfied. Examples include a `michaelis_constants` key that is not a reactant, a species declared as both an allosteric inhibitor and an allosteric activator, or two species in one declaration given the same label.

By default the binding polynomial follows from the stoichiometry: substrates bind in one random-order complex, products in another, and each competitive inhibitor forms a dead end. If that is the enzyme you have, you can save work by using the default `MichaelisMenten` rate law. If not, read on!

## Write your own binding polynomial

`MichaelisMenten` works out the enzyme's binding polynomial, i.e. `Z` in its rate equation, by constructing states based on the reaction's stoichiometry. This assumes that the substrates bind in one random-order complex, the products in another, and that nothing else happens. Real enzymes often do something different: for example, a substrate and a product may be bound at the same time in a complex that cannot react. To represent cases like this you can use enzax's `SaturableRateEquation` class.

`SaturableRateEquation` is `MichaelisMenten` with added functionality for customising the enzyme states that determine its binding polynomial. This means that everything from the previous section still applies: the same fields, labels, thermodynamics and turnover number.

The binding polynomial `Z` is a sum over the states an enzyme can be in, with `1 / Z` being the fraction of enzyme that is unbound. See the [`enzax.binding` API page](api/binding.md) for the underlying theory.

Enzax provides two primitive functions for building binding polynomials, each representing the state at one part of an enzyme: `site` and `dead_end`. They are called by referring to the ids of the species involved: for `site` these are species that compete to bind; for `dead_end` they are species that bind together. Independent states, i.e. those that can both exist or not at the same time, can be combined with the `*` operator. For mutually exclusive states, use the `+` operator.

Here are some examples:

```python
from enzax.binding import ONE, dead_end, site

site("a", "b")                 # one site; a and b compete
site("a") * site("b")          # two sites; a and b bind independently
site("a", exponent=2.0)        # two equivalent copies of one site
dead_end("a") + dead_end("b")  # a/k_a + b/k_b; two alternative dead-end states
ONE                            # 1; the unbound enzyme on its own
```

It is also possible to multiply a state function by a scalar. This scales the whole polynomial, allowing `SaturableRateEquation` to represent a lumped constant that might appear in a literature rate law.

Alongside the two functions, `enzax.binding` exports the polynomial `ONE`. This is a value rather than a function, representing just the number 1. Three things make it useful:

- As a tense or relaxed state, `ONE` says that this state of the enzyme is simply the unbound one: this is how the G6PDH and HEX2 examples appear below.
- Since expressions have no subtraction operator, `-1.0 * ONE` allows a polynomial to include a -1 term. The polynomial enzax derives for a reversible reaction uses this to avoid counting the unbound enzyme twice.
- Multiplying by `ONE` changes nothing, so it is a convenient starting point when building up a polynomial in a loop.

### Several copies of one site

An enzyme with two equivalent copies of a site could be declared as `site("a") * site("a")`, but `site` takes a keyword argument `exponent` for saying the same thing more briefly. Note that this represents copies of a site that fill up independently of each other. An enzyme whose sites influence one another is a job for the Monod-Wyman-Changeux factor and its `subunits` field, not for an exponent.

`MichaelisMenten` uses this itself. When it derives a complex from the stoichiometry, each species' site is raised to the absolute value of that species' stoichiometric coefficient, so a reaction that consumes two molecules of `a` gets `(1 + a/k_a) ** 2` rather than `1 + a/k_a`.

### Which constant a species divides by

Each species named in a polynomial divides by a constant from `log_saturation_constant`. Passing a bare species id makes enzax use a default label, whereas passing a `{species: label}` mapping lets you choose the constant:

```python
site("nadp_c")                            # a new constant
site({"nadp_c": "km|G6PDH|nadp_c"})       # G6PDH's Michaelis constant for nadp_c
```

The default label's prefix depends on which polynomial the species ends up in: `km` in the catalytic polynomial, `dc` in a tense or relaxed state. This is why the mapping form matters most in an allosteric state. Writing `site("nadp_c")` in G6PDH's relaxed state would create a new parameter `dc|G6PDH|nadp_c`, giving NADP one dissociation constant for binding the relaxed enzyme and a separate Michaelis constant for being a substrate. The mapping form says instead that these are the same constant, which is what the model intends.

### The four polynomials

`SaturableRateEquation` takes everything `MichaelisMenten` takes, plus four optional expressions. Each defaults to `None`, meaning "derive this one", so you only write out the part that differs.

| Field | What it replaces |
| --- | --- |
| `dead_end_states_expression` | *adds to* the derived polynomial |
| `binding_polynomial_expression` | the whole derived polynomial |
| `tense_state_expression` | the tense state, otherwise one state per allosteric inhibitor |
| `relaxed_state_expression` | the relaxed state, otherwise the catalytic polynomial times one state per allosteric activator |

Note that passing a tense or relaxed state expression implicitly makes the rate equation allosteric. In this case enzax will create default non-user-provided allosteric terms and parameters like the transfer constant parameter `tc`.

### Worked examples

These are all taken from `enzax.examples.glycolysis`. In the formulas below, `km_glc_c` is short for the value labelled `km|HEX1|glc_c` and so on.

**A dead-end state.** Hexokinase can hold glucose and glucose-6-phosphate at the same time, in a complex that goes nowhere. That state is not implied by the stoichiometry, since glucose is a substrate and glucose-6-phosphate a product, so the derived polynomial has them in separate complexes:

```python
"HEX1": SaturableRateEquation(
    dead_end_states_expression=dead_end("g6p_c", "glc_c"),
),
```

The dead end is added to the polynomial the stoichiometry implies, giving a state where both are bound:

```
Z = -1 + (1 + glc_c/km_glc_c)(1 + atp_c/km_atp_c)
       + (1 + g6p_c/km_g6p_c)(1 + adp_c/km_adp_c)
       + (g6p_c/km_g6p_c)(glc_c/km_glc_c)
```

Aldolase's abortive complex works the same way with three species bound at once, which is a single `dead_end` with three arguments:

```python
"FBA": SaturableRateEquation(
    dead_end_states_expression=dead_end("fdp_c", "g3p_c", "dhap_c"),
),
```

**Allosteric states that are not effector states.** Glucose-6-phosphate dehydrogenase is inhibited unless NADP is bound, which is not "an activator stabilises the relaxed state" but something more specific: the relaxed state *is* the NADP site, and the tense state is the unbound enzyme.

```python
"G6PDH": SaturableRateEquation(
    subunits=2,
    tense_state_expression=ONE,
    relaxed_state_expression=site({"nadp_c": "km|G6PDH|nadp_c"}),
),
```

The allosteric factor is then

```
1 / (1 + tc * (1 / (1 + nadp_c/km_nadp_c)) ** 2)
```

so more NADP means a larger relaxed state, a smaller ratio and less inhibition. Note that the catalytic polynomial does not appear here, unlike in the derived relaxed state: this enzyme's two states depend on NADP alone.

**Two states written out in full.** Phosphofructokinase is the most involved case. Both states are written by hand, both reuse the reaction's own Michaelis constants, and the tense state carries a constant factor of 18:

```python
"PFKM": SaturableRateEquation(
    subunits=4,
    tense_state_expression=(
        18.0 * site({"atp_c": "km|PFKM|atp_c"}) * site("lac_c")
    ),
    relaxed_state_expression=(
        site({"f6p_c": "km|PFKM|f6p_c", "fdp_c": "km|PFKM|fdp_c"})
        * site("f26bp_c")
    ),
),
```

```
tense   = 18 (1 + atp_c/km_atp_c)(1 + lac_c/dc_lac_c)
relaxed = (1 + f6p_c/km_f6p_c + fdp_c/km_fdp_c)(1 + f26bp_c/dc_f26bp_c)
```

ATP and lactate stabilise the tense state; fructose-2,6-bisphosphate stabilises the relaxed one. Fructose-6-phosphate and fructose-1,6-bisphosphate are the reaction's own substrate and product, and they appear in one `site` call because they compete for the same place on the relaxed enzyme. Lactate and fructose-2,6-bisphosphate are named as bare ids, so they get new `dc` constants; everything else reuses a Michaelis constant by name.

**A constant factor.** Sometimes the allosteric machinery is being used to express a number rather than a mechanism. HEX2's two states are both the unbound enzyme, which makes the factor `1 / (1 + tc)`. The overall effect is that `tc` no longer represents an allosteric transfer constant but rather behaves as a (from enzax's point of view) arbitrary constant:

```python
"HEX2": SaturableRateEquation(
    dead_end_states_expression=(
        dead_end("g6p_c", "glc_c") + dead_end("gdp_c", "glc_c")
    ),
    tense_state_expression=ONE,
    relaxed_state_expression=ONE,
),
```

This also shows two dead ends added together, and `gdp_c`, a species that reaches the model only by being named here.

### Rules to respect

It is possible to write incorrect models using `SaturableRateEquation`! To avoid this, make sure to follow these guidelines.

**Only dead ends belong in `dead_end_states_expression`.** What you write there is added to the derived polynomial, which already counts the unbound enzyme once. A `site` contributes a `1` of its own, so putting one here would count the unbound state twice and understate every saturation.

**A polynomial you write in full has to count the unbound state exactly once.** `binding_polynomial_expression` replaces the derived polynomial, so nothing corrects it for you. This is why the term `-1.0 * ONE` appears in the default reversible polynomial: the substrate complex and the product complex each contain the unbound enzyme, so one copy has to come back out. If you write two complexes as a sum without this correction, `1 / Z` incorrectly stops representing the unbound enzyme fraction.

**Tense and relaxed states are exempt from that rule.** Only their ratio enters the allosteric factor, so an overall constant in either one just rescales `tc`, as PFKM's 18 and HEX2's `1 / (1 + tc)` both rely on. It does change what a fitted `tc` means, so it is worth a comment in your own models.

**A species you name does not have to be a reactant, and does not have to exist yet.** Naming one is how an effector joins the model, and there is no check that you meant to: a misspelled species id gives you a new unbalanced species with its own concentration and formation energy to declare, rather than an error. If a model has grown a species you do not recognise, carefully check `model.species` and `model.parameter_labelling`.

## Write a rate equation from scratch

The rate equations above all describe a saturating enzyme, and a binding polynomial can say a great deal about one. But some fluxes are not of that shape at all: elementary mass action kinetics, a transporter obeying a rate law from a particular paper, or an empirical function fitted to data. For these, write your own `RateEquation` subclass.

Be warned that this is more work than writing one function. A rate equation refers to its parameters by label, and labels become positions in the model's flat parameter arrays once, when the model is built, so that evaluating a flux is only array indexing and arithmetic. Getting a parameter value into a flux therefore takes three stages rather than one, and a subclass has to implement each of them.

### The methods to implement

| Method | When it runs | What it returns |
| --- | --- | --- |
| `get_species` | model construction | extra species ids the reaction names but the stoichiometry does not (optional; defaults to `()`) |
| `get_labels` | model construction | a `RateEquationLabels` subclass listing every label the equation refers to |
| `resolve` | model construction | a PyTree of positions: where in each parameter's array this reaction's values live |
| `get_input` | once per flux evaluation | the parameter values, gathered from those positions |
| `__call__` | once per flux evaluation | the flux, as a scalar |

The `RateEquationLabels` subclass has one field per group of labels the rate equation declares, and its `by_parameter` method says which flat array each group is gathered from. That method is the only record of the correspondence, so all three of `by_parameter`, `resolve` and `get_input` have to name the same arrays.

Two of these mistakes fail differently, which is worth knowing before you make one. If `resolve` looks for a position in an array that `by_parameter` did not name, the label is not there and the model raises as it is built:

```
KeyError: "'log_enzyme' has no value labelled 'r2'."
```

But if `resolve` is right and `get_input` reads a different array at the position it found, nothing raises at all. JAX clamps an out-of-range index rather than complaining, so the rate equation quietly gathers some other reaction's value and the flux is wrong.

Note that `resolve` returns a bundle of its own, rather than positions the model assembles for it. This is partly because each reaction's arrays are ragged -- one reaction has three substrates, the next has one -- so each needs its own [jaxtyping](https://docs.kidger.site/jaxtyping/) scope for shape annotations like `n_rxn_substrate` to be meaningful. It is also a convenient place to put anything else about the reaction that never changes, such as its stoichiometric coefficients.

### What a rate equation gets told

`resolve` and `get_labels` are handed a `ReactionScope`, which is everything the rate equation is allowed to know about the reaction it belongs to:

- `reaction_id`, the reaction's own id, which is the default label for anything the reaction has one of.
- `species`, the model's species ids, in the model's order.
- `stoichiometry`, this reaction's coefficient for every one of those species, mostly zeroes.
- `species_to_dgf_ix`, where each species' formation energy lives.

Rather than reading these directly, use the helpers in `enzax.rate_equation`: `get_substrates`, `get_products` and `get_reactants` return species ids in the model's order, and `get_species_positions` turns species ids into positions, raising for a species the model does not have.

Positions matter because `__call__` is handed the concentrations of *all* the model's species, in the model's order, not just the ones this reaction uses. A rate equation picks out what it needs by indexing with the positions that `resolve` worked out.

### A complete example

Irreversible mass action kinetics is a good illustration, being about as far from a binding polynomial as a rate equation gets: the flux is a rate constant times each substrate's concentration raised to the number of molecules the reaction consumes.

```python
from dataclasses import dataclass

import equinox as eqx
import numpy as np
from jax import numpy as jnp
from jaxtyping import Scalar

from enzax.array_types import (
    ConcArray,
    ParamDict,
    ParamLabelling,
    StaticSubstrateArr,
    SubstrateIx,
)
from enzax.parameters import get_parameter_position
from enzax.rate_equation import (
    RateEquation,
    RateEquationLabels,
    ReactionScope,
    get_reaction_label,
    get_species_positions,
    get_substrates,
)


@dataclass(frozen=True)
class MassActionLabels(RateEquationLabels):
    """The labels a mass action reaction refers to."""

    rate_constant: str

    def by_parameter(self) -> ParamLabelling:
        return {"log_kcat": (self.rate_constant,)}


class MassActionIx(eqx.Module):
    """Where a mass action reaction reads its inputs."""

    ix_rate_constant: int
    ix_substrate: SubstrateIx
    order: StaticSubstrateArr


class MassActionInput(eqx.Module):
    rate_constant: Scalar
    ix_substrate: SubstrateIx
    order: StaticSubstrateArr


class MassAction(RateEquation):
    """Irreversible mass action kinetics.

    Fields:

    * `rate_constant`: label of the reaction's rate constant. Defaults to the
      reaction id.
    """

    rate_constant: str | None = None

    def get_labels(self, scope: ReactionScope) -> MassActionLabels:
        return MassActionLabels(
            rate_constant=get_reaction_label(
                self.rate_constant, scope.reaction_id
            )
        )

    def resolve(
        self,
        scope: ReactionScope,
        labelling: ParamLabelling,
    ) -> MassActionIx:
        lab = self.get_labels(scope)
        ix_substrate = get_species_positions(scope, get_substrates(scope))
        return MassActionIx(
            ix_rate_constant=get_parameter_position(
                labelling, "log_kcat", lab.rate_constant
            ),
            ix_substrate=ix_substrate,
            order=np.abs(scope.stoichiometry[ix_substrate]),
        )

    def get_input(
        self, parameters: ParamDict, ix: MassActionIx
    ) -> MassActionInput:
        return MassActionInput(
            rate_constant=jnp.exp(parameters["log_kcat"][ix.ix_rate_constant]),
            ix_substrate=ix.ix_substrate,
            order=ix.order,
        )

    def __call__(
        self, conc: ConcArray, rate_equation_input: MassActionInput
    ) -> Scalar:
        conc_substrate = conc[rate_equation_input.ix_substrate]
        return rate_equation_input.rate_constant * jnp.prod(
            conc_substrate**rate_equation_input.order
        )
```

A few things to notice. The rate constant is stored in `log_kcat`, since a turnover number is the closest thing enzax has to a rate constant; the [next subsection](#what-a-rate-equation-may-not-do) explains why it cannot have an array of its own. Its label defaults to the reaction id, exactly as `MichaelisMenten`'s does, so two mass action reactions share a rate constant by declaring the same label. The reaction orders come out of the stoichiometry in `resolve`, where they are computed once, and they travel as a numpy array because they never change and JAX should not trace them.

Using it in a model is no different from using a built-in rate equation. Here reaction `r2` consumes two molecules of `m1c`:

```python
model = RateEquationModel(
    stoichiometry={
        "r1": {"m1e": -1.0, "m1c": 1.0},
        "r2": {"m1c": -2.0, "m2c": 1.0},
        "r3": {"m2c": -1.0, "m2e": 1.0},
    },
    balanced_species=["m1c", "m2c"],
    rate_equations={
        "r1": MichaelisMenten(),
        "r2": MassAction(),
        "r3": MichaelisMenten(),
    },
)
model.parameter_labelling
```

```
log_kcat: ('r1', 'r2', 'r3')
log_enzyme: ('r1', 'r3')
log_saturation_constant: ('km|r1|m1e', 'km|r1|m1c', 'km|r3|m2c', 'km|r3|m2e')
dgf: ('m1e', 'm1c', 'm2c', 'm2e')
log_conc_unbalanced: ('m1e', 'm2e')
temperature: ()
```

The labelling is a good check that a new rate equation declares what you meant. `r2` has a `log_kcat` entry and nothing else: no Michaelis constants, and no `log_enzyme` entry, because `MassActionLabels` never mentions one.

Given parameters, the flux is what the formula says. With `log_kcat` for `r2` set to `log(2.5)` and `m1c` at 0.4, its middle entry is `2.5 * 0.4 ** 2`, the two Michaelis-Menten reactions either side depending on the rest of the parameters:

```python
model.flux(jnp.array([0.4, 0.2]), parameters)
```

```
Array([0.05263158, 0.4       , 0.07692308], dtype=float64)
```

`Drain` in `enzax.rate_equations.drain` is the shortest complete rate equation in enzax, and worth reading as a second example: it declares one label, resolves one position and ignores the concentrations entirely.

### What a rate equation may not do

Enzax's parameters are a closed set, listed as `PARAMETERS` in `enzax.parameters`: `log_saturation_constant`, `log_kcat`, `log_enzyme`, `log_tc` and `log_drain` come from rate equations, and `dgf`, `log_conc_unbalanced`, `conserved_pools` and `temperature` come from the model's structure. A rate equation whose `by_parameter` names anything else raises when the model is constructed:

```
ValueError: Unknown parameters: ['log_my_thing'].
```

Within `log_saturation_constant`, every label must start with `km|`, `ki|` or `dc|`.

So a rate equation with a genuinely new kind of parameter cannot be written in your own script alone. The good news is that adding one is a small change: putting its name in `KINETIC_PARAMETERS` is enough, since packing, unpacking and priors all work from the model's labelling rather than from a hardcoded list of parameters. The bad news is that it is a change to enzax itself rather than to your model, so the [contributing guide](contributing.md) is the place to start.

Until then, the way to express a new quantity is to reuse whichever existing parameter it most resembles, as `MassAction` does with `log_kcat`. Bear in mind that the reused name will follow the parameter around: it is what appears in `model.parameter_labelling`, in a parameter set and in any Jacobian with respect to the parameters.

## Checklist

Whichever of the three routes you took, it is worth checking that the rate equation you declared is the one you meant:

- Is the rate equation registered in the model's `rate_equations` dictionary, under the right reaction id?
- Does `model.parameter_labelling` contain the labels you expected, and no others? A label you did not expect usually means a species id typo or a sharing declaration that did not take effect; a missing one means a field that is not doing anything.
- Does `model.species` contain only species you meant to model? Naming a species anywhere is enough to add it to the model.
- Does `model.flux` at a known concentration vector agree with the formula worked out by hand? For a binding polynomial, `1 / Z` should be a fraction: if it is bigger than 1 or negative, the polynomial is not counting the unbound enzyme once.
