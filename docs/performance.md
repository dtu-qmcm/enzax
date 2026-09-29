# Performance

Enzax includes a range of performance optimisations, primarily aimed at fast steady state solving during Hamiltonian Monte Carlo sampling. The overall improvement from these optimisations on our test model is 64x! This page explains why we need these optimisations, what they are, and how enzax compares with other similar software.

## Problem: HMC with embedded steady state solving

While, thanks to its modular design, enzax is useful for many other problems, its primary motivation is fitting Bayesian statistical models of multi-omics measurements of steady state metabolic networks, as described in [this paper](https://doi.org/10.1021/acssynbio.3c00662).

The best algorithm for fitting such a model is a gradient-based Markov chain Monte Carlo sampler like Hamiltonian Monte Carlo. HMC and its variants achieve good performance by numerically simulating Hamiltonian trajectories to choose new points in parameter space. These simulations require repeatedly evaluating the target probability density and its parameter gradients: in a typical run with four chains, 2000 iterations and 125 evaluations per iteration, the target density must be evaluated and differentiated one million times!

For steady state metabolic network models, by far the most costly part of a gradient evaluation is solving and differentiating a steady state problem, i.e. finding a set of ODE state species concentrations that do not change under the model's flux. Most of enzax's optimisations therefore target this problem.

## The four optimisations

**A hybrid solver** There are two main approaches to numerically solving a steady state problem: directly applying a root-finding algorithm like Newton's method or solving the initial value problem using an ODE solver, starting at a guess and ending when a steady state event occurs. Root-finding is typically faster, but tends to fail without a very good guess, whereas the initial value problem method is slower but more reliable. Enzax provides a steady state solver called `get_steady_state_hybrid` that combines these two methods. First it attempts to solve the steady state problem with a Newton solver, then solves the initial value problem from the Newton solution in case of success or from the original guess in case of failure. When the Newton solve succeeds, the ODE solve takes no steps because the steady state condition is already met; if it fails, the cost in Newton steps is typically small compared with the required ODE solve.

**A BDF ODE solver** The ODE system defined by a metabolic network's kinetics is usually stiff because the states change at different timescales. Stiff ODE systems require specialised solvers, of which diffrax provides some, including the Kvaerno3/4/5 family. However, the recommended default for ODE models in systems biology is a solver based on the backward differentiation formula ([Städter et al. 2021](https://doi.org/10.1038/s41598-021-82196-2)), which diffrax does not provide. We implemented a BDF solver for diffrax in a separate package called [diffrax-bdf](https://github.com/dtu-qmcm/diffrax-bdf) and found that it improves performance compared with Kvaerno5. You can use it with enzax by importing `diffrax_bdf.BDF` and its companion step size controller `diffrax_bdf.BDFController` and passing them to `enzax.steady_state.get_steady_state`.

**Implicit differentiation** The standard way to differentiate an ODE solve involves differentiating every step of the ODE solve. This is usually expensive, and the cost increases with the number of steps. As a root-finding problem, a steady state problem can be differentiated much more cheaply by exploiting the implicit function theorem; enzax does this by default via [`diffrax.ImplicitAdjoint`](https://docs.kidger.site/diffrax/api/adjoints/#diffrax.ImplicitAdjoint).

**The grapevine method** A Hamiltonian trajectory moves through parameter space in small steps, so the solution of the steady state problem at one step is often close to the solution at the next step. We developed the [grapevine method](https://openreview.net/forum?id=z4PfNDNAcN) to exploit this by allowing adjacent trajectory steps to share root-finding information. To use the grapevine method with enzax, use the `grapenuts` sampler from our package [grapevine](https://github.com/dtu-qmcm/grapevine) and use something like the log density function `enzax_log_density_grapevine`: see `scripts/mcmc_demo.py` for an example.

The grapevine method complements the hybrid solver: better guesses make it more likely that the Newton solve will succeed, so that the steady state problem can be solved with no ODE integration required.

## What the four are worth

To quantify the benefit from enzax's performance optimisations, we ran an optimised MCMC sampler targeting our glycolysis model, then measured how long steps along a new trajectory tended to take after successively removing optimisations. First we removed the hybrid solver, then the grapevine method, then we replaced our BDF solver with diffrax's Kvaerno5, then we replaced implicit differentiation with backpropagation through the ODE solve.

![What each of enzax's optimisations is worth](img/optimisation_benchmark.png)

With all four optimisations, one NUTS iteration of 63 leapfrog steps took 754 ms, compared with 48 s with no optimisations: the optimised sampler was 64 times faster. Extrapolating these results to the one-million-gradient run described above, an unoptimised run would take over a week, compared with just over three hours for the optimised run.

!!! note

    The extrapolation depends on details of the model and sampling. In particular, we have observed some variation in how big an advantage the BDF solver achieves compared with Kvaerno5, depending on the target model. In addition, the benefit from the grapevine method may be different in the post-warmup phase where we measured compared with the warmup phase: this means that our experiment may be overly optimistic (or pessimistic) about the benefit from both grapevine and the hybrid solver.

To regenerate the figure, run this command from the root of the enzax repository, using Python 3.13 or later:

```sh
uv run --group mcmc python scripts/optimisation_benchmark.py
```

This writes the figure to `docs/img/optimisation_benchmark.png` and, alongside it, a csv file with one row per timed leapfrog step.

## Comparison with other frameworks

Several other software packages also provide gradient-based MCMC for kinetic models, and have implemented some of the same performance optimisations as enzax. The table below compares these:

| | [enzax](https://github.com/dtu-qmcm/enzax) | [Maud](https://github.com/biosustain/Maud) | [pyPESTO](https://github.com/ICB-DCM/pyPESTO) + [AMICI](https://github.com/AMICI-dev/AMICI) | [PEtab.jl](https://github.com/sebapersson/PEtab.jl) | [Turing.jl](https://turinglang.org) + SciML | [PyMC](https://www.pymc.io) + [sunode](https://github.com/pymc-devs/sunode) | [Stan](https://mc-stan.org) |
|---|---|---|---|---|---|---|---|
| Hybrid Newton-then-integrate forward solve | yes | no | yes | no | no | n/a | no |
| Stiff BDF with cross-step Jacobian reuse | yes | yes | yes | yes | yes | yes | yes |
| Implicit differentiation of the steady state | yes | no | yes | no | yes | n/a | yes |
| Guess reuse along the Hamiltonian trajectory | yes | no | no | no | no | no | no |
| Gradient-based sampler | GrapeNUTS (blackjax) | NUTS (Stan) | NUTS (via PyMC) | NUTS (AdvancedHMC) | NUTS | NUTS | NUTS |

The n/a cells are because sunode has no steady state solver.

The table shows that only the grapevine method is unique to enzax: in particular, pyPESTO with AMICI provides the other three.

Other software also implements performance optimisations that are missing from enzax. In particular, it is possible to exploit the sparsity of a model's Jacobian and to better differentiate initial value problems.

**Sparsity** A metabolic network usually has a sparse Jacobian because each reaction changes only a few of the network's species. Some software exploits this: in particular AMICI can use the sparse solver KLU and PEtab.jl can build sparse Jacobians. Enzax does not currently exploit sparsity, mainly because the models it fits are too small. In our tests on CPU, sparse methods only started to pay off at around 128 species, which is already outside the envelope of models that can easily be fit using enzax.

**IVP sensitivities** When a metabolic network is measured over time at non-steady states, the appropriate statistical model must embed an initial value problem rather than a steady state problem. AMICI, PEtab.jl and Stan provide performance optimisations that target this case, including adjoint sensitivity analysis and related methods. Diffrax provides similar methods through its `adjoint` argument, but enzax does not currently use these as it focuses on the steady state case.
