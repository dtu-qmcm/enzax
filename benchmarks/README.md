# Benchmarks against AMICI

These scripts compare how fast and how accurately enzax and [AMICI](https://github.com/AMICI-dev/AMICI) compute the gradient of a log likelihood with respect to a model's parameters. AMICI is run through both of its backends: the C++ one, which uses CVODES, and the JAX one, which uses diffrax.

## Problems

- **Two problems from the [PEtab benchmark collection](https://github.com/Benchmarking-Initiative/Benchmark-Models-PEtab)** (`petab_collection.py`), both used to benchmark AMICI's steady-state adjoint method (https://doi.org/10.1371/journal.pcbi.1010783): a model of histone H4 acetylation with steady-state data (https://doi.org/10.1016/j.cels.2016.01.002) and a model of histone H3 methylation with a time course that starts from a steady state (https://doi.org/10.1073/pnas.1205707109). Their PEtab files are copied into `data/petab_collection/` under the collection's licence.
- **The glycolysis and pentose phosphate pathway example** (`glycolysis.py`), with simulated steady-state measurements. The script writes the model as SBML with `enzax_to_sbml` and the problem as PEtab, into `data/glycolysis/`.

## What is measured

For each problem, each tool and solver is timed on ten parameter sets drawn around the problem's nominal values, at tolerances of 1e-6, 1e-8 and 1e-10. enzax runs Kvaerno5 and diffrax-bdf's BDF, each on its own and after a Newton solve (the hybrid method). AMICI JAX runs the same two solvers. AMICI C++ runs adjoint and forward sensitivities, each by integration or by a Newton solve at the steady state, and also runs adjoint sensitivities with a Newton solve for the steady state itself. One tolerance sets both the integrator's tolerances and the steady-state stopping criterion. Each time is the median of five gradient evaluations after compilation; AMICI C++ is timed by its own CPU time. Each gradient is compared with a reference from AMICI C++ with forward sensitivities, solved by Newton at the steady state, at tight tolerances.

enzax is given the same diffrax solvers and step size controllers as AMICI's JAX backend. AMICI is run with `AMICI_EXPERIMENTAL_SBML_NONCONST_CLS` set, so that it removes conservation laws; without it, its JAX gradients on the histone acetylation model are wrong. AMICI's models are compiled afresh on every run, in `build/`.

## Running

AMICI is built from source when the `benchmark` dependency group is installed, and compiles every model it imports, so a C++20 compiler is needed. AMICI recommends WSL on Windows.

```sh
uv run --group benchmark python benchmarks/petab_collection.py
uv run --group benchmark python benchmarks/glycolysis.py
uv run --group benchmark python benchmarks/plot.py
```

The first two write `results/<problem>.csv`, one row per tool, solver, tolerance and parameter set. The third draws time and gradient error against tolerance for every problem, as `results/work_precision.pdf` and `.png`. A full run takes about an hour.
