import sympy
import sympy2jax
from jaxtyping import PyTree, Scalar
from enzax.rate_equation import ConcArray, RateEquation


class SymbolicRateEquation(RateEquation):
    fn_jax: sympy2jax.SymbolicModule

    def __init__(self, expr: sympy.Expr):
        self.fn_jax = sympy2jax.SymbolicModule(expr)

    def __call__(self, conc: ConcArray, parameters: PyTree) -> Scalar:
        return self.fn_jax(conc=conc, parameters=parameters)
