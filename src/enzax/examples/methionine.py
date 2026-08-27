"""A kinetic model of the methionine cycle.

See here for more about the methionine cycle:
https://doi.org/10.1021/acssynbio.3c00662

"""

from jax import numpy as jnp

from enzax.kinetic_model import RateEquationModel
from enzax.parameters import pack_parameters
from enzax.rate_equations import (
    AllostericIrreversibleMichaelisMenten,
    Drain,
    IrreversibleMichaelisMenten,
    ReversibleMichaelisMenten,
)

stoichiometry = {
    "the_drain": {"met-L": 1.0},
    "MAT1": {"met-L": -1.0, "atp": -1.0, "pi": 1.0, "ppi": 1.0, "amet": 1.0},
    "MAT3": {"met-L": -1.0, "atp": -1.0, "pi": 1.0, "ppi": 1.0, "amet": 1.0},
    "METH-Gen": {"amet": -1.0, "ahcys": 1.0},
    "GNMT1": {"amet": -1.0, "ahcys": 1.0, "gly": -1.0, "sarcs": 1.0},
    "AHC1": {"ahcys": -1.0, "hcys-L": 1.0, "adn": 1.0},
    "MS1": {"hcys-L": -1.0, "thf": 1.0, "met-L": 1.0, "5mthf": -1.0},
    "BHMT1": {"hcys-L": -1.0, "glyb": -1.0, "met-L": 1.0, "dmgly": 1.0},
    "CBS1": {"hcys-L": -1.0, "ser-L": -1.0, "cyst-L": 1.0},
    "MTHFR1": {"5mthf": 1.0, "mlthf": -1.0, "nadp": 1.0, "nadph": -1.0},
    "PROT1": {"met-L": -1.0},
}
species = [
    "met-L",
    "atp",
    "pi",
    "ppi",
    "amet",
    "ahcys",
    "gly",
    "sarcs",
    "hcys-L",
    "adn",
    "thf",
    "5mthf",
    "mlthf",
    "glyb",
    "dmgly",
    "ser-L",
    "nadp",
    "nadph",
    "cyst-L",
]
balanced_species = [
    "met-L",
    "amet",
    "ahcys",
    "hcys-L",
    "5mthf",
]
reactions = [
    "the_drain",
    "MAT1",
    "MAT3",
    "METH-Gen",
    "GNMT1",
    "AHC1",
    "MS1",
    "BHMT1",
    "CBS1",
    "MTHFR1",
    "PROT1",
]
model = RateEquationModel(
    stoichiometry=stoichiometry,
    species=species,
    reactions=reactions,
    balanced_species=balanced_species,
    rate_equations=[
        Drain(sign=1.0),  # met-L source
        IrreversibleMichaelisMenten(  # MAT1
            ki=["amet"],
        ),
        AllostericIrreversibleMichaelisMenten(  # MAT3
            subunits=2,
            dc_activator=["met-L", "amet"],
        ),
        IrreversibleMichaelisMenten(  # METH
            ki=["ahcys"],
        ),
        AllostericIrreversibleMichaelisMenten(  # GNMT1
            subunits=4,
            ki=["ahcys"],
            dc_inhibitor=["mlthf"],
            dc_activator=["amet"],
        ),
        ReversibleMichaelisMenten(  # AHC
            water_stoichiometry=-1.0,
        ),
        IrreversibleMichaelisMenten(),  # MS
        IrreversibleMichaelisMenten(),  # BHMT
        AllostericIrreversibleMichaelisMenten(  # CBS1
            subunits=2,
            dc_inhibitor=["amet"],
        ),
        AllostericIrreversibleMichaelisMenten(  # MTHFR
            subunits=2,
            dc_inhibitor=["amet"],
            dc_activator=["ahcys"],
        ),
        IrreversibleMichaelisMenten(),  # PROT
    ],
)
parameters = pack_parameters(
    model.parameter_labelling,
    {
        # Every dissociation constant, whatever its role. The prefix says
        # which: km for a Michaelis constant, ki for a competitive inhibition
        # constant, dc for an allosteric one.
        "log_k": {
            "km|MAT1|met-L": jnp.log(0.000106919),
            "km|MAT1|atp": jnp.log(0.00203015),
            "ki|MAT1|amet": jnp.log(0.000346704),
            "km|MAT3|met-L": jnp.log(0.00113258),
            "km|MAT3|atp": jnp.log(0.00236759),
            "dc|MAT3|met-L": jnp.log(0.00059999),
            "dc|MAT3|amet": jnp.log(0.000316641),
            "km|METH-Gen|amet": jnp.log(9.37e-06),
            "ki|METH-Gen|ahcys": jnp.log(5.56e-06),
            "km|GNMT1|amet": jnp.log(0.000520015),
            "km|GNMT1|gly": jnp.log(0.00253545),
            "ki|GNMT1|ahcys": jnp.log(5.31e-05),
            "dc|GNMT1|mlthf": jnp.log(0.000228576),
            "dc|GNMT1|amet": jnp.log(1.98e-05),
            "km|AHC1|ahcys": jnp.log(2.32e-05),
            "km|AHC1|hcys-L": jnp.log(1.06e-05),
            "km|AHC1|adn": jnp.log(5.66e-06),
            "km|MS1|hcys-L": jnp.log(1.71e-06),
            "km|MS1|5mthf": jnp.log(6.94e-05),
            "km|BHMT1|hcys-L": jnp.log(1.98e-05),
            "km|BHMT1|glyb": jnp.log(0.00845898),
            "km|CBS1|hcys-L": jnp.log(4.24e-05),
            "km|CBS1|ser-L": jnp.log(2.83e-06),
            "dc|CBS1|amet": jnp.log(9.30e-05),
            "km|MTHFR1|mlthf": jnp.log(8.08e-05),
            "km|MTHFR1|nadph": jnp.log(2.09e-05),
            "dc|MTHFR1|amet": jnp.log(1.46e-05),
            "dc|MTHFR1|ahcys": jnp.log(2.45e-06),
            "km|PROT1|met-L": jnp.log(4.39e-05),
        },
        "log_kcat": {
            "MAT1": jnp.log(7.89577),
            "MAT3": jnp.log(19.9215),
            "METH-Gen": jnp.log(1.15777),
            "GNMT1": jnp.log(10.5307),
            "AHC1": jnp.log(234.284),
            "MS1": jnp.log(1.77471),
            "BHMT1": jnp.log(13.7676),
            "CBS1": jnp.log(7.02307),
            "MTHFR1": jnp.log(3.1654),
            "PROT1": jnp.log(0.264744),
        },
        "log_enzyme": {
            "MAT1": jnp.log(0.000961712),
            "MAT3": jnp.log(0.00098812),
            "METH-Gen": jnp.log(0.00103396),
            "GNMT1": jnp.log(0.000983692),
            "AHC1": jnp.log(0.000977878),
            "MS1": jnp.log(0.00105094),
            "BHMT1": jnp.log(0.000996603),
            "CBS1": jnp.log(0.00134056),
            "MTHFR1": jnp.log(0.0010054),
            "PROT1": jnp.log(0.000995525),
        },
        "log_tc": {
            "MAT3": jnp.log(0.107657),
            "GNMT1": jnp.log(131.207),
            "CBS1": jnp.log(1.03452),
            "MTHFR1": jnp.log(0.392035),
        },
        "log_drain": {"the_drain": jnp.log(0.000850127)},
        "dgf": {
            "met-L": 160.953,
            "atp": -2263.31,
            "pi": -1055.95,
            "ppi": -1943.8,
            "amet": 636.255,
            "ahcys": 547.319,
            "gly": -161.373,
            "sarcs": -39.4573,
            "hcys-L": 44.2,
            "adn": 375.758,
            "thf": 108.366,
            "5mthf": 223.646,
            "mlthf": 198.009,
            "glyb": 173.094,
            "dmgly": 49.4547,
            "ser-L": -216.712,
            "nadp": -2014.52,
            "nadph": -1948.58,
            "cyst-L": -46.4737,
        },
        "log_conc_unbalanced": {  # dataset1
            "atp": jnp.log(0.00131546),
            "pi": jnp.log(0.001),
            "ppi": jnp.log(0.000500016),
            "gly": jnp.log(0.00145177),
            "sarcs": jnp.log(1.00e-07),
            "adn": jnp.log(1.01e-06),
            "thf": jnp.log(2.24e-05),
            "mlthf": jnp.log(3.15e-06),
            "glyb": jnp.log(0.00106758),
            "dmgly": jnp.log(5.00e-05),
            "ser-L": jnp.log(0.0015873),
            "nadp": jnp.log(1.22e-06),
            "nadph": jnp.log(0.000245139),
            "cyst-L": jnp.log(2.24e-06),
        },
        "temperature": 298.15,
    },
)
steady_state = jnp.array(
    [
        4.233000e-05,  # met-L
        3.099670e-05,  # amet
        2.170170e-07,  # ahcys
        3.521780e-06,  # hcys
        6.534400e-06,  # 5mthf
    ]
)
