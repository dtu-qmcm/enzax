"""A kinetic model of the methionine cycle.

See here for more about the methionine cycle:
https://doi.org/10.1021/acssynbio.3c00662

"""

import numpy as np
from jax import numpy as jnp

from enzax.kinetic_model import RateEquationModel
from enzax.rate_equations import (
    AllostericIrreversibleMichaelisMenten,
    Drain,
    IrreversibleMichaelisMenten,
    ReversibleMichaelisMenten,
)


stoichiometry = {
    "the_drain": {"met-L": 1},
    "MAT1": {"met-L": -1, "atp": -1, "pi": 1, "ppi": 1, "amet": 1},
    "MAT3": {"met-L": -1, "atp": -1, "pi": 1, "ppi": 1, "amet": 1},
    "METH-Gen": {"amet": -1, "ahcys": 1},
    "GNMT1": {"amet": -1, "ahcys": 1, "gly": -1, "sarcs": 1},
    "AHC1": {"ahcys": -1, "hcys-L": 1, "adn": 1},
    "MS1": {"hcys-L": -1, "thf": 1, "met-L": 1, "5mthf": -1},
    "BHMT1": {"hcys-L": -1, "glyb": -1, "met-L": 1, "dmgly": 1},
    "CBS1": {"hcys-L": -1, "ser-L": -1, "cyst-L": 1},
    "MTHFR1": {"5mthf": 1, "mlthf": -1, "nadp": 1, "nadph": -1},
    "PROT1": {"met-L": -1},
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
parameters = dict(
    log_kcat={
        "MAT1": jnp.log(jnp.array(7.89577)),  # MAT1
        "MAT3": jnp.log(jnp.array(19.9215)),  # MAT3
        "METH-Gen": jnp.log(jnp.array(1.15777)),  # METH-Gen
        "GNMT1": jnp.log(jnp.array(10.5307)),  # GNMT1
        "AHC1": jnp.log(jnp.array(234.284)),  # AHC1
        "MS1": jnp.log(jnp.array(1.77471)),  # MS1
        "BHMT1": jnp.log(jnp.array(13.7676)),  # BHMT1
        "CBS1": jnp.log(jnp.array(7.02307)),  # CBS1
        "MTHFR1": jnp.log(jnp.array(3.1654)),  # MTHFR1
        "PROT1": jnp.log(jnp.array(0.264744)),  # PROT1
    },
    log_enzyme={
        "MAT1": jnp.log(jnp.array(0.000961712)),  # MAT1
        "MAT3": jnp.log(jnp.array(0.00098812)),  # MAT3
        "METH-Gen": jnp.log(jnp.array(0.00103396)),  # METH-Gen
        "GNMT1": jnp.log(jnp.array(0.000983692)),  # GNMT1
        "AHC1": jnp.log(jnp.array(0.000977878)),  # AHC1
        "MS1": jnp.log(jnp.array(0.00105094)),  # MS1
        "BHMT1": jnp.log(jnp.array(0.000996603)),  # BHMT1
        "CBS1": jnp.log(jnp.array(0.00134056)),  # CBS1
        "MTHFR1": jnp.log(jnp.array(0.0010054)),  # MTHFR1
        "PROT1": jnp.log(jnp.array(0.000995525)),  # PROT1
    },
    log_drain={"the_drain": jnp.log(jnp.array(0.000850127))},
    dgf=jnp.array(
        [
            160.953,  # met-L
            -2263.31,  # atp
            -1055.95,  # pi
            -1943.8,  # ppi
            636.255,  # amet
            547.319,  # ahcys
            -161.373,  # gly
            -39.4573,  # sarcs
            44.2,  # hcys-L
            375.758,  # adn
            108.366,  # thf
            223.646,  # 5mthf
            198.009,  # mlthf
            173.094,  # glyb
            49.4547,  # dmgly
            -216.712,  # ser-L
            -2014.52,  # nadp
            -1948.58,  # nadph
            -46.4737,  # cyst-L
        ]
    ),
    log_product_km={
        "AHC1": jnp.log(
            jnp.array([1.06e-05, 5.66e-06])
        ),  # hcys-L AHC1, adn AHC1
    },
    log_substrate_km={
        "MAT1": jnp.log(
            jnp.array([0.000106919, 0.00203015])
        ),  # MAT1 met-L, atp
        "MAT3": jnp.log(jnp.array([0.00113258, 0.00236759])),  # MAT3 met-L atp
        "METH-Gen": jnp.log(jnp.array([9.37e-06])),  # METH-Gen amet
        "GNMT1": jnp.log(
            jnp.array([0.000520015, 0.00253545])
        ),  # GNMT1, amet, gly
        "AHC1": jnp.log(jnp.array([2.32e-05])),  # ahcys AHC1
        "MS1": jnp.log(jnp.array([1.71e-06, 6.94e-05])),  # MS1 hcys-L, 5mthf
        "BHMT1": jnp.log(
            jnp.array([1.98e-05, 0.00845898])
        ),  # BHMT1 hcys-L, glyb
        "CBS1": jnp.log(jnp.array([4.24e-05, 2.83e-06])),  #  CBS1 hcys-L, ser-L
        "MTHFR1": jnp.log(
            jnp.array([8.08e-05, 2.09e-05])
        ),  # MTHFR1 mlthf, nadph
        "PROT1": jnp.log(jnp.array([4.39e-05])),  # PROT1 met-L
    },
    temperature=jnp.array(298.15),
    log_ki={
        "MAT1": jnp.array([jnp.log(0.000346704)]),  # MAT1
        "MAT3": jnp.array([]),
        "METH-Gen": jnp.array([jnp.log(5.56e-06)]),  # METH-Gen
        "GNMT1": jnp.array([jnp.log(5.31e-05)]),  # GNMT1
        "AHC1": jnp.array([]),
        "MS1": jnp.array([]),
        "BHMT1": jnp.array([]),
        "CBS1": jnp.array([]),
        "MTHFR1": jnp.array([]),
        "PROT1": jnp.array([]),
    },
    log_conc_unbalanced=jnp.log(
        jnp.array(
            [
                # dataset1
                0.00131546,  # atp
                0.001,  # pi
                0.000500016,  # ppi
                0.00145177,  # gly
                1.00e-07,  # sarcs
                1.01e-06,  # adn
                2.24e-05,  # thf
                3.15e-06,  # mlthf
                0.00106758,  # glyb
                5.00e-05,  # dmgly
                0.0015873,  # ser-L
                1.22e-06,  # nadp
                0.000245139,  # nadph
                2.24e-06,  # cyst-L
            ]
        )
    ),
    log_tc={
        "MAT3": jnp.array(jnp.log(0.107657)),  # MAT3
        "GNMT1": jnp.array(jnp.log(131.207)),  # GNMT
        "CBS1": jnp.array(jnp.log(1.03452)),  # CBS
        "MTHFR1": jnp.array(jnp.log(0.392035)),  # MTHFR
    },
    log_dc_activator={
        "MAT3": jnp.log(
            jnp.array([0.00059999, 0.000316641])
        ),  # met-L MAT3,  # amet MAT3
        "GNMT1": jnp.log(jnp.array([1.98e-05])),  # amet GNMT1
        "CBS1": jnp.array([]),  # CBS1
        "MTHFR1": jnp.log(jnp.array([2.45e-06])),  # ahcys MTHFR1,
    },
    log_dc_inhibitor={
        "MAT3": jnp.array([]),  # MAT3
        "GNMT1": jnp.log(jnp.array([0.000228576])),  # mlthf GNMT1
        "CBS1": jnp.log(jnp.array([9.30e-05])),  # amet CBS1
        "MTHFR1": jnp.log(jnp.array([1.46e-05])),  # amet MTHFR1
    },
)

model = RateEquationModel(
    stoichiometry=stoichiometry,
    species=species,
    reactions=reactions,
    balanced_species=balanced_species,
    rate_equations=[
        Drain(sign=1.0),  # met-L source
        IrreversibleMichaelisMenten(  # MAT1
            ix_ki_species=np.array([4], dtype=np.int16),
        ),
        AllostericIrreversibleMichaelisMenten(  # MAT3
            subunits=2,
            ix_allosteric_activators=np.array([0, 4], dtype=np.int16),
        ),
        IrreversibleMichaelisMenten(  # METH
            ix_ki_species=np.array([5], dtype=np.int16)
        ),
        AllostericIrreversibleMichaelisMenten(  # GNMT1
            subunits=4,
            ix_allosteric_inhibitors=np.array([12], dtype=np.int16),
            ix_allosteric_activators=np.array([4], dtype=np.int16),
            ix_ki_species=np.array([5], dtype=np.int16),
        ),
        ReversibleMichaelisMenten(  # AHC
            water_stoichiometry=-1.0,
        ),
        IrreversibleMichaelisMenten(),  # MS
        IrreversibleMichaelisMenten(),  # BHMT
        AllostericIrreversibleMichaelisMenten(  # CBS1
            subunits=2,
            ix_allosteric_inhibitors=np.array([4], dtype=np.int16),
        ),
        AllostericIrreversibleMichaelisMenten(  # MTHFR
            subunits=2,
            ix_allosteric_inhibitors=np.array([4], dtype=np.int16),
            ix_allosteric_activators=np.array([5], dtype=np.int16),
        ),
        IrreversibleMichaelisMenten(),  # PROT
    ],
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
