"""A kinetic model of the methionine cycle.

See here for more about the methionine cycle:
https://doi.org/10.1021/acssynbio.3c00662

"""

import equinox as eqx
import numpy as np
from jax import numpy as jnp
from jaxtyping import Array, Scalar

from enzax.kinetic_model import (
    RateEquationKineticModelStructure,
    RateEquationModel,
)
from enzax.rate_equations import (
    AllostericIrreversibleMichaelisMenten,
    Drain,
    IrreversibleMichaelisMenten,
    ReversibleMichaelisMenten,
)


class ParameterDefinition(eqx.Module):
    log_km: dict[int, list[Array]]
    log_kcat: dict[int, Scalar]
    log_enzyme: dict[int, Array]
    log_ki: dict[int, Array]
    dgf: Array
    temperature: Scalar
    log_conc_unbalanced: Array
    log_dc_inhibitor: dict[int, Array]
    log_dc_activator: dict[int, Array]
    log_tc: dict[int, Array]
    log_drain: dict[int, Scalar]


S = np.array(
    [
        [1, -1, -1, 0, 0, 0, 1, 1, 0, 0, -1],  # met-L b
        [0, -1, -1, 0, 0, 0, 0, 0, 0, 0, 0],  # atp
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # pi
        [0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],  # ppi
        [0, 1, 1, -1, -1, 0, 0, 0, 0, 0, 0],  # amet b
        [0, 0, 0, 1, 1, -1, 0, 0, 0, 0, 0],  # ahcys b
        [0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0],  # gly
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],  # sarcs
        [0, 0, 0, 0, 0, 1, -1, -1, -1, 0, 0],  # hcys-L b
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # adn
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # thf
        [0, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0],  # 5mthf b
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],  # mlthf
        [0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0],  # glyb
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # dmgly
        [0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0],  # ser-L
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # nadp
        [0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0],  # nadph
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # cyst-L
    ],
    dtype=np.float64,
)
reactions = []
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
parameters = ParameterDefinition(
    log_kcat={
        1: jnp.log(jnp.array(7.89577)),  # MAT1
        2: jnp.log(jnp.array(19.9215)),  # MAT3
        3: jnp.log(jnp.array(1.15777)),  # METH-Gen
        4: jnp.log(jnp.array(10.5307)),  # GNMT1
        5: jnp.log(jnp.array(234.284)),  # AHC1
        6: jnp.log(jnp.array(1.77471)),  # MS1
        7: jnp.log(jnp.array(13.7676)),  # BHMT1
        8: jnp.log(jnp.array(7.02307)),  # CBS1
        9: jnp.log(jnp.array(3.1654)),  # MTHFR1
        10: jnp.log(jnp.array(0.264744)),  # PROT1
    },
    log_enzyme={
        1: jnp.log(jnp.array(0.000961712)),  # MAT1
        2: jnp.log(jnp.array(0.00098812)),  # MAT3
        3: jnp.log(jnp.array(0.00103396)),  # METH-Gen
        4: jnp.log(jnp.array(0.000983692)),  # GNMT1
        5: jnp.log(jnp.array(0.000977878)),  # AHC1
        6: jnp.log(jnp.array(0.00105094)),  # MS1
        7: jnp.log(jnp.array(0.000996603)),  # BHMT1
        8: jnp.log(jnp.array(0.00134056)),  # CBS1
        9: jnp.log(jnp.array(0.0010054)),  # MTHFR1
        10: jnp.log(jnp.array(0.000995525)),  # PROT1
    },
    log_drain={0: jnp.log(jnp.array(0.000850127))},
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
    log_km={
        1: [jnp.log(jnp.array([0.000106919, 0.00203015]))],  # MAT1 met-L, atp
        2: [jnp.log(jnp.array([0.00113258, 0.00236759]))],  # MAT3 met-L atp
        3: [jnp.log(jnp.array([9.37e-06]))],  # amet METH-Gen
        4: [
            jnp.log(jnp.array([0.000520015, 0.00253545]))
        ],  # amet GNMT1,  # gly GNMT1
        5: [
            jnp.log(jnp.array([2.32e-05])),  # ahcys AHC1
            jnp.log(
                jnp.array([1.06e-05, 5.66e-06])
            ),  # hcys-L AHC1,  # adn AHC1
        ],
        6: [
            jnp.log(jnp.array([1.71e-06, 6.94e-05]))
        ],  # hcys-L MS1,  # 5mthf MS1
        7: [
            jnp.log(jnp.array([1.98e-05, 0.00845898]))
        ],  # hcys-L BHMT1,  # glyb BHMT1
        8: [
            jnp.log(jnp.array([4.24e-05, 2.83e-06]))
        ],  # hcys-L CBS1,  # ser-L CBS1
        9: [
            jnp.log(jnp.array([8.08e-05, 2.09e-05]))
        ],  # mlthf MTHFR1,  # nadph MTHFR1
        10: [jnp.log(jnp.array([4.39e-05]))],  # met-L PROT1
    },
    temperature=jnp.array(298.15),
    log_ki={
        1: jnp.array([jnp.log(0.000346704)]),  # MAT1
        2: jnp.array([]),
        3: jnp.array([jnp.log(5.56e-06)]),  # METH-Gen
        4: jnp.array([jnp.log(5.31e-05)]),  # GNMT1
        5: jnp.array([]),
        6: jnp.array([]),
        7: jnp.array([]),
        8: jnp.array([]),
        9: jnp.array([]),
        10: jnp.array([]),
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
        2: jnp.array(jnp.log(0.107657)),  # MAT3
        4: jnp.array(jnp.log(131.207)),  # GNMT
        8: jnp.array(jnp.log(1.03452)),  # CBS
        9: jnp.array(jnp.log(0.392035)),  # MTHFR
    },
    log_dc_activator={
        2: jnp.log(
            jnp.array([0.00059999, 0.000316641])
        ),  # met-L MAT3,  # amet MAT3
        4: jnp.log(jnp.array([1.98e-05])),  # amet GNMT1
        8: jnp.array([]),  # CBS1
        9: jnp.log(jnp.array([2.45e-06])),  # ahcys MTHFR1,
    },
    log_dc_inhibitor={
        2: jnp.array([]),  # MAT3
        4: jnp.log(jnp.array([0.000228576])),  # mlthf GNMT1
        8: jnp.log(jnp.array([9.30e-05])),  # amet CBS1
        9: jnp.log(jnp.array([1.46e-05])),  # amet MTHFR1
    },
)

structure = RateEquationKineticModelStructure(
    S=S,
    species=species,
    reactions=reactions,
    balanced_species=balanced_species,
    rate_equations=[
        Drain(sign=1.0),  # met-L source
        IrreversibleMichaelisMenten(),  # MAT1
        AllostericIrreversibleMichaelisMenten(  # MAT3
            subunits=2,
            ix_allosteric_activators=np.array([0, 4], dtype=np.int16),
        ),
        IrreversibleMichaelisMenten(),  # METH
        AllostericIrreversibleMichaelisMenten(  # GNMT1
            subunits=4,
            ix_allosteric_inhibitors=np.array([12], dtype=np.int16),
            ix_allosteric_activators=np.array([4], dtype=np.int16),
        ),
        ReversibleMichaelisMenten(
            water_stoichiometry=-1.0,
        ),  # AHC
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
model = RateEquationModel(parameters, structure)
