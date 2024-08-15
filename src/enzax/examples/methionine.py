"""A simple linear kinetic model."""

from jax import config
from jax import numpy as jnp

from enzax.kinetic_model import (
    KineticModel,
    KineticModelParameters,
    KineticModelStructure,
    UnparameterisedKineticModel,
)
from enzax.rate_equations import (
    AllostericIrreversibleMichaelisMenten,
    Drain,
    IrreversibleMichaelisMenten,
    ReversibleMichaelisMenten,
)


config.update("jax_enable_x64", True)

parameters = KineticModelParameters(
    log_kcat=jnp.log(
        jnp.array(
            [
                7.89577,  # MAT1
                19.9215,  # MAT3
                1.15777,  # METH-Gen
                10.5307,  # GNMT1
                234.284,  # AHC1
                1.77471,  # MS1
                13.7676,  # BHMT1
                7.02307,  # CBS1
                3.1654,  # MTHFR1
                0.264744,  # PROT1
            ]
        )
    ),
    log_enzyme=jnp.log(
        jnp.array(
            [
                0.000961712,  # MAT1
                0.00098812,  # MAT3
                0.00103396,  # METH-Gen
                0.000983692,  # GNMT1
                0.000977878,  # AHC1
                0.00105094,  # MS1
                0.000996603,  # BHMT1
                0.00134056,  # CBS1
                0.0010054,  # MTHFR1
                0.000995525,  # PROT1
            ]
        )
    ),
    log_drain=jnp.log(jnp.array([0.000850127])),
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
    log_km=jnp.log(
        jnp.array(
            [
                0.000106919,  # met-L MAT1
                0.00203015,  # atp MAT1
                0.00113258,  # met-L MAT3
                0.00236759,  # atp MAT3
                9.37e-06,  # amet METH-Gen
                0.000520015,  # amet GNMT1
                0.00253545,  # gly GNMT1
                2.32e-05,  # ahcys AHC1
                1.06e-05,  # hcys-L AHC1
                5.66e-06,  # adn AHC1
                1.71e-06,  # hcys-L MS1
                6.94e-05,  # 5mthf MS1
                1.98e-05,  # hcys-L BHMT1
                0.00845898,  # glyb BHMT1
                4.24e-05,  # hcys-L CBS1
                2.83e-06,  # ser-L CBS1
                8.08e-05,  # mlthf MTHFR1
                2.09e-05,  # nadph MTHFR1
                4.39e-05,  # met-L PROT1
            ]
        )
    ),
    log_ki=jnp.log(
        jnp.array(
            [
                0.000346704,  # MAT1
                5.56e-06,  # METH-Gen
                5.31e-05,  # GNMT1
            ]
        )
    ),
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
    temperature=jnp.array(298.15),
    log_transfer_constant=jnp.log(
        jnp.array(
            [
                0.107657,  # METAT
                131.207,  # GNMT
                1.03452,  # CBS
                0.392035,  # MTHFR
            ]
        )
    ),
    log_dissociation_constant=jnp.log(
        jnp.array(
            [
                0.000316641,  # amet MAT3
                0.00059999,  # met-L MAT3
                1.98e-05,  # amet GNMT1
                0.000228576,  # mlthf GNMT1
                9.30e-05,  # amet CBS1
                1.46e-05,  # amet MTHFR1
                2.45e-06,  # ahcys MTHFR1
            ]
        )
    ),
)
structure = KineticModelStructure(
    S=jnp.array(
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
        ]
    ),
    water_stoichiometry=jnp.array(
        [
            0.0,  # met-L source
            -1.0,  # MAT1 METAT
            -1.0,  # MAT3 METAT
            0.0,  # METH Gen
            0.0,  # GNMT1
            -1.0,  # AHC
            0.0,  # MS
            0.0,  # BHMT
            1.0,  # CBS
            0.0,  # MTHFR
            0.0,  # PROT
        ]
    ),
    balanced_species=jnp.array(
        [
            0,  # met-L
            4,  # amet
            5,  # ahcys
            8,  # hcys-L
            11,  # 5mthf
        ]
    ),
    rate_to_enzyme_ix=[
        [],  # met-L source
        [0],  # MAT1 METAT
        [1],  # MAT3 METAT
        [2],  # METH Gen
        [3],  # GNMT1
        [4],  # AHC
        [5],  # MS
        [6],  # BHMT
        [7],  # CBS
        [8],  # MTHFR
        [9],  # PROT
    ],
    rate_to_km_ixs=[
        [],  # met-L source
        [0, 1],  # MAT1 METAT
        [2, 3],  # MAT3 METAT
        [4],  # METH Gen
        [5, 6],  # GNMT1
        [7, 8, 9],  # AHC
        [10, 11],  # MS
        [12, 13],  # BHMT
        [14, 15],  # CBS
        [16, 17],  # MTHFR
        [18],  # PROT
    ],
    species_to_metabolite_ix=jnp.arange(19),
    rate_to_subunits=jnp.array(
        [
            0,  # met-L source
            1,  # MAT1 METAT
            2,  # MAT3 METAT
            1,  # METH Gen
            4,  # GNMT1
            1,  # AHC
            1,  # MS
            1,  # BHMT
            2,  # CBS
            2,  # MTHFR
            1,  # PROT
        ]
    ),
    rate_to_tc_ix=[
        [],  # met-L source
        [],  # MAT1 METAT
        [0],  # MAT3 METAT
        [],  # METH Gen
        [1],  # GNMT1
        [],  # AHC
        [],  # MS
        [],  # BHMT
        [2],  # CBS
        [3],  # MTHFR
        [],  # PROT
    ],
    rate_to_dc_ixs_activation=[
        [],  # met-L source
        [],  # MAT1 METAT
        [0, 1],  # MAT3 METAT
        [],  # METH Gen
        [2],  # GNMT1
        [],  # AHC
        [],  # MS
        [],  # BHMT
        [4],  # CBS
        [6],  # MTHFR
        [],  # PROT
    ],
    rate_to_dc_ixs_inhibition=[
        [],  # met-L source
        [],  # MAT1 METAT
        [],  # MAT3 METAT
        [],  # METH Gen
        [3],  # GNMT1
        [],  # AHC
        [],  # MS
        [],  # BHMT
        [],  # CBS
        [5],  # MTHFR
        [],  # PROT
    ],
    rate_to_drain_ix=[
        [0],  # met-L source
        [],  # MAT1 METAT
        [],  # MAT3 METAT
        [],  # METH Gen
        [],  # GNMT1
        [],  # AHC
        [],  # MS
        [],  # BHMT
        [],  # CBS
        [],  # MTHFR
        [],  # PROT
    ],
    drain_sign=jnp.array([1.0]),
    dc_to_species_ix=jnp.array(
        [
            4,  # amet -> MAT3
            0,  # met-L -> MAT3
            4,  # amet -> GNMT1
            12,  # mlthf -| GNMT1
            4,  # amet -> CBS1
            4,  # amet -| MTHFR1
            5,  # ahcys -> MTHFR1
        ]
    ),
    ki_to_species_ix=jnp.array(
        [
            4,  # amet -| MAT1
            5,  # ahcys -| METH-GEN
            5,  # ahcys -| GNMT1
        ]
    ),
    rate_to_ki_ixs=[
        [],  # met-L source
        [0],  # MAT1 METAT
        [],  # MAT3 METAT
        [1],  # METH Gen
        [2],  # GNMT1
        [],  # AHC
        [],  # MS
        [],  # BHMT
        [],  # CBS
        [],  # MTHFR
        [],  # PROT
    ],
)
unparameterised_model = UnparameterisedKineticModel(
    structure,
    [
        Drain,  # met-L source
        IrreversibleMichaelisMenten,  # MAT1
        AllostericIrreversibleMichaelisMenten,  # MAT3
        IrreversibleMichaelisMenten,  # METH
        AllostericIrreversibleMichaelisMenten,  # GNMT1
        ReversibleMichaelisMenten,  # AHC
        IrreversibleMichaelisMenten,  # MS
        IrreversibleMichaelisMenten,  # BHMT
        AllostericIrreversibleMichaelisMenten,  # CBS
        AllostericIrreversibleMichaelisMenten,  # MTHFR
        IrreversibleMichaelisMenten,  # PROT
    ],
)
model = KineticModel(parameters, unparameterised_model)
steady_state = jnp.array(
    [
        4.233000e-05,  # met-L
        3.099670e-05,  # amet
        2.170170e-07,  # ahcys
        3.521780e-06,  # hcys
        6.534400e-06,  # 5mthf
    ]
)
