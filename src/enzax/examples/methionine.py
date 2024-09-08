"""A kinetic model of the methionine cycle.

See here for more about the methionine cycle:
https://doi.org/10.1021/acssynbio.3c00662

"""

from jax import numpy as jnp

from enzax.kinetic_model import (
    RateEquationModel,
    KineticModelStructure,
)
from enzax.rate_equations import (
    AllostericIrreversibleMichaelisMenten,
    Drain,
    IrreversibleMichaelisMenten,
    ReversibleMichaelisMenten,
)
from enzax.parameters import AllostericMichaelisMentenParameterSet

coords = {
    "enzyme": [
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
    ],
    "drain": ["the_drain"],
    "rate": [
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
    ],
    "metabolite": [
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
    ],
    "km": [
        "met-L MAT1",
        "atp MAT1",
        "met-L MAT3",
        "atp MAT3",
        "amet METH-Gen",
        "amet GNMT1",
        "gly GNMT1",
        "ahcys AHC1",
        "hcys-L AHC1",
        "adn AHC1",
        "hcys-L MS1",
        "5mthf MS1",
        "hcys-L BHMT1",
        "glyb BHMT1",
        "hcys-L CBS1",
        "ser-L CBS1",
        "mlthf MTHFR1",
        "nadph MTHFR1",
        "met-L PROT1",
    ],
    "ki": [
        "MAT1",
        "METH-Gen",
        "GNMT1",
    ],
    "species": [
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
    ],
    "balanced_species": [
        "met-L",
        "amet",
        "ahcys",
        "hcys-L",
        "5mthf",
    ],
    "unbalanced_species": [
        "atp",
        "pi",
        "ppi",
        "gly",
        "sarcs",
        "adn",
        "thf",
        "mlthf",
        "glyb",
        "dmgly",
        "ser-L",
        "nadp",
        "nadph",
        "cyst-L",
    ],
    "transfer_constant": [
        "METAT",
        "GNMT",
        "CBS",
        "MTHFR",
    ],
    "dissociation_constant": [
        "met-L MAT3",
        "amet MAT3",
        "amet GNMT1",
        "mlthf GNMT1",
        "amet CBS1",
        "amet MTHFR1",
        "ahcys MTHFR1",
    ],
}
dims = {
    "log_kcat": ["enzyme"],
    "log_enzyme": ["enzyme"],
    "log_drain": ["drain"],
    "log_km": ["km"],
    "dgf": ["metabolite"],
    "log_ki": ["ki"],
    "log_conc_unbalanced": ["unbalanced_species"],
    "log_transfer_constant": ["transfer_constant"],
    "log_dissociation_constant": ["dissociation_constant"],
}
parameters = AllostericMichaelisMentenParameterSet(
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
                0.00059999,  # met-L MAT3
                0.000316641,  # amet MAT3
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
        ],
        dtype=jnp.float64,
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
    unbalanced_species=jnp.array(
        [
            1,  # atp
            2,  # pi
            3,  # ppi
            6,  # gly
            7,  #  sarcs
            9,  # adn
            10,  # thf
            12,  # mlthf
            13,  # glyb
            14,  # dmgly
            15,  # ser-L
            16,  # nadp
            17,  # nadph
            18,  # cyst-L
        ]
    ),
)
rate_equations = [
    Drain(sign=jnp.array(1.0), drain_ix=0),  # met-L source
    IrreversibleMichaelisMenten(  # MAT1
        kcat_ix=0,
        enzyme_ix=0,
        km_ix=jnp.array([0, 1], dtype=jnp.int16),
        ki_ix=jnp.array([0], dtype=jnp.int16),
        reactant_stoichiometry=jnp.array(
            [-1.0, -1.0, 1.0, 1.0, 1.0], dtype=jnp.int16
        ),
        ix_substrate=jnp.array([0, 1], dtype=jnp.int16),
        ix_ki_species=jnp.array([4], dtype=jnp.int16),
        substrate_km_positions=jnp.array([0, 1], dtype=jnp.int16),
        substrate_reactant_positions=jnp.array([0, 1], dtype=jnp.int16),
    ),
    AllostericIrreversibleMichaelisMenten(  # MAT3
        kcat_ix=1,
        enzyme_ix=1,
        km_ix=jnp.array([2, 3], dtype=jnp.int16),
        ki_ix=jnp.array([], dtype=jnp.int16),
        reactant_stoichiometry=jnp.array(
            [-1.0, -1.0, 1.0, 1.0, 1.0], dtype=jnp.int16
        ),
        ix_substrate=jnp.array([0, 1], dtype=jnp.int16),
        ix_ki_species=jnp.array([], dtype=jnp.int16),
        substrate_km_positions=jnp.array([0, 1], dtype=jnp.int16),
        substrate_reactant_positions=jnp.array([0, 1], dtype=jnp.int16),
        subunits=2,
        tc_ix=0,
        ix_dc_inhibition=jnp.array([], dtype=jnp.int16),
        ix_dc_activation=jnp.array([0, 1], dtype=jnp.int16),
        species_inhibition=jnp.array([], dtype=jnp.int16),
        species_activation=jnp.array([0, 4], dtype=jnp.int16),
    ),
    IrreversibleMichaelisMenten(  # METH
        kcat_ix=2,
        enzyme_ix=2,
        km_ix=jnp.array([4], dtype=jnp.int16),
        ki_ix=jnp.array([1], dtype=jnp.int16),
        reactant_stoichiometry=jnp.array([-1, 1], dtype=jnp.int16),
        ix_substrate=jnp.array([4], dtype=jnp.int16),
        ix_ki_species=jnp.array([5], dtype=jnp.int16),
        substrate_km_positions=jnp.array([0], dtype=jnp.int16),
        substrate_reactant_positions=jnp.array([0], dtype=jnp.int16),
    ),
    AllostericIrreversibleMichaelisMenten(  # GNMT1
        kcat_ix=3,
        enzyme_ix=3,
        km_ix=jnp.array([5, 6], dtype=jnp.int16),
        ki_ix=jnp.array([2], dtype=jnp.int16),
        reactant_stoichiometry=jnp.array(
            [-1.0, 1.0, -1.0, 1.0], dtype=jnp.int16
        ),
        ix_substrate=jnp.array([4, 6], dtype=jnp.int16),
        ix_ki_species=jnp.array([5], dtype=jnp.int16),
        substrate_km_positions=jnp.array([0, 1], dtype=jnp.int16),
        substrate_reactant_positions=jnp.array([0, 2], dtype=jnp.int16),
        subunits=4,
        tc_ix=1,
        ix_dc_activation=jnp.array([2], dtype=jnp.int16),
        ix_dc_inhibition=jnp.array([3], dtype=jnp.int16),
        species_inhibition=jnp.array([12], dtype=jnp.int16),
        species_activation=jnp.array([4], dtype=jnp.int16),
    ),
    ReversibleMichaelisMenten(  # AHC
        kcat_ix=4,
        enzyme_ix=4,
        km_ix=jnp.array([7, 8, 9], dtype=jnp.int16),
        ki_ix=jnp.array([], dtype=jnp.int16),
        reactant_stoichiometry=jnp.array([-1.0, 1.0, 1.0], dtype=jnp.int16),
        ix_substrate=jnp.array([5], dtype=jnp.int16),
        ix_product=jnp.array([8, 9], dtype=jnp.int16),
        ix_reactants=jnp.array([5, 8, 9], dtype=jnp.int16),
        ix_ki_species=jnp.array([], dtype=jnp.int16),
        substrate_km_positions=jnp.array([0], dtype=jnp.int16),
        product_km_positions=jnp.array([1, 2], dtype=jnp.int16),
        substrate_reactant_positions=jnp.array([0], dtype=jnp.int16),
        product_reactant_positions=jnp.array([1, 2], dtype=jnp.int16),
        water_stoichiometry=jnp.array(-1.0),
        reactant_to_dgf=jnp.array([5, 8, 9], dtype=jnp.int16),
    ),
    IrreversibleMichaelisMenten(  # MS
        kcat_ix=5,
        enzyme_ix=5,
        km_ix=jnp.array([10, 11], dtype=jnp.int16),
        ki_ix=jnp.array([], dtype=jnp.int16),
        reactant_stoichiometry=jnp.array([1, -1, 1, -1], dtype=jnp.int16),
        ix_substrate=jnp.array([8, 11], dtype=jnp.int16),
        ix_ki_species=jnp.array([], dtype=jnp.int16),
        substrate_km_positions=jnp.array([0, 1], dtype=jnp.int16),
        substrate_reactant_positions=jnp.array([1, 3], dtype=jnp.int16),
    ),
    IrreversibleMichaelisMenten(  # BHMT
        kcat_ix=6,
        enzyme_ix=6,
        km_ix=jnp.array([12, 13], dtype=jnp.int16),
        ki_ix=jnp.array([], dtype=jnp.int16),
        reactant_stoichiometry=jnp.array([1, -1, -1, 1], dtype=jnp.int16),
        ix_substrate=jnp.array([8, 13], dtype=jnp.int16),
        ix_ki_species=jnp.array([], dtype=jnp.int16),
        substrate_km_positions=jnp.array([0, 1], dtype=jnp.int16),
        substrate_reactant_positions=jnp.array([1, 2], dtype=jnp.int16),
    ),
    AllostericIrreversibleMichaelisMenten(  # CBS1
        kcat_ix=7,
        enzyme_ix=7,
        km_ix=jnp.array([14, 15], dtype=jnp.int16),
        ki_ix=jnp.array([], dtype=jnp.int16),
        reactant_stoichiometry=jnp.array([-1, -1, 1], dtype=jnp.int16),
        ix_substrate=jnp.array([8, 15], dtype=jnp.int16),
        ix_ki_species=jnp.array([], dtype=jnp.int16),
        substrate_km_positions=jnp.array([0, 1], dtype=jnp.int16),
        substrate_reactant_positions=jnp.array([0, 1], dtype=jnp.int16),
        subunits=2,
        tc_ix=2,
        ix_dc_activation=jnp.array([4], dtype=jnp.int16),
        ix_dc_inhibition=jnp.array([], dtype=jnp.int16),
        species_inhibition=jnp.array([4], dtype=jnp.int16),
        species_activation=jnp.array([], dtype=jnp.int16),
    ),
    AllostericIrreversibleMichaelisMenten(  # MTHFR
        kcat_ix=8,
        enzyme_ix=8,
        km_ix=jnp.array([16, 17], dtype=jnp.int16),
        ki_ix=jnp.array([], dtype=jnp.int16),
        reactant_stoichiometry=jnp.array([1, -1, 1, -1], dtype=jnp.int16),
        ix_substrate=jnp.array([12, 17], dtype=jnp.int16),
        ix_ki_species=jnp.array([], dtype=jnp.int16),
        substrate_km_positions=jnp.array([0, 1], dtype=jnp.int16),
        substrate_reactant_positions=jnp.array([1, 3], dtype=jnp.int16),
        subunits=2,
        tc_ix=3,
        ix_dc_activation=jnp.array([6], dtype=jnp.int16),
        ix_dc_inhibition=jnp.array([5], dtype=jnp.int16),
        species_inhibition=jnp.array([4], dtype=jnp.int16),
        species_activation=jnp.array([5], dtype=jnp.int16),
    ),
    IrreversibleMichaelisMenten(  # PROT
        kcat_ix=9,
        enzyme_ix=9,
        km_ix=jnp.array([18], dtype=jnp.int16),
        ki_ix=jnp.array([], dtype=jnp.int16),
        reactant_stoichiometry=jnp.array([-1.0], dtype=jnp.int16),
        ix_substrate=jnp.array([0], dtype=jnp.int16),
        ix_ki_species=jnp.array([], dtype=jnp.int16),
        substrate_km_positions=jnp.array([0], dtype=jnp.int16),
        substrate_reactant_positions=jnp.array([0], dtype=jnp.int16),
    ),
]
steady_state = jnp.array(
    [
        4.233000e-05,  # met-L
        3.099670e-05,  # amet
        2.170170e-07,  # ahcys
        3.521780e-06,  # hcys
        6.534400e-06,  # 5mthf
    ]
)
model = RateEquationModel(parameters, structure, rate_equations)
