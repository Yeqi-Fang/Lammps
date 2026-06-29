"""Paper constants for Zeng et al., J. Chem. Phys. 163, 084512 (2025)."""

SYSTEM_BD = "BD"
SYSTEM_MD3D = "MD3D"
SYSTEM_MD2D = "MD2D"

LC2 = {
    SYSTEM_BD: 0.05,
    SYSTEM_MD3D: 0.057,
    SYSTEM_MD2D: 0.048,
}

OVERLAP_A = {
    SYSTEM_BD: 0.2,
    SYSTEM_MD3D: 0.25,
    SYSTEM_MD2D: 0.3,
}

DIMENSION = {
    SYSTEM_BD: 3,
    SYSTEM_MD3D: 3,
    SYSTEM_MD2D: 2,
}

BIG_TYPE = 1
SMALL_TYPE = 2

ANALYSIS_DT = {
    SYSTEM_BD: {
        1.0: 0.005,
        3.0: 0.005,
        10.0: 0.002,
        20.0: 0.001,
    },
    SYSTEM_MD3D: {
        0.003: 0.2,
        0.005: 0.15,
        0.015: 0.1,
        0.03: 0.1,
        0.06: 0.05,
    },
    SYSTEM_MD2D: {
        1.0e-4: 2.0,
        5.0e-4: 1.0,
        1.0e-3: 0.5,
        5.0e-3: 0.1,
        1.0e-2: 0.05,
    },
}

FIG2_BD_RATE = 3.0
FIG6_MD3D_RATE = 0.015
FIG6_MD2D_RATE = 0.001

THRESHOLD_TOL = 0.10
C_PRIME = 2.0
STZ_DIAMETER = 3.0


def analysis_dt(system, shear_rate):
    table = ANALYSIS_DT[system]
    for key, value in table.items():
        if abs(float(shear_rate) - key) <= max(1.0e-12, 1.0e-8 * abs(key)):
            return value
    raise ValueError("No paper Delta t for system={} shear_rate={}".format(system, shear_rate))


def bd_coarse_length(box_length, shear_rate):
    if abs(float(shear_rate) - 20.0) < 1.0e-12:
        return box_length / 20.0
    return box_length / 10.0


def undersize_grid_length(system, box_length):
    if system == SYSTEM_MD2D:
        return box_length / 80.0
    return box_length / 30.0
