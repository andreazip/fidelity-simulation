import numpy as np
from dataclasses import dataclass

# =========================
# Data structure
# =========================

@dataclass(frozen=True)
class GateAngles:
    theta1: float
    theta2: float
    theta3: float
    theta4: float


# =========================
# Default per-gate settings
# =========================

@dataclass(frozen=True)
class GateDefaults:
    deltat: float | None = None
    deltaV: float | None = None
    T: float | None = None


# =========================
# Fundamental angles (Table S1)
# =========================

THETA_1 = np.arctan(np.sqrt(8))                # ≈ 70.529°
THETA_2 = np.pi - np.arctan(np.sqrt(5) / 2)    # ≈ 131.81°
THETA_3 = np.deg2rad(74.755)
THETA_4 = np.deg2rad(201.625)


# =========================
# Clifford gate library
# Order: Rz(theta1) → Rn(theta2) → Rz(theta3) → Rn(theta4)
# A value of 0 means "not used"
# =========================

GATE_LIBRARY = {

    # ---- Identity & Paulis ----
    "I": GateAngles(0, 0, 0, 0),

    "X": GateAngles(
        0,
        np.pi - THETA_1,
        THETA_1,
        np.pi - THETA_1,
    ),

    "Y": GateAngles(
        np.pi,
        np.pi - THETA_1,
        THETA_1,
        np.pi - THETA_1,
    ),

    "Z": GateAngles(
        np.pi,
        0,
        0,
        0,
    ),

    # ---- Phase gates ----
    "S": GateAngles(
        3 * np.pi / 2,
        0,
        0,
        0,
    ),

    "S†": GateAngles(
        np.pi / 2,
        0,
        0,
        0,
    ),

    # ---- SX family ----
    "SX": GateAngles(
        0,
        THETA_3,
        THETA_2,
        THETA_4,
    ),

    "S†X": GateAngles(
        0,
        THETA_4,
        THETA_2,
        THETA_3,
    ),

    # ---- Hadamard and variants ----
    "H": GateAngles(
        (np.pi - THETA_1) / 2,
        np.pi + THETA_1,
        (np.pi - THETA_1) / 2,
        0,
    ),

    "XH": GateAngles(
        (np.pi + THETA_1) / 2,
        np.pi - THETA_1,
        (3 * np.pi + THETA_1) / 2,
        0,
    ),

    "YH": GateAngles(
        (np.pi + THETA_1) / 2,
        np.pi - THETA_1,
        (np.pi + THETA_1) / 2,
        0,
    ),

    "ZH": GateAngles(
        (3 * np.pi + THETA_1) / 2,
        np.pi - THETA_1,
        (np.pi + THETA_1) / 2,
        0,
    ),

    # ---- SH / HS family ----
    "SH": GateAngles(
        (np.pi - THETA_1) / 2,
        np.pi + THETA_1,
        2 * np.pi - THETA_1 / 2,
        0,
    ),

    "HS": GateAngles(
        2 * np.pi - THETA_1 / 2,
        np.pi + THETA_1,
        (np.pi - THETA_1) / 2,
        0,
    ),

    "S†H": GateAngles(
        (3 * np.pi + THETA_1) / 2,
        np.pi - THETA_1,
        THETA_1 / 2,
        0,
    ),

    "HS†": GateAngles(
        THETA_1 / 2,
        np.pi - THETA_1,
        (3 * np.pi + THETA_1) / 2,
        0,
    ),

    # ---- HSH family ----
    "HSH": GateAngles(
        THETA_1 / 2,
        np.pi - THETA_1,
        THETA_1 / 2,
        0,
    ),

    "HS†H": GateAngles(
        np.pi + THETA_1 / 2,
        np.pi - THETA_1,
        np.pi + THETA_1 / 2,
        0,
    ),

    "S†HS": GateAngles(
        np.pi + THETA_1 / 2,
        np.pi - THETA_1,
        THETA_1 / 2,
        0,
    ),

    "SHS†": GateAngles(
        THETA_1 / 2,
        np.pi - THETA_1,
        np.pi + THETA_1 / 2,
        0,
    ),

    # ---- SXH family ----
    "HSX": GateAngles(
        THETA_1 / 2,
        np.pi - THETA_1,
        (np.pi + THETA_1) / 2,
        0,
    ),

    "S†XH": GateAngles(
        (np.pi + THETA_1) / 2,
        np.pi - THETA_1,
        THETA_1 / 2,
        0,
    ),

    "HS†X": GateAngles(
        2 * np.pi - THETA_1 / 2,
        np.pi + THETA_1,
        (3 * np.pi - THETA_1) / 2,
        0,
    ),

    "SXH": GateAngles(
        (3 * np.pi - THETA_1) / 2,
        np.pi + THETA_1,
        2 * np.pi - THETA_1 / 2,
        0,
    ),
}

# Recommended resolution/time per gate (optional)
# Values provided for commonly used gates; others default to None.
GATE_DEFAULTS: dict[str, GateDefaults] = {
    # Based on current experiments
    "Y": GateDefaults(deltat=60e-12, deltaV=80e-6, T=80e-9),
    "X": GateDefaults(deltat=69e-12, deltaV=100e-6, T=60e-9),
    "SXH": GateDefaults(deltat=70e-12, deltaV=40e-6, T=120e-9),
}

def get_gate_angles(gate: str) -> GateAngles:
    try:
        return GATE_LIBRARY[gate]
    except KeyError:
        raise ValueError(
            f"Unknown gate '{gate}'. Available gates:\n{list(GATE_LIBRARY.keys())}"
        )


def get_gate_defaults(gate: str) -> GateDefaults:
    """Return recommended `deltat`, `deltaV`, and `T` for a gate if available.

    If a gate has no defaults defined, returns a `GateDefaults` with `None` values.
    """
    return GATE_DEFAULTS.get(gate, GateDefaults())