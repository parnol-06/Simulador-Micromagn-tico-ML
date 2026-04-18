"""
=============================================================================
 temperature_model.py  ·  Post-ML thermal correction
 Micromagnetic ML Simulator — Phase 4

 Separates ML ↔ physics so the model does NOT extrapolate at extreme
 temperatures. The ML always predicts at T_sim (training temperature)
 and this module applies the physical correction afterwards.

 Laws used:
   · Bloch (1930)          — Ms(T)  ∝  (1 − τ^1.5)^(1/3)   τ = T/Tc
   · Callen-Callen (1966)  — K1(T)  ∝  Ms(T)^(10/3)
                           — Hc(T)  ∝  Ms(T)^(7/3)  ·  K1(T)/Ms(T)
                           — Mr(T)  ∝  Ms(T)
   · SPM (Néel criterion)  — E_b = K1(T) · V / (kB · T)
                             E_b < 25 → superparamagnetic → Hc→0, Mr→0

 Public functions:
   · to_kelvin(value, unit)               → float
   · from_kelvin(T_K, unit)              → float
   · reduced_magnetization(T_K, Tc_K)   → float  [0, 1]
   · apply_temperature_to_hc_mr(...)    → (Hc_T, Mr_T, barrier)
=============================================================================
"""

from __future__ import annotations
import numpy as np

# Boltzmann constant [J/K]
_kB: float = 1.380_649e-23

# SPM threshold (in kBT units): E_b/kBT < SPM_THRESHOLD → superparamagnetic
SPM_THRESHOLD: float = 25.0

# ML engine training reference temperature [K]
T_SIM_REF: float = 300.0

# ── Physical exponents (named to avoid magic numbers in calculations) ─────────

# Bloch (1930): ms(T) = (1 − τ^BLOCH_TAU_EXP)^BLOCH_REDUCED_EXP
# τ = T/Tc
_BLOCH_TAU_EXP: float = 1.5        # exponent on reduced temperature τ
_BLOCH_REDUCED_EXP: float = 1.0 / 3.0  # outer exponent

# Callen-Callen (1966): Hc ∝ ms^HC_EXP, K1 ∝ ms^K1_EXP
_CALLEN_HC_EXP: float = 7.0 / 3.0   # Hc temperature scaling exponent
_CALLEN_K1_EXP: float = 10.0 / 3.0  # K1 temperature scaling exponent

# Geometry: sphere volume V = SPHERE_VOL_FACTOR · π · r³
_SPHERE_VOL_FACTOR: float = 4.0 / 3.0

# Unit conversions
_NM_TO_M: float = 1e-9    # nanometres → metres
_KJM3_TO_JM3: float = 1e3  # kJ/m³ → J/m³
_MIN_TEMPERATURE_K: float = 1.0  # clamp floor for T_K


# ─────────────────────────────────────────────────────────────────────────────
#  Unit conversion
# ─────────────────────────────────────────────────────────────────────────────

def to_kelvin(value: float, unit: str) -> float:
    """Convert a temperature value to Kelvin.

    Parameters
    ----------
    value : float
        Temperature value in the specified unit.
    unit  : str
        ``'K'`` for Kelvin or ``'C'`` / ``'°C'`` for Celsius.

    Returns
    -------
    float
        Temperature in Kelvin (minimum 1 K).
    """
    unit = unit.strip().upper().replace('°', '')
    if unit == 'C':
        T_K = value + 273.15
    else:
        T_K = float(value)
    return max(_MIN_TEMPERATURE_K, float(T_K))


def from_kelvin(T_K: float, unit: str) -> float:
    """Convert Kelvin to the specified unit.

    Parameters
    ----------
    T_K  : float  Temperature in Kelvin.
    unit : str    ``'K'`` or ``'C'`` / ``'°C'``.

    Returns
    -------
    float  Temperature in the target unit.
    """
    unit = unit.strip().upper().replace('°', '')
    if unit == 'C':
        return float(T_K) - 273.15
    return float(T_K)


# ─────────────────────────────────────────────────────────────────────────────
#  Reduced magnetization  (Bloch 1930)
# ─────────────────────────────────────────────────────────────────────────────

def reduced_magnetization(T_K: float, Tc_K: float) -> float:
    """Compute ms(T) = Ms(T)/Ms(0) using the Bloch law.

    .. math::
        m_s(T) = \\left(1 - \\tau^{3/2}\\right)^{1/3}, \\quad \\tau = T/T_c

    Returns 0.0 for T ≥ Tc and 1.0 for T = 0 K.

    Parameters
    ----------
    T_K  : float  Operating temperature [K].
    Tc_K : float  Curie temperature [K].

    Returns
    -------
    float  Reduced magnetization in [0, 1].
    """
    if Tc_K <= 0.0:
        return 1.0
    tau = float(T_K) / float(Tc_K)
    if tau >= 1.0:
        return 0.0
    tau = max(0.0, tau)
    return float(np.clip((1.0 - tau ** _BLOCH_TAU_EXP) ** _BLOCH_REDUCED_EXP, 0.0, 1.0))


# ─────────────────────────────────────────────────────────────────────────────
#  Callen-Callen correction + SPM criterion
# ─────────────────────────────────────────────────────────────────────────────

def apply_temperature_to_hc_mr(
    Hc_ref_mT: float,
    Mr_ref: float,
    *,
    d_nm: float,
    Ms_MA_m: float,
    K1_kJ_m3: float,
    Tc_K: float,
    T_K: float,
    T_ref_K: float = T_SIM_REF,
) -> tuple[float, float, float]:
    """Apply Callen-Callen thermal correction + SPM criterion.

    The ML predicts Hc and Mr at ``T_ref_K`` (training temperature).
    This function scales the results to ``T_K`` using physical laws and
    checks whether the system crosses into the superparamagnetic regime.

    Callen-Callen scaling (1966):
    -------
    .. math::
        m_s(T)   = \\left(1 - (T/T_c)^{1.5}\\right)^{1/3}

        H_c(T)   = H_c^{ref} \\cdot \\frac{m_s(T)^{7/3}}{m_s(T_{ref})^{7/3}}

        M_r(T)   = M_r^{ref} \\cdot \\frac{m_s(T)}{m_s(T_{ref})}

        K_1(T)   = K_1(0) \\cdot m_s(T)^{10/3}

    SPM criterion (Néel):
    -------
    .. math::
        E_b = K_1(T) \\cdot V \\;,\\quad V = \\frac{4}{3}\\pi r^3

        \\text{SPM if } E_b / (k_B T) < 25

    In the SPM regime a smooth reduction of Hc and Mr towards 0 is applied.

    Parameters
    ----------
    Hc_ref_mT : float  Coercive field predicted by ML at T_ref_K [mT].
    Mr_ref    : float  Normalized remanence (Mr/Ms) at T_ref_K [0-1].
    d_nm      : float  Nanoparticle diameter [nm].
    Ms_MA_m   : float  Saturation magnetization at 0 K [MA/m].
    K1_kJ_m3  : float  Anisotropy constant at 0 K [kJ/m³]  (from MATERIALS_DB).
    Tc_K      : float  Curie temperature [K].
    T_K       : float  Target temperature [K].
    T_ref_K   : float  ML reference temperature [K]. Default 300 K.

    Returns
    -------
    tuple[float, float, float]
        ``(Hc_T, Mr_T, barrier)``
        - Hc_T    : Hc corrected to T_K [mT].
        - Mr_T    : Mr/Ms corrected to T_K.
        - barrier : Dimensionless energy barrier E_b / (k_B·T).
    """
    T_K = max(_MIN_TEMPERATURE_K, float(T_K))

    # T ≥ Tc → paramagnetic state
    if T_K >= float(Tc_K):
        return 0.0, 0.0, 0.0

    # Reduced magnetizations at T_ref and T_K
    ms_ref = reduced_magnetization(T_ref_K, Tc_K)
    ms_T   = reduced_magnetization(T_K,     Tc_K)

    # Avoid division by zero if ms_ref is nearly zero
    if ms_ref < 1e-9:
        return 0.0, 0.0, 0.0

    # ── Callen-Callen scaling ─────────────────────────────────────────────────
    ratio_ms = ms_T / ms_ref

    # Hc ∝ ms^(7/3)  (anisotropy field × domain-wall reduction)
    Hc_T = float(Hc_ref_mT) * (ratio_ms ** _CALLEN_HC_EXP)

    # Mr ∝ ms
    Mr_T = float(Mr_ref) * ratio_ms

    # ── K1(T) via Callen-Callen: K1 ∝ ms^(10/3) ─────────────────────────────
    # Use the actual K1 from MATERIALS_DB (not a proxy from Hc or Ms)
    K1_0_J_m3 = abs(float(K1_kJ_m3)) * _KJM3_TO_JM3      # kJ/m³ → J/m³
    K1_T_J_m3 = K1_0_J_m3 * (ms_T ** _CALLEN_K1_EXP)      # Callen-Callen scaling

    # ── SPM barrier: E_b = K1(T) · V ─────────────────────────────────────────
    r_m  = (float(d_nm) / 2.0) * _NM_TO_M             # radius [m]
    V_m3 = _SPHERE_VOL_FACTOR * np.pi * r_m ** 3       # sphere volume [m³]

    E_b     = K1_T_J_m3 * V_m3                        # [J]
    barrier = E_b / (_kB * T_K)                        # dimensionless

    # ── Smooth SPM reduction ──────────────────────────────────────────────────
    if barrier < SPM_THRESHOLD:
        # Linear reduction: 0 at barrier=0, full at barrier=SPM_THRESHOLD
        spm_factor = max(0.0, barrier / SPM_THRESHOLD)
        Hc_T = Hc_T * spm_factor
        Mr_T = Mr_T * spm_factor

    # Final clamp
    Hc_T = max(0.0, Hc_T)
    Mr_T = float(np.clip(Mr_T, 0.0, 1.0))

    return Hc_T, Mr_T, float(barrier)
