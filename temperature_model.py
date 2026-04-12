"""temperature_model.py — utilidades de temperatura

Convierte unidades (°C ↔ K) y aplica una corrección físico-heurística
para adaptar (Hc, Mr) predichos a una temperatura dada.

Nota:
- El motor ML del proyecto se entrena con datos de referencia a ~300 K.
- Para permitir explorar temperatura sin reentrenar con datasets T-dependientes,
  aquí se aplica una corrección suave basada en:
    * magnetización reducida Ms(T) ~ (1 - (T/Tc)^(3/2))
    * activación térmica vía barrera Δ = K_eff(T)·V / (kB·T)
    * apagado total por encima de Tc

El objetivo es un comportamiento cualitativamente correcto (tendencias), no
una ley universal para todos los materiales.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

_kB = 1.380649e-23  # J/K


def to_kelvin(value: float, unit: str) -> float:
    """Convierte temperatura a Kelvin.

    Parameters
    ----------
    value: valor numérico
    unit: 'K' o '°C' (también acepta 'C', 'celsius')
    """
    u = unit.strip().lower()
    if u in ('k', 'kelvin'):
        T = float(value)
    elif u in ('°c', 'c', 'celsius', 'centigrados', 'centígrados'):
        T = float(value) + 273.15
    else:
        raise ValueError(f"Unidad de temperatura no soportada: {unit!r}")

    # evita 0K exacto para divisiones
    return max(T, 1e-6)


def from_kelvin(T_K: float, unit: str) -> float:
    u = unit.strip().lower()
    if u in ('k', 'kelvin'):
        return float(T_K)
    if u in ('°c', 'c', 'celsius', 'centigrados', 'centígrados'):
        return float(T_K) - 273.15
    raise ValueError(f"Unidad de temperatura no soportada: {unit!r}")


@dataclass(frozen=True)
class TemperatureCorrection:
    """Parámetros de la corrección térmica."""

    barrier_crit: float = 12.0
    barrier_width: float = 2.0
    ms_exp: float = 1.5


def reduced_magnetization(T_K: float, Tc_K: float, ms_exp: float = 1.5) -> float:
    """Ms(T)/Ms(0) aproximada: 1 - (T/Tc)^(3/2) (clipeada a [0,1])."""
    Tc = max(float(Tc_K), 1.0)
    t = float(T_K) / Tc
    if t >= 1.0:
        return 0.0
    return float(np.clip(1.0 - t ** ms_exp, 0.0, 1.0))


def _logistic(x: float) -> float:
    # estable numéricamente
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def apply_temperature_to_hc_mr(
    *,
    Hc_ref_mT: float,
    Mr_ref: float,
    d_nm: float,
    Ms_MA_m: float,
    Tc_K: float,
    T_K: float,
    cfg: TemperatureCorrection = TemperatureCorrection(),
) -> tuple[float, float, float]:
    """Aplica corrección térmica a (Hc, Mr).

    Returns
    -------
    (Hc_mT, Mr, barrier)
    """
    T = max(float(T_K), 1e-6)
    Tc = max(float(Tc_K), 1.0)

    if T >= Tc:
        return 0.0, 0.0, 0.0

    # Ms(T)/Ms(0)
    ms_r = reduced_magnetization(T, Tc, ms_exp=cfg.ms_exp)

    # Volumen (aprox esfera). Esto se usa SOLO para la barrera térmica.
    r_m = (float(d_nm) * 1e-9) / 2.0
    V = (4.0 / 3.0) * math.pi * r_m**3

    # K_eff estimada desde Hc_ref: Hk ~ 2Keff/(μ0 Ms)  ⇒ Keff ~ 0.5 Ms Bc
    # donde Bc = μ0 Hc (aquí Hc se expresa en mT → Tesla).
    Ms_Am = float(Ms_MA_m) * 1e6
    Bc_T = max(float(Hc_ref_mT), 0.0) * 1e-3
    Keff_ref = 0.5 * Ms_Am * Bc_T  # J/m^3

    # Escalado térmico simple: Keff(T) ~ Keff_ref · (Ms(T)/Ms(0))^2
    Keff_T = Keff_ref * (ms_r**2)

    barrier = (Keff_T * V) / (_kB * T)

    # Factor superparamagnético: pasa suavemente 0→1 alrededor de Δ≈barrier_crit
    spm = _logistic((barrier - cfg.barrier_crit) / max(cfg.barrier_width, 1e-6))

    # Hc cae con Ms^2 y con activación térmica (spm)
    Hc_T_mT = float(max(Hc_ref_mT, 0.0) * (ms_r**2) * spm)

    # Mr/Ms también cae con T (más suave que Hc) y con spm
    Mr_T = float(np.clip(max(Mr_ref, 0.0) * (ms_r**0.5) * spm, 0.0, 1.10))

    return Hc_T_mT, Mr_T, float(barrier)
