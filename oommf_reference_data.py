"""
oommf_reference_data.py
=======================
Acceso a datos de referencia OOMMF — wrapper dinámico sobre oommf_data_manager.

Este módulo expone la misma API que en versiones anteriores (load_hysteresis,
load_energies, extract_hc_mr, REFERENCE_PARAMS, NOTEBOOK_CODE) pero ahora
delega en `oommf_data_manager` para detectar y cargar automáticamente
cualquier nuevo archivo que se deposite en `oommf_data/`.

Compatibilidad retroactiva: el API público no cambia.
"""
from __future__ import annotations

import os
import numpy as np

import oommf_data_manager as _dm

# ── Ruta de datos ─────────────────────────────────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'oommf_data')


def data_available() -> bool:
    """True si hay al menos un archivo de datos en oommf_data/."""
    ds = _dm.scan_datasets(_DATA_DIR)
    return len(ds['hysteresis']) > 0 or len(ds['energies']) > 0


def load_hysteresis(idx: int = 0) -> dict:
    """
    Devuelve el ciclo de histéresis #idx (por orden de nombre).

    Incluye claves:
      fd, mg, fd_desc, mg_desc, fd_asc, mg_asc, Hc_mT, Mr_Ms, H_max_mT
    """
    ds = _dm.scan_datasets(_DATA_DIR)
    if not ds['hysteresis']:
        raise FileNotFoundError('No se encontraron datos de histéresis en oommf_data/')
    h = ds['hysteresis'][idx % len(ds['hysteresis'])]
    return h


def load_energies() -> dict:
    """
    Devuelve todas las series de energía detectadas como dict{tipo: dataset}.

    Claves posibles: 'zeeman', 'dipolar', 'exchange', 'anisotropy', ...
    Cada valor es un dict con fd, mg, fd_desc, mg_desc, fd_asc, mg_asc, label.
    """
    ds = _dm.scan_datasets(_DATA_DIR)
    result = {}
    for e in ds['energies']:
        dtype = e['dtype']
        result[dtype] = e
    return result


def load_all_hysteresis() -> list[dict]:
    """Devuelve todos los ciclos de histéresis detectados."""
    return _dm.scan_datasets(_DATA_DIR)['hysteresis']


def extract_hc_mr(idx: int = 0) -> tuple[float, float]:
    """Devuelve (Hc_mT, Mr_Ms) del ciclo de histéresis #idx."""
    h = load_hysteresis(idx)
    return h['Hc_mT'], h['Mr_Ms']


def get_training_points() -> list[dict]:
    """Puntos de entrenamiento extraídos de todos los datasets disponibles."""
    return _dm.get_training_points(_DATA_DIR)


def dataset_summary() -> dict:
    """Resumen del estado del directorio de datos."""
    return _dm.dataset_summary(_DATA_DIR)


def ingest_file(src_path: str, mat_id: str | None = None,
                d_nm: float | None = None, geom_id: str = 'sphere') -> dict:
    """Incorpora un nuevo archivo al dataset y actualiza la calibración."""
    return _dm.ingest_uploaded_file(src_path, _DATA_DIR, mat_id, d_nm, geom_id)


# ── Parámetros de referencia (primer notebook detectado) ─────────────────────
def _build_reference_params() -> dict:
    """Construye REFERENCE_PARAMS dinámicamente desde el notebook disponible."""
    ds   = _dm.scan_datasets(_DATA_DIR)
    hds  = ds['hysteresis'][0] if ds['hysteresis'] else {}
    nb   = ds['notebooks'][0]  if ds['notebooks']  else {}

    return {
        'material':     f"{nb.get('material_guess','?')} ({nb.get('source_nb','?')})",
        'Ms_Am':        nb.get('Ms_Am',   1.70e6),
        'K1_Jm3':       nb.get('K1_Jm3',  48e3),
        'A_Jm':         nb.get('A_Jm',    2.1e-11),
        'radius_nm':    nb.get('radius_nm', 21.0),
        'separation_nm': nb.get('separation_nm', 6.0),
        'E0_nm':        nb.get('E0_nm', 24.0),
        'box_nm':       nb.get('box_nm', (114, 42, 42)),
        'cell_nm':      nb.get('cell_nm', 3.0),
        'H_max_mT':     hds.get('H_max_mT', 400.0),
        'n_points':     hds.get('n_points', 160),
        'Hc_mT':        hds.get('Hc_mT', 69.2),
        'Hc_desc_mT':   hds.get('Hc_desc_mT', 71.9),
        'Hc_asc_mT':    hds.get('Hc_asc_mT', 66.5),
        'Mr_Ms':        hds.get('Mr_Ms', 0.3701),
        'runner':       nb.get('runner', 'ExeOOMMFRunner'),
        'source_nb':    nb.get('source_nb', '12nm.ipynb'),
    }


try:
    REFERENCE_PARAMS: dict = _build_reference_params()
except Exception:
    REFERENCE_PARAMS = {
        'material': 'Fe', 'Ms_Am': 1.70e6, 'K1_Jm3': 48e3,
        'A_Jm': 2.1e-11, 'radius_nm': 21.0, 'separation_nm': 6.0,
        'E0_nm': 24.0, 'box_nm': (114, 42, 42), 'cell_nm': 3.0,
        'H_max_mT': 400.0, 'n_points': 160, 'Hc_mT': 69.2,
        'Hc_desc_mT': 71.9, 'Hc_asc_mT': 66.5, 'Mr_Ms': 0.3701,
        'runner': 'ExeOOMMFRunner', 'source_nb': '12nm.ipynb',
    }


# ── Código Ubermag reproducible (generado dinámicamente) ─────────────────────
def _build_notebook_code() -> str:
    from ubermag_validator import generate_two_sphere_script
    rp = REFERENCE_PARAMS
    try:
        return generate_two_sphere_script(
            radius_nm = rp.get('radius_nm', 21.0),
            sep_nm    = rp.get('separation_nm', 6.0),
            Ms_Am     = rp.get('Ms_Am', 1.70e6),
            K1_Jm3   = rp.get('K1_Jm3', 48e3),
            A_Jm     = rp.get('A_Jm', 2.1e-11),
            H_max_mT  = rp.get('H_max_mT', 400.0),
            cell_nm   = rp.get('cell_nm', 3.0),
        )
    except Exception:
        return '# Código no disponible — ubermag_validator no cargado'


try:
    NOTEBOOK_CODE: str = _build_notebook_code()
except Exception:
    NOTEBOOK_CODE = '# Código no disponible'
