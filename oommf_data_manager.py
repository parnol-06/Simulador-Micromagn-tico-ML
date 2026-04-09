"""
oommf_data_manager.py
=====================
Gestor dinámico de datos OOMMF/Ubermag.

Detecta, clasifica y parsea automáticamente cualquier archivo .txt (fd/mg)
o .ipynb de simulación micromagnética que se deposite en la carpeta `oommf_data/`.

Flujo:
  1. scan_datasets()         → lista todos los archivos disponibles
  2. classify_series()       → identifica el tipo físico de cada serie
  3. extract_hyst_params()   → extrae Hc, Mr de ciclos de histéresis
  4. parse_ipynb_params()    → extrae Ms, K, A, geometría de notebooks
  5. get_training_points()   → devuelve puntos listos para calibrar el ML

Convención de archivos .txt:
  - Primera línea: encabezado (cualquier texto)
  - Columnas: fd  mg  (tab-separado o espacio)
    • fd  = campo aplicado en mT (puede ser positivo o negativo)
    • mg  = cantidad medida (M/Ms adim., energía en J, etc.)

Clasificación automática por rangos:
  Type          | mg range típico           | sign
  ------------- | ------------------------- | -----
  hysteresis    | [-1, 1]                   | ±
  zeeman        | [~1e-8, ~1e-6]            | +
  dipolar       | [~1e-18, ~1e-16]          | +
  exchange      | [~1e-20, ~1e-16]          | +
  anisotropy    | [~-1e11, ~-1e7]           | −
  topological   | [~-1e-9, ~1e-9]           | ±
  unknown       | (todo lo demás)           |
"""

from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Any

import numpy as np

# ── Carpeta de datos por defecto ─────────────────────────────────────────────
_DEFAULT_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 'oommf_data')

# ── Palabras clave para clasificación por nombre de archivo ──────────────────
_NAME_HINTS: list[tuple[list[str], str]] = [
    (['histeresis', 'ciclo', 'loop', 'hysteresis', 'mg'],     'hysteresis'),
    (['zeeman', 'ez'],                                          'zeeman'),
    (['dipolar', 'demag', 'ed'],                                'dipolar'),
    (['intercambio', 'exchange', 'ex'],                         'exchange'),
    (['anisotrop', 'ek'],                                       'anisotropy'),
    (['topol', 'tp'],                                           'topological'),
]

# ── Rangos físicos para clasificación automática por valores ─────────────────
def classify_series(fd: np.ndarray, mg: np.ndarray,
                    filename: str = '') -> str:
    """
    Clasifica una serie fd/mg en su tipo físico.

    Estrategia:
      1. Pistas del nombre de archivo (más fiable)
      2. Rangos de magnitud de mg (respaldo)
    """
    # 1. Por nombre de archivo
    name_lower = filename.lower().replace('_', ' ')
    for keywords, dtype in _NAME_HINTS:
        if any(kw in name_lower for kw in keywords):
            return dtype

    # 2. Por magnitud/signo
    mg_abs = np.abs(mg)
    mg_max = float(mg_abs.max())
    mg_min = float(mg.min())

    if mg_max <= 1.05 and mg_min >= -1.05:
        return 'hysteresis'
    if mg_min >= 0 and 1e-8 <= mg_max <= 5e-6:
        return 'zeeman'
    if mg_min >= 0 and 1e-19 <= mg_max <= 5e-16:
        if mg_max < 5e-17:
            return 'exchange'
        return 'dipolar'
    if mg_max < 5e-17:
        return 'exchange'
    if mg_min < 0 and mg_max > 1e6:
        return 'anisotropy'
    if mg_abs.max() < 1e-8 and mg_min < 0:
        return 'topological'

    return 'unknown'


# ── Parser de archivos fd/mg ─────────────────────────────────────────────────
def parse_fdmg_file(filepath: str | Path) -> dict:
    """
    Carga un archivo fd/mg genérico.

    Returns
    -------
    dict con:
      fd, mg        : arrays numpy completos
      fd_desc, mg_desc : rama descendente (primera mitad)
      fd_asc,  mg_asc  : rama ascendente  (segunda mitad)
      dtype         : tipo físico ('hysteresis', 'zeeman', ...)
      filename      : nombre base del archivo
      filepath      : ruta completa
      label         : etiqueta legible
    """
    filepath = Path(filepath)
    fd_list, mg_list = [], []
    with open(filepath, encoding='utf-8', errors='replace') as f:
        for i, line in enumerate(f):
            if i == 0:
                continue
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 2:
                try:
                    fd_list.append(float(parts[0]))
                    mg_list.append(float(parts[1]))
                except ValueError:
                    pass

    if not fd_list:
        return {}

    fd = np.array(fd_list)
    mg = np.array(mg_list)
    n  = len(fd) // 2

    dtype = classify_series(fd, mg, filepath.name)
    label = _dtype_label(dtype, filepath.name)

    return {
        'fd':       fd,
        'mg':       mg,
        'fd_desc':  fd[:n],  'mg_desc': mg[:n],
        'fd_asc':   fd[n:],  'mg_asc':  mg[n:],
        'dtype':    dtype,
        'filename': filepath.name,
        'filepath': str(filepath),
        'label':    label,
        'n_points': len(fd),
    }


def _dtype_label(dtype: str, filename: str) -> str:
    _map = {
        'hysteresis':  'Ciclo de Histéresis (M/Ms)',
        'zeeman':      'Energía Zeeman (J)',
        'dipolar':     'Energía Dipolar (J)',
        'exchange':    'Energía de Intercambio (J)',
        'anisotropy':  'Energía de Anisotropía (J)',
        'topological': 'Carga Topológica',
        'unknown':     f'Serie — {filename}',
    }
    return _map.get(dtype, f'Serie — {filename}')


# ── Extracción de parámetros de histéresis ───────────────────────────────────
def extract_hyst_params(data: dict) -> dict:
    """
    Extrae Hc, Mr, H_max de un dataset de histéresis.

    Parameters
    ----------
    data : dict devuelto por parse_fdmg_file() con dtype='hysteresis'

    Returns
    -------
    dict con Hc_mT, Hc_desc_mT, Hc_asc_mT, Mr_Ms, H_max_mT
    """
    fd_d = np.asarray(data.get('fd_desc', []), dtype=float)
    mg_d = np.asarray(data.get('mg_desc', []), dtype=float)
    fd_a = np.asarray(data.get('fd_asc',  []), dtype=float)
    mg_a = np.asarray(data.get('mg_asc',  []), dtype=float)

    # Datos mínimos requeridos
    if len(fd_d) < 2 or len(fd_a) < 2:
        return {'Hc_mT': 0.0, 'Hc_desc_mT': 0.0,
                'Hc_asc_mT': 0.0, 'Mr_Ms': 0.0, 'H_max_mT': 0.0}

    fd_all = np.asarray(data.get('fd', []), dtype=float)
    H_max  = float(np.abs(fd_all).max()) if len(fd_all) else 0.0

    # Hc: interpolación lineal en el cruce por cero
    def _hc(fd_arr, mg_arr, sign=1) -> float:
        idx = np.where(mg_arr * sign < 0)[0]
        if len(idx) == 0:
            return float(np.abs(fd_arr[np.argmin(np.abs(mg_arr))]))
        i0 = idx[0]
        if i0 == 0:
            return float(np.abs(fd_arr[0]))
        dfd = fd_arr[i0] - fd_arr[i0 - 1]
        dmg = mg_arr[i0] - mg_arr[i0 - 1]
        if abs(dmg) < 1e-30:
            return float(np.abs(fd_arr[i0]))
        fd_zero = fd_arr[i0 - 1] - mg_arr[i0 - 1] * dfd / dmg
        return float(abs(fd_zero))

    Hc_desc = _hc(fd_d, mg_d, sign=1)   # descendente: busca mg→negativo
    Hc_asc  = _hc(fd_a, mg_a, sign=-1)  # ascendente:  busca mg→positivo
    Hc_avg  = round((Hc_desc + Hc_asc) / 2.0, 2)

    # Mr: valor de mg en fd≈0
    idx0_d = int(np.argmin(np.abs(fd_d)))
    idx0_a = int(np.argmin(np.abs(fd_a)))
    Mr = round(float((abs(mg_d[idx0_d]) + abs(mg_a[idx0_a])) / 2.0), 4)

    return {
        'Hc_mT':      Hc_avg,
        'Hc_desc_mT': round(Hc_desc, 2),
        'Hc_asc_mT':  round(Hc_asc, 2),
        'Mr_Ms':      Mr,
        'H_max_mT':   round(H_max, 1),
    }


# ── Extracción de parámetros desde .ipynb ───────────────────────────────────
def parse_ipynb_params(nb_path: str | Path) -> dict:
    """
    Extrae parámetros de simulación de un notebook Jupyter (.ipynb).

    Busca patrones como:
      Ms = 1.7e6, K = 48e3, A = 2.1e-11, radius = 21e-9, cell = 3e-9

    Returns
    -------
    dict con claves opcionales: Ms_Am, K1_Jm3, A_Jm, radius_nm, cell_nm,
      separation_nm, box_nm, material_guess
    """
    nb_path = Path(nb_path)
    if not nb_path.exists():
        return {}

    try:
        with open(nb_path, encoding='utf-8', errors='replace') as f:
            nb = json.load(f)
    except Exception:
        return {}

    # Concatenar todo el código del notebook
    code = '\n'.join(
        ''.join(cell.get('source', []))
        for cell in nb.get('cells', [])
        if cell.get('cell_type') == 'code'
    )

    params: dict[str, Any] = {}

    def _find(pattern: str, text: str) -> float | None:
        m = re.search(pattern, text)
        if m:
            try:
                return float(m.group(1).replace(' ', ''))
            except Exception:
                pass
        return None

    # Ms
    ms = _find(r'Ms\s*=\s*([\d.e+\-]+)', code)
    if ms:
        params['Ms_Am'] = ms

    # K / K1 / anisotropy constant
    k = _find(r'\bK\b\s*=\s*([\d.e+\-]+)', code) or \
        _find(r'\bK1\b\s*=\s*([\d.e+\-]+)', code) or \
        _find(r'\bKu\b\s*=\s*([\d.e+\-]+)', code)
    if k:
        params['K1_Jm3'] = k

    # A (exchange)
    a = _find(r'\bA\b\s*=\s*([\d.e+\-]+)', code)
    if a:
        params['A_Jm'] = a

    # radius
    r = _find(r'radius\s*=\s*([\d.e+\-]+)', code) or \
        _find(r'radius_squared\s*=\s*([\d.e+\-]+)', code)
    if r:
        params['radius_nm'] = r * 1e9 if r < 1 else r   # m → nm

    # cell size
    c = _find(r'\bc\b\s*=\s*([\d.e+\-]+)', code) or \
        _find(r'cell\s*=\s*([\d.e+\-]+)', code)
    if c:
        params['cell_nm'] = c * 1e9 if c < 1 else c

    # E_0 / separation
    e0 = _find(r'E_0\s*=\s*([\d.e+\-]+)', code)
    if e0 and r:
        params['E0_nm'] = e0 * 1e9 if e0 < 1 else e0
        sep = 2 * (params.get('E0_nm', 0) - params.get('radius_nm', 0))
        if sep > 0:
            params['separation_nm'] = round(sep, 2)

    # Box dimensions
    lx = _find(r'Lx\s*=\s*([\d.e+\-]+)', code)
    ly = _find(r'Ly\s*=\s*([\d.e+\-]+)', code)
    lz = _find(r'Lz\s*=\s*([\d.e+\-]+)', code)
    if lx and ly and lz:
        params['box_nm'] = (
            round(lx * 1e9 if lx < 1 else lx, 1),
            round(ly * 1e9 if ly < 1 else ly, 1),
            round(lz * 1e9 if lz < 1 else lz, 1),
        )

    # Material guess by Ms value
    if 'Ms_Am' in params:
        params['material_guess'] = _guess_material(params['Ms_Am'])

    # Runner
    if 'ExeOOMMFRunner' in code:
        params['runner'] = 'ExeOOMMFRunner'
    elif 'DockerOOMMFRunner' in code:
        params['runner'] = 'DockerOOMMFRunner'
    else:
        params['runner'] = 'unknown'

    params['source_nb'] = nb_path.name
    return params


def _guess_material(Ms_Am: float) -> str:
    """Adivina el material por el valor de Ms."""
    # Ms en A/m → nombre aproximado
    refs = {
        'fe':        1.707e6,
        'co':        1.440e6,
        'ni':        0.490e6,
        'permalloy': 0.860e6,
        'magnetite': 0.480e6,
        'smco5':     0.860e6,
        'fept':      1.000e6,
        'bafeo':     0.380e6,
    }
    best = min(refs, key=lambda k: abs(refs[k] - Ms_Am))
    if abs(refs[best] - Ms_Am) / Ms_Am < 0.15:
        return best
    return 'unknown'


# ── Escaneo dinámico del directorio ─────────────────────────────────────────
def scan_datasets(data_dir: str | None = None) -> dict:
    """
    Escanea `data_dir` y devuelve un diccionario con todos los datasets.

    Returns
    -------
    dict:
      'hysteresis'  : list of dict  (cada dict = parse_fdmg_file + extract_hyst_params)
      'energies'    : list of dict  (Zeeman, dipolar, exchange, anisotropy, ...)
      'notebooks'   : list of dict  (parse_ipynb_params por cada .ipynb)
      'all_files'   : list of str   (paths de todos los archivos detectados)
      'data_dir'    : str
    """
    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR

    data_dir = Path(data_dir)
    result: dict[str, Any] = {
        'hysteresis': [],
        'energies':   [],
        'notebooks':  [],
        'all_files':  [],
        'data_dir':   str(data_dir),
    }

    if not data_dir.exists():
        return result

    # Archivos .txt — cada archivo en su propio try para aislar errores
    for fpath in sorted(data_dir.glob('*.txt')):
        result['all_files'].append(str(fpath))
        try:
            ds = parse_fdmg_file(fpath)
            if not ds:
                continue
            if ds['dtype'] == 'hysteresis':
                hp = extract_hyst_params(ds)
                ds.update(hp)
                result['hysteresis'].append(ds)
            elif ds['dtype'] != 'unknown':
                result['energies'].append(ds)
        except Exception:
            # Archivo malformado: lo registramos pero no detenemos el escaneo
            continue

    # Archivos .ipynb — ídem
    for fpath in sorted(data_dir.glob('*.ipynb')):
        result['all_files'].append(str(fpath))
        try:
            nb_params = parse_ipynb_params(fpath)
            if nb_params:
                result['notebooks'].append(nb_params)
        except Exception:
            continue

    return result


# ── Extracción de puntos de entrenamiento para el ML ────────────────────────
def get_training_points(data_dir: str | None = None) -> list[dict]:
    """
    Extrae puntos de entrenamiento de todos los datasets disponibles.

    Cada punto es un dict:
      d_nm       : diámetro equivalente (nm)  — si disponible del notebook
      mat_id     : id del material             — estimado por Ms
      Hc_mT      : campo coercitivo (mT)
      Mr_Ms      : remanencia normalizada
      H_max_mT   : campo máximo (mT)
      source     : nombre del archivo fuente
      Ms_Am      : Ms del material (si disponible del .ipynb)
      K1_Jm3     : K1 (si disponible)
      A_Jm       : A  (si disponible)

    Returns
    -------
    list[dict]
    """
    datasets = scan_datasets(data_dir)
    notebooks = {nb['source_nb']: nb for nb in datasets['notebooks']}

    # Buscar notebook más reciente con parámetros
    nb_params = next(iter(notebooks.values()), {}) if notebooks else {}

    points = []
    for hds in datasets['hysteresis']:
        pt: dict[str, Any] = {
            'Hc_mT':   hds['Hc_mT'],
            'Mr_Ms':   hds['Mr_Ms'],
            'H_max_mT': hds['H_max_mT'],
            'source':  hds['filename'],
        }
        # Enriquecer con datos del notebook si disponibles
        if nb_params:
            pt['Ms_Am']    = nb_params.get('Ms_Am')
            pt['K1_Jm3']   = nb_params.get('K1_Jm3')
            pt['A_Jm']     = nb_params.get('A_Jm')
            pt['mat_id']   = nb_params.get('material_guess', 'unknown')
            r_nm           = nb_params.get('radius_nm')
            pt['d_nm']     = r_nm * 2 if r_nm else None
        points.append(pt)

    return points


# ── Calibración incremental: leer/escribir base de datos de puntos reales ───
_CALIBRATION_FILE = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'oommf_data', 'calibration_db.json'
)


def load_calibration_db() -> list[dict]:
    """Carga la base de datos de calibración (puntos OOMMF reales)."""
    if not os.path.exists(_CALIBRATION_FILE):
        return []
    try:
        with open(_CALIBRATION_FILE, encoding='utf-8') as f:
            return json.load(f)
    except Exception:
        return []


def save_calibration_point(
        mat_id: str,
        d_nm: float,
        Hc_mT: float,
        Mr_Ms: float,
        geom_id: str = 'sphere',
        source: str = 'manual',
        extra: dict | None = None,
) -> None:
    """
    Añade un punto de calibración a la base de datos persistente.

    Cada llamada agrega un registro; registros duplicados se actualizan.
    """
    db = load_calibration_db()

    # Buscar registro existente para (mat_id, d_nm, geom_id)
    existing = next(
        (p for p in db
         if p.get('mat_id') == mat_id
         and abs(p.get('d_nm', -1) - d_nm) < 0.5
         and p.get('geom_id') == geom_id),
        None,
    )
    record = {
        'mat_id':  mat_id,
        'd_nm':    round(float(d_nm), 2),
        'geom_id': geom_id,
        'Hc_mT':   round(float(Hc_mT), 3),
        'Mr_Ms':   round(float(Mr_Ms), 5),
        'source':  source,
        **(extra or {}),
    }
    if existing is not None:
        existing.update(record)
    else:
        db.append(record)

    os.makedirs(os.path.dirname(_CALIBRATION_FILE), exist_ok=True)
    with open(_CALIBRATION_FILE, 'w', encoding='utf-8') as f:
        json.dump(db, f, indent=2, ensure_ascii=False)


def calibration_correction(
        mat_id: str,
        d_nm: float,
        geom_id: str,
        Hc_pred: float,
        Mr_pred: float,
        sigma_nm: float = 5.0,
) -> tuple[float, float]:
    """
    Aplica corrección Gaussiana de los datos reales al valor ML predicho.

    Si hay puntos reales cercanos en el espacio (mat_id, d_nm), mezcla
    la predicción ML con los valores reales ponderados por distancia.

    Parameters
    ----------
    sigma_nm : ancho gaussiano de interpolación en nm

    Returns
    -------
    (Hc_corr, Mr_corr) : valores corregidos
    """
    db = load_calibration_db()
    relevant = [
        p for p in db
        if p.get('mat_id') == mat_id and p.get('geom_id') == geom_id
    ]
    if not relevant:
        return Hc_pred, Mr_pred

    # Pesos gaussianos por distancia en d_nm
    weights, Hc_real_vals, Mr_real_vals = [], [], []
    for p in relevant:
        dist = abs(p['d_nm'] - d_nm)
        w = float(np.exp(-0.5 * (dist / sigma_nm) ** 2))
        if w > 1e-4:
            weights.append(w)
            Hc_real_vals.append(p['Hc_mT'])
            Mr_real_vals.append(p['Mr_Ms'])

    if not weights:
        return Hc_pred, Mr_pred

    w_tot = sum(weights)
    # Mezcla: más peso a real cuanto más cerca
    alpha = min(w_tot / (w_tot + 1.0), 0.90)   # máx. 90 % peso real

    Hc_real_avg = sum(w * v for w, v in zip(weights, Hc_real_vals)) / w_tot
    Mr_real_avg = sum(w * v for w, v in zip(weights, Mr_real_vals)) / w_tot

    Hc_corr = alpha * Hc_real_avg + (1 - alpha) * Hc_pred
    Mr_corr = alpha * Mr_real_avg + (1 - alpha) * Mr_pred
    return round(Hc_corr, 3), round(Mr_corr, 5)


# ── Incorporar nuevos archivos al dataset ────────────────────────────────────
def ingest_uploaded_file(
        src_path: str,
        data_dir: str | None = None,
        mat_id: str | None = None,
        d_nm: float | None = None,
        geom_id: str = 'sphere',
) -> dict:
    """
    Copia un archivo al directorio de datos, lo parsea y, si es histéresis,
    guarda el punto de calibración en la base de datos.

    Parameters
    ----------
    src_path : ruta al archivo origen (ej. upload temporal de Streamlit)
    data_dir : destino (por defecto oommf_data/)
    mat_id   : id del material (ej. 'fe'); si None se intenta inferir
    d_nm     : diámetro de la partícula (nm); si None se intenta inferir
    geom_id  : geometría ('sphere', 'cuboid', ...)

    Returns
    -------
    dict con:
      'status'    : 'ok' | 'error'
      'dtype'     : tipo físico detectado
      'dataset'   : dict del dataset parseado
      'hyst_params': parámetros de histéresis (si aplica)
      'message'   : descripción del resultado
    """
    import shutil

    if data_dir is None:
        data_dir = _DEFAULT_DATA_DIR

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    src = Path(src_path)
    dst = data_dir / src.name

    try:
        shutil.copy2(str(src), str(dst))
    except Exception as e:
        return {'status': 'error', 'message': str(e)}

    ds = parse_fdmg_file(dst)
    if not ds:
        return {'status': 'error', 'message': 'No se pudieron leer datos fd/mg'}

    result: dict[str, Any] = {
        'status':  'ok',
        'dtype':   ds['dtype'],
        'dataset': ds,
        'message': f'Archivo cargado: {dst.name}  ({ds["n_points"]} puntos, tipo: {ds["dtype"]})',
    }

    # Si es histéresis, extraer Hc/Mr y guardar calibración
    if ds['dtype'] == 'hysteresis':
        hp = extract_hyst_params(ds)
        ds.update(hp)
        result['hyst_params'] = hp

        # Guardar calibración si tenemos mat_id y d_nm
        if mat_id and d_nm:
            save_calibration_point(
                mat_id=mat_id,
                d_nm=float(d_nm),
                Hc_mT=hp['Hc_mT'],
                Mr_Ms=hp['Mr_Ms'],
                geom_id=geom_id,
                source=src.name,
            )
            result['calibration_saved'] = True
            result['message'] += (
                f'  →  Calibración guardada: Hc={hp["Hc_mT"]:.1f} mT, '
                f'Mr={hp["Mr_Ms"]:.4f}'
            )

    return result


# ── Utilidad: resumen del estado del dataset ─────────────────────────────────
def dataset_summary(data_dir: str | None = None) -> dict:
    """
    Devuelve un resumen compacto del estado del directorio de datos.

    Returns
    -------
    dict:
      n_hysteresis : número de ciclos de histéresis
      n_energies   : número de series de energía
      n_notebooks  : número de notebooks
      n_calibration: número de puntos de calibración guardados
      materials    : materiales inferidos
      hc_range     : (Hc_min, Hc_max) en mT
      mr_range     : (Mr_min, Mr_max)
    """
    ds = scan_datasets(data_dir)
    cal = load_calibration_db()

    hc_vals = [h['Hc_mT'] for h in ds['hysteresis'] if 'Hc_mT' in h]
    mr_vals = [h['Mr_Ms'] for h in ds['hysteresis'] if 'Mr_Ms' in h]
    mats    = list({
        nb.get('material_guess', 'unknown')
        for nb in ds['notebooks']
        if nb.get('material_guess')
    })

    return {
        'n_hysteresis':  len(ds['hysteresis']),
        'n_energies':    len(ds['energies']),
        'n_notebooks':   len(ds['notebooks']),
        'n_calibration': len(cal),
        'materials':     mats,
        'hc_range':      (min(hc_vals, default=0), max(hc_vals, default=0)),
        'mr_range':      (min(mr_vals, default=0), max(mr_vals, default=0)),
        'data_dir':      ds['data_dir'],
    }
