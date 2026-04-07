"""
=============================================================================
 SISTEMA DE SIMULACIÓN MICROMAGNÉTICA CON MACHINE LEARNING — v2.0
 Fase 1: Motor Dinámico Multi-Material con CLI

 Basado en datos experimentales de:
   - Fe nanoparticles (Results in Physics 2025)      [Galvis, Mesa, et al.]
   - Permalloy nanodots (Comp. Mat. Sci. 2024)        [Galvis, Mesa, Restrepo]

 Uso básico:
   python micromagnetic_simulator_v2.py
   python micromagnetic_simulator_v2.py --material fe --sizes 16 30 44 60 --pred 30
   python micromagnetic_simulator_v2.py --material permalloy --sizes 20 40 80 --pred 40
   python micromagnetic_simulator_v2.py --sizes 10 25 50 75 --pred 42 --dpi 300 --format svg
   python micromagnetic_simulator_v2.py --help

 Requisitos (ver requirements.txt):
   numpy >= 1.24  |  matplotlib >= 3.7  |  scikit-learn >= 1.3
=============================================================================
"""

import argparse
import os
import sys
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')                          # RF-13 (RNF-09): sin display necesario
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

# =============================================================================
#  PALETA DE COLORES  (RF-17, RF-18)
#  Tema oscuro tipo editor de código — coherente con Prueba_Final_Profe_v1.py
# =============================================================================

COLORS = {
    'bg':      '#0d1117',
    'panel':   '#161b22',
    'border':  '#30363d',
    'accent1': '#58a6ff',   # azul    — esfera Fe / curva principal
    'accent2': '#f78166',   # rojo    — cuboide Fe / energía Exchange
    'accent3': '#3fb950',   # verde   — energía Desmagnetización
    'accent4': '#d2a8ff',   # púrpura — energía Anisotropía
    'accent5': '#ffa657',   # naranja — material 3 (Co)
    'accent6': '#79c0ff',   # azul claro — material 4 (Fe3O4)
    'text':    '#e6edf3',
    'subtext': '#8b949e',
    'warn':    '#f0883e',   # naranja suave para avisos de extrapolación
}

# Ciclo de colores para curvas múltiples
COLOR_CYCLE = [
    COLORS['accent1'], COLORS['accent2'], COLORS['accent3'],
    COLORS['accent4'], COLORS['accent5'], COLORS['accent6'],
    '#ff7b72', '#56d364',
]

plt.rcParams.update({
    'figure.facecolor': COLORS['bg'],
    'axes.facecolor':   COLORS['panel'],
    'axes.edgecolor':   COLORS['border'],
    'axes.labelcolor':  COLORS['text'],
    'xtick.color':      COLORS['subtext'],
    'ytick.color':      COLORS['subtext'],
    'text.color':       COLORS['text'],
    'grid.color':       COLORS['border'],
    'grid.alpha':       0.5,
    'font.family':      'monospace',
    'font.size':        9,
    'legend.facecolor': '#1c2433',
    'legend.edgecolor': COLORS['border'],
})

# =============================================================================
#  BASE DE DATOS DE MATERIALES — MaterialsDB  (RF-20)
#
#  Cada entrada contiene:
#    'name'     : Nombre completo del material
#    'formula'  : Fórmula química
#    'sphere'   : np.array (N,3) → [diámetro_nm, Hc_mT, Mr/Ms]  (None si no hay datos)
#    'cuboid'   : np.array (N,3) → [arista_nm,  Hc_mT, Mr/Ms]   (None si no hay datos)
#    'params'   : Parámetros físicos del material
#    'range'    : [D_min, D_max] rango de datos de entrenamiento
#    'field_max': Campo máximo H recomendado para el lazo (mT)
#    'ref'      : Referencia bibliográfica
# =============================================================================

MATERIALS_DB = {

    # ── Hierro (Fe)  ──────────────────────────────────────────────────────────
    # Fuente: Galvis et al., Results in Physics 77 (2025) 108460
    'fe': {
        'name':      'Hierro',
        'formula':   'Fe',
        'sphere': np.array([
            [16, 210, 0.88],
            [20, 250, 0.86],
            [24, 270, 0.82],
            [30, 290, 0.78],
            [36, 280, 0.70],
            [44, 230, 0.55],
            [52, 165, 0.42],
            [60, 110, 0.32],
        ]),
        'cuboid': np.array([
            [16, 320, 0.91],
            [20, 370, 0.90],
            [24, 400, 0.88],
            [30, 420, 0.85],
            [36, 410, 0.80],
            [44, 370, 0.72],
            [52, 300, 0.62],
            [60, 230, 0.52],
        ]),
        'params': {
            'K1_kJ_m3':    48.0,
            'aniso_type':  'Cubica',
            'A_pJ_m':      21.0,
            'alpha':        1.0,
            'Ms_MA_m':      1.70,
            'lambda_ex_nm': 3.40,
            'Tc_K':         1043,
        },
        'range':     [16, 60],
        'field_max': 600,
        'ref':       'Galvis et al., Results in Physics 77 (2025) 108460',
    },

    # ── Permalloy (Ni80Fe20)  ─────────────────────────────────────────────────
    # Fuente: Galvis et al., Comp. Materials Science 245 (2024) 113330
    # Dataset sintético calibrado con parámetros físicos publicados (Fig. 1-3)
    # D = 20–120 nm, t = 20 nm (espesor fijo para Fase 1)
    'permalloy': {
        'name':    'Permalloy',
        'formula': 'Ni80Fe20',
        'sphere': np.array([
            [20,  45, 0.92],
            [30,  55, 0.90],
            [40,  65, 0.87],
            [60,  80, 0.82],
            [80,  90, 0.75],
            [100, 85, 0.65],
            [120, 70, 0.52],
        ]),
        'cuboid': np.array([
            [20,  60, 0.93],
            [30,  75, 0.91],
            [40,  88, 0.88],
            [60, 100, 0.84],
            [80, 105, 0.78],
            [100, 95, 0.68],
            [120, 78, 0.55],
        ]),
        'params': {
            'K1_kJ_m3':    0.15,
            'aniso_type':  'Cubica',
            'A_pJ_m':      11.0,
            'alpha':        1.0,
            'Ms_MA_m':      0.84,
            'lambda_ex_nm': 4.98,
            'Tc_K':         843,
        },
        'range':     [20, 120],
        'field_max': 300,
        'ref':       'Galvis et al., Comp. Materials Science 245 (2024) 113330',
    },

    # ── Cobalto (Co)  ─────────────────────────────────────────────────────────
    # Parámetros de literatura estándar (OOMMF material database)
    # Dataset sintético generado con modelo LLG y K1 alta
    'co': {
        'name':    'Cobalto',
        'formula': 'Co',
        'sphere': np.array([
            [5,   280, 0.85],
            [10,  520, 0.90],
            [15,  800, 0.92],
            [20,  950, 0.88],
            [30, 1050, 0.80],
            [40,  980, 0.70],
            [60,  750, 0.55],
            [80,  520, 0.42],
        ]),
        'cuboid': np.array([
            [5,   350, 0.87],
            [10,  620, 0.91],
            [15,  920, 0.93],
            [20, 1100, 0.90],
            [30, 1200, 0.84],
            [40, 1150, 0.75],
            [60,  900, 0.62],
            [80,  660, 0.50],
        ]),
        'params': {
            'K1_kJ_m3':    410.0,
            'aniso_type':  'Uniaxial',
            'A_pJ_m':      30.0,
            'alpha':        0.1,
            'Ms_MA_m':      1.42,
            'lambda_ex_nm': 4.70,
            'Tc_K':         1388,
        },
        'range':     [5, 80],
        'field_max': 2000,
        'ref':       'Literatura estándar — OOMMF material database',
    },

    # ── Magnetita (Fe3O4)  ────────────────────────────────────────────────────
    # Parámetros de literatura; relevante para aplicaciones biomédicas
    'fe3o4': {
        'name':    'Magnetita',
        'formula': 'Fe3O4',
        'sphere': np.array([
            [5,   12, 0.50],
            [8,   28, 0.65],
            [10,  45, 0.72],
            [15,  75, 0.78],
            [20,  90, 0.80],
            [30,  85, 0.72],
            [50,  60, 0.58],
            [80,  38, 0.45],
        ]),
        'cuboid': np.array([
            [5,   18, 0.55],
            [8,   38, 0.68],
            [10,  58, 0.75],
            [15,  95, 0.82],
            [20, 112, 0.84],
            [30, 105, 0.76],
            [50,  78, 0.62],
            [80,  50, 0.48],
        ]),
        'params': {
            'K1_kJ_m3':   -11.0,
            'aniso_type':  'Cubica',
            'A_pJ_m':      12.0,
            'alpha':        0.5,
            'Ms_MA_m':      0.48,
            'lambda_ex_nm': 6.20,
            'Tc_K':         858,
        },
        'range':     [5, 80],
        'field_max': 200,
        'ref':       'Literatura estándar — biomedical magnetic nanoparticles',
    },
}

# =============================================================================
#  CONSTRUCCIÓN GENÉRICA DE MODELOS GBR  (RF-04, RF-05, RF-23)
# =============================================================================

def build_model(data: np.ndarray, random_state: int = 0):
    """
    Construye y entrena un par de modelos GBR (Hc y Mr/Ms) para un dataset dado.

    Args:
        data        : np.array (N,3) con columnas [diámetro_nm, Hc_mT, Mr/Ms]
        random_state: semilla para reproducibilidad

    Returns:
        tuple (StandardScaler, GBR_Hc, GBR_Mr)
    """
    X    = data[:, 0].reshape(-1, 1)
    y_hc = data[:, 1]
    y_mr = data[:, 2]

    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)         # RF-06: normalización Z-score

    m_hc = GradientBoostingRegressor(        # RF-04: hiperparámetros fijos
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        random_state=random_state
    )
    m_mr = GradientBoostingRegressor(
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        random_state=random_state
    )
    m_hc.fit(Xs, y_hc)
    m_mr.fit(Xs, y_mr)

    return scaler, m_hc, m_mr


def predict(d_nm: float, model_tuple: tuple) -> tuple:
    """
    Predice (Hc_mT, Mr_Ms) para un diámetro dado usando el modelo entrenado.

    Args:
        d_nm        : diámetro de la nanopartícula en nm
        model_tuple : (StandardScaler, GBR_Hc, GBR_Mr)

    Returns:
        (Hc_mT: float, Mr_Ms: float)
    """
    scaler, m_hc, m_mr = model_tuple
    Xs = scaler.transform([[d_nm]])
    return float(m_hc.predict(Xs)[0]), float(m_mr.predict(Xs)[0])


def validate_range(d_nm: float, mat_id: str, geom: str) -> bool:
    """
    Verifica si d_nm está dentro del rango de entrenamiento del material.
    Imprime un aviso de extrapolación si está fuera. (RF-21)

    Returns:
        True si está dentro del rango, False si es extrapolación.
    """
    mat     = MATERIALS_DB[mat_id]
    D_min, D_max = mat['range']
    if d_nm < D_min or d_nm > D_max:
        print(
            f"  ⚠  Extrapolación: {d_nm:.1f} nm fuera del rango de entrenamiento "
            f"[{D_min}–{D_max} nm] para {mat['name']} ({geom}). "
            f"La predicción es extrapolada."
        )
        return False
    return True


def load_models(mat_id: str):
    """
    Carga y entrena los modelos GBR para el material seleccionado.
    Retorna dict con modelos disponibles: {'sphere': tuple, 'cuboid': tuple}
    """
    mat    = MATERIALS_DB[mat_id]
    models = {}

    if mat['sphere'] is not None:
        models['sphere'] = build_model(mat['sphere'], random_state=0)

    if mat['cuboid'] is not None:
        models['cuboid'] = build_model(mat['cuboid'], random_state=1)

    return models

# =============================================================================
#  SIMULACIÓN FÍSICA  (RF-09, RF-10)
# =============================================================================

def llg_hysteresis(
    Hc_mT: float,
    Mr: float,
    H_max: float  = 600,
    n_pts: int    = 500,
    noise_level: float = 0.008,
) -> tuple:
    """
    Genera el lazo de histéresis completo usando la aproximación analítica LLG.

    Args:
        Hc_mT       : Campo coercitivo en mT
        Mr          : Remanencia normalizada Mr/Ms
        H_max       : Campo máximo aplicado en mT  (se adapta al material)
        n_pts       : Puntos por rama (lazo total = 2 * n_pts)
        noise_level : Desviación estándar del ruido gaussiano (en unidades M/Ms)

    Returns:
        (H_full, M_full): arrays de forma (2 * n_pts,)
    """
    H = np.linspace(-H_max, H_max, n_pts)
    k = 4.0 / Hc_mT

    # Ramas de magnetización (RF-09)
    M_desc = Mr * np.tanh(k * (H - Hc_mT * 0.5)) + (1 - Mr) * H / H_max
    M_asc  = Mr * np.tanh(k * (H + Hc_mT * 0.5)) + (1 - Mr) * H / H_max

    noise  = np.random.normal(0, noise_level, n_pts)

    H_full = np.concatenate([H,       H[::-1]])
    M_full = np.concatenate([M_desc + noise, M_asc + noise[::-1]])

    # Limitar M al rango físico [-1.05, 1.05]
    M_full = np.clip(M_full, -1.05, 1.05)

    return H_full, M_full


def energy_landscape(Hc_mT: float, n_pts: int = 300) -> dict:
    """
    Calcula las cuatro contribuciones energéticas micromagnéticas. (RF-10)
    Los valores son energías normalizadas en unidades arbitrarias (u.a.)
    para comparación relativa.

    Args:
        Hc_mT : Campo coercitivo en mT (usado para escalar sigma de las gaussianas)
        n_pts : Número de puntos del eje H

    Returns:
        dict con claves: 'H', 'zeeman', 'exchange', 'demag', 'aniso'
    """
    H     = np.linspace(-600, 600, n_pts)
    sigma = Hc_mT * 0.8

    E_zee   = -0.5  * (H / 600) ** 2
    E_exc   = (0.15 * np.exp(-((H - Hc_mT) ** 2) / (2 * sigma ** 2)) +
               0.15 * np.exp(-((H + Hc_mT) ** 2) / (2 * sigma ** 2)))
    E_demag = 0.18  * np.exp(-H ** 2 / (2 * (Hc_mT * 1.2) ** 2))
    E_aniso = 0.05  * np.cos(np.pi * H / 400) ** 2

    return {'H': H, 'zeeman': E_zee, 'exchange': E_exc,
            'demag': E_demag, 'aniso': E_aniso}

# =============================================================================
#  FIGURA ADAPTATIVA  (RF-11 a RF-18, RF-22)
# =============================================================================

def build_figure(
    mat_id:    str,
    models:    dict,
    sizes:     list,
    pred:      float,
    noise:     float,
    dpi:       int,
    fmt:       str,
    output_dir: str,
):
    """
    Construye y exporta la figura GridSpec 2×2 adaptada al material y parámetros.

    Layout:
        [0,0] Histéresis comparativa — curvas para todos los tamaños en `sizes`
        [0,1] Paisaje de energías    — 4 contribuciones para el tamaño `pred`
        [1,:] Tabla resumen          — Hc y Mr/Ms para esfera y cuboide
    """
    mat      = MATERIALS_DB[mat_id]
    H_max    = mat['field_max']
    mat_name = f"{mat['name']} ({mat['formula']})"

    np.random.seed(7)                          # RF-07 (RNF-07): reproducibilidad del ruido

    fig = plt.figure(figsize=(18, 11))         # RF-11: dimensiones de la figura
    fig.patch.set_facecolor(COLORS['bg'])
    gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.42, wspace=0.32)

    # ── [0,0] HISTÉRESIS COMPARATIVA ─────────────────────────────────────────
    ax_hyst = fig.add_subplot(gs[0, 0])
    ax_hyst.set_facecolor(COLORS['panel'])
    ax_hyst.tick_params(colors=COLORS['subtext'])
    ax_hyst.grid(True, alpha=0.3)

    for idx, sz in enumerate(sizes):
        color = COLOR_CYCLE[idx % len(COLOR_CYCLE)]

        # Curva de esfera (si el modelo existe)
        if 'sphere' in models:
            validate_range(sz, mat_id, 'esfera')
            hc, mr = predict(sz, models['sphere'])
            H, M   = llg_hysteresis(hc, mr, H_max=H_max, noise_level=noise)
            ax_hyst.plot(H, M, color=color, linewidth=1.6,
                         label=f'{sz:.0f} nm (esf.)')

        # Curva de cuboide superpuesta con línea punteada (si el modelo existe)
        if 'cuboid' in models:
            validate_range(sz, mat_id, 'cuboide')
            hc_c, mr_c = predict(sz, models['cuboid'])
            H_c, M_c   = llg_hysteresis(hc_c, mr_c, H_max=H_max, noise_level=noise)
            ax_hyst.plot(H_c, M_c, color=color, linewidth=1.2, linestyle='--',
                         alpha=0.75, label=f'{sz:.0f} nm (cub.)')

    ax_hyst.axhline(0, color=COLORS['border'], linewidth=0.6)
    ax_hyst.axvline(0, color=COLORS['border'], linewidth=0.6)
    ax_hyst.set_xlabel('H (mT)')
    ax_hyst.set_ylabel('M / Ms')
    ax_hyst.set_title(f'Histéresis — {mat_name}',
                      color=COLORS['text'], fontsize=10, pad=8)
    ax_hyst.legend(loc='lower right', fontsize=7.5, ncol=2)
    ax_hyst.set_xlim(-H_max, H_max)
    ax_hyst.set_ylim(-1.15, 1.15)

    # ── [0,1] PAISAJE DE ENERGÍAS (4 contribuciones)  ────────────────────────
    ax_ener = fig.add_subplot(gs[0, 1])
    ax_ener.set_facecolor(COLORS['panel'])
    ax_ener.tick_params(colors=COLORS['subtext'])
    ax_ener.grid(True, alpha=0.3)

    validate_range(pred, mat_id, 'esfera')
    hc_pred, _ = predict(pred, models.get('sphere', models.get('cuboid')))
    energies   = energy_landscape(hc_pred)

    # RF-22: graficar las 4 contribuciones con colores del tema
    ax_ener.plot(energies['H'], energies['zeeman'],  color=COLORS['accent1'],
                 linewidth=1.8, label='Zeeman')
    ax_ener.plot(energies['H'], energies['exchange'], color=COLORS['accent2'],
                 linewidth=1.8, label='Intercambio')
    ax_ener.plot(energies['H'], energies['demag'],   color=COLORS['accent3'],
                 linewidth=1.8, label='Desmagnetiz.')
    ax_ener.plot(energies['H'], energies['aniso'],   color=COLORS['accent4'],
                 linewidth=1.8, label='Anisotropía')

    # Marcador vertical en Hc y -Hc
    ax_ener.axvline( hc_pred, color=COLORS['subtext'], linewidth=0.8,
                     linestyle=':', alpha=0.7)
    ax_ener.axvline(-hc_pred, color=COLORS['subtext'], linewidth=0.8,
                     linestyle=':', alpha=0.7)
    ax_ener.text( hc_pred + 8, 0.25, f'+Hc\n{hc_pred:.0f} mT',
                  color=COLORS['subtext'], fontsize=7)

    ax_ener.axhline(0, color=COLORS['border'], linewidth=0.6)
    ax_ener.set_xlabel('H (mT)')
    ax_ener.set_ylabel('Energía (u.a.)')
    ax_ener.set_title(f'Paisaje de Energías — {mat_name} — {pred:.0f} nm',
                      color=COLORS['text'], fontsize=10, pad=8)
    ax_ener.legend(loc='upper right', fontsize=7.5)
    ax_ener.set_xlim(-620, 680)

    # ── [1,:] TABLA RESUMEN  ──────────────────────────────────────────────────
    ax_tbl = fig.add_subplot(gs[1, :])
    ax_tbl.set_facecolor(COLORS['bg'])
    ax_tbl.axis('off')

    table_data = []
    for sz in sizes:
        # Fila esfera
        if 'sphere' in models:
            hc_s, mr_s = predict(sz, models['sphere'])
            in_range   = '✓' if mat['range'][0] <= sz <= mat['range'][1] else '⚠ extrap.'
            table_data.append([f'{sz:.0f}', 'Esfera', f'{hc_s:.1f}', f'{mr_s:.3f}', in_range])
        # Fila cuboide
        if 'cuboid' in models:
            hc_c, mr_c = predict(sz, models['cuboid'])
            in_range   = '✓' if mat['range'][0] <= sz <= mat['range'][1] else '⚠ extrap.'
            table_data.append([f'{sz:.0f}', 'Cuboide', f'{hc_c:.1f}', f'{mr_c:.3f}', in_range])

    columns = ['Tamaño (nm)', 'Geometría', 'Hc (mT)', 'Mr/Ms', 'Rango ML']

    table = ax_tbl.table(
        cellText=table_data,
        colLabels=columns,
        loc='center',
        cellLoc='center',
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.5)

    # Estilo oscuro coherente con el tema (RF-15)
    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor(COLORS['border'])
        if row == 0:
            cell.set_facecolor('#1f2937')
            cell.set_text_props(color=COLORS['accent1'], weight='bold')
        else:
            cell.set_facecolor(COLORS['panel'])
            # Colorear la columna "Rango ML" según el valor
            if col == 4 and row > 0:
                text = table_data[row - 1][4]
                cell.set_text_props(
                    color=COLORS['accent3'] if text == '✓' else COLORS['warn']
                )
            else:
                cell.set_text_props(color=COLORS['text'])

    ax_tbl.set_title(
        f'Resumen de Propiedades Magnéticas (GBR-ML) — {mat_name} — '
        f'Entrenado con {len(MATERIALS_DB[mat_id]["sphere"])} pts | '
        f'Rango: {mat["range"][0]}–{mat["range"][1]} nm | '
        f'Ref: {mat["ref"]}',
        color=COLORS['subtext'],
        fontsize=7.5,
        pad=10,
    )

    # ── Título principal de la figura ─────────────────────────────────────────
    fig.suptitle(
        f'Simulación Micromagnética con ML  ·  {mat_name}  ·  '
        f'Tamaños comparados: {[f"{s:.0f}" for s in sizes]} nm',
        color=COLORS['text'],
        fontsize=11,
        fontweight='bold',
        y=0.98,
    )

    # ── Exportación (RF-16)  ─────────────────────────────────────────────────
    sizes_str = '_'.join(str(int(s)) for s in sizes)
    filename  = f'fig_{mat_id}_{sizes_str}nm.{fmt}'
    filepath  = os.path.join(output_dir, filename)

    fig.savefig(filepath, dpi=dpi, bbox_inches='tight',
                facecolor=COLORS['bg'])
    plt.close(fig)

    return filepath

# =============================================================================
#  INTERFAZ DE LÍNEA DE COMANDOS — argparse  (RF-19)
# =============================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        prog='micromagnetic_simulator_v2.py',
        description=(
            'Sistema de Simulación Micromagnética con ML — v2.0\n'
            'Genera figuras de histéresis, energías y tabla comparativa\n'
            'para nanopartículas magnéticas usando Gradient Boosting Regression.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Ejemplos:\n'
            '  python micromagnetic_simulator_v2.py\n'
            '  python micromagnetic_simulator_v2.py --material fe --sizes 16 30 44 60\n'
            '  python micromagnetic_simulator_v2.py --material co --sizes 10 20 40 --pred 20\n'
            '  python micromagnetic_simulator_v2.py --sizes 5 10 20 50 --dpi 300 --format svg\n'
        )
    )

    parser.add_argument(
        '--material', '-m',
        type=str,
        choices=list(MATERIALS_DB.keys()),
        default='fe',
        help='Material a simular. Opciones: fe, permalloy, co, fe3o4 (default: fe)',
    )
    parser.add_argument(
        '--sizes', '-s',
        type=float,
        nargs='+',
        default=[16, 30, 44, 60],
        help='Lista de tamaños (nm) para el gráfico de histéresis (default: 16 30 44 60)',
    )
    parser.add_argument(
        '--pred', '-p',
        type=float,
        default=None,
        help='Tamaño (nm) para el paisaje de energías. Si no se indica, usa el primero de --sizes',
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Carpeta de salida para las figuras (default: ./outputs/ junto al script)',
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=170,
        help='Resolución de exportación en DPI (default: 170)',
    )
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['png', 'svg', 'pdf', 'eps'],
        default='png',
        dest='fmt',
        help='Formato de exportación (default: png)',
    )
    parser.add_argument(
        '--noise',
        type=float,
        default=0.008,
        help='Nivel de ruido gaussiano para el lazo LLG (default: 0.008)',
    )

    return parser.parse_args()

# =============================================================================
#  PUNTO DE ENTRADA PRINCIPAL
# =============================================================================

def main():
    t_start = time.perf_counter()

    args = parse_args()

    # ── Carpeta de salida ─────────────────────────────────────────────────────
    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'outputs')
    else:
        output_dir = os.path.normpath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    # ── Tamaño de predicción de energías ──────────────────────────────────────
    pred = args.pred if args.pred is not None else args.sizes[0]

    # ── Cabecera informativa ──────────────────────────────────────────────────
    mat      = MATERIALS_DB[args.material]
    sep      = '─' * 62
    print(sep)
    print(f'  Simulador Micromagnético ML  v2.0  —  Fase 1')
    print(sep)
    print(f'  Material      : {mat["name"]} ({mat["formula"]})')
    print(f'  Tamaños (nm)  : {args.sizes}')
    print(f'  Pred. energía : {pred} nm')
    print(f'  Rango ML      : {mat["range"][0]}–{mat["range"][1]} nm')
    print(f'  Salida        : {output_dir}')
    print(f'  DPI / Formato : {args.dpi} / {args.fmt.upper()}')
    print(f'  Ruido LLG     : {args.noise}')
    print(sep)

    # ── Entrenamiento de modelos ──────────────────────────────────────────────
    print('  Entrenando modelos GBR...')
    t_ml = time.perf_counter()
    models = load_models(args.material)
    t_ml   = time.perf_counter() - t_ml
    geoms  = list(models.keys())
    print(f'  ✓ Modelos entrenados: {geoms}  ({t_ml*1000:.1f} ms)')

    # ── Validación de tamaños ─────────────────────────────────────────────────
    print('  Validando tamaños ingresados...')
    for sz in args.sizes + [pred]:
        for geom in geoms:
            validate_range(sz, args.material, geom)

    # ── Generación de figura ──────────────────────────────────────────────────
    print('  Generando figura...')
    t_fig   = time.perf_counter()
    outpath = build_figure(
        mat_id=args.material,
        models=models,
        sizes=args.sizes,
        pred=pred,
        noise=args.noise,
        dpi=args.dpi,
        fmt=args.fmt,
        output_dir=output_dir,
    )
    t_fig = time.perf_counter() - t_fig

    # ── Resumen final ─────────────────────────────────────────────────────────
    t_total = time.perf_counter() - t_start
    print(sep)
    print(f'  ✓ Figura guardada : {outpath}')
    print(f'  Tiempo ML         : {t_ml*1000:.1f} ms')
    print(f'  Tiempo figura     : {t_fig*1000:.0f} ms')
    print(f'  Tiempo total      : {t_total*1000:.0f} ms')
    print(sep)


if __name__ == '__main__':
    main()
