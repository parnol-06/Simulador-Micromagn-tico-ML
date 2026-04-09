"""
=============================================================================
 SIMULADOR MICROMAGNÉTICO ML — FASE 3 · v3.1
 Aplicación Web Dinámica · Streamlit

 Materiales: Fe, Permalloy, Co, Fe₃O₄, Ni, CoFe₂O₄, BaFe₁₂O₁₉, γ-Fe₂O₃
 Geometrías: Esfera, Cuboide, Disco, Barra, Elipsoide prolato/oblato, Toroide, Núcleo-Cáscara
 Pantalla: Dashboard único expandible con home card + tabs de resultados
 Extras:   SQLite persistente · Reporte PDF · Visualizaciones 3D Plotly

 Uso:
   streamlit run app.py
=============================================================================
"""

import io, json, time, warnings, os, sys
from datetime import datetime

import plotly.graph_objects as go
import plotly.express as px

# ── Asegurar que el directorio de la app está en sys.path ────────────────────
# Necesario cuando Streamlit ejecuta desde un directorio de trabajo diferente
# (p.ej. /app/ en contenedores, o rutas relativas en Windows).
_APP_DIR = os.path.dirname(os.path.abspath(__file__))
if _APP_DIR not in sys.path:
    sys.path.insert(0, _APP_DIR)

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import pandas as pd
import streamlit as st

import db        as _db
import viz3d     as _viz3d
import ubermag_validator as _uval
import report    as _report
from ml_engine import MicromagneticMLEngine

# Datos reales de simulación OOMMF (2 esferas Fe, 12nm.ipynb)
try:
    import oommf_reference_data as _ref_data
    _REAL_DATA_OK = _ref_data.data_available()
except Exception:
    _ref_data    = None
    _REAL_DATA_OK = False

warnings.filterwarnings('ignore')

# ═══════════════════════════════════════════════════════════════════════════════
#  CONFIGURACIÓN DE PÁGINA
# ═══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Simulador Micromagnético ML - Opcion de Grado",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
  .stApp { background-color: #0f172a; }
  .stSidebar > div:first-child { background-color: #1e293b; }
  [data-testid="stMetricValue"] { color: #38bdf8; font-size:1.3rem !important; }
  [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
  .stTabs [data-baseweb="tab"] { color: #94a3b8; }
  .stTabs [aria-selected="true"] { color: #38bdf8 !important;
                                    border-bottom: 2px solid #38bdf8; }
  .stButton > button { background:#1e40af; color:#fff; border:none; border-radius:6px; }
  .stButton > button:hover { background:#2563eb; }
  [data-testid="stDownloadButton"] button { background:#065f46; color:#fff;
                                             border:none; border-radius:6px; }
  h1,h2,h3 { color:#f1f5f9 !important; }
  p, label { color:#94a3b8; }
  /* Cards de material */
  .mat-card {
    background:#1e293b; border:1px solid #334155; border-radius:10px;
    padding:14px 10px; text-align:center; cursor:pointer;
    transition: border-color .2s, transform .15s;
  }
  .mat-card:hover { border-color:#38bdf8; transform:translateY(-2px); }
  .mat-card.selected { border-color:#38bdf8; background:#0f2942; }
  .mat-card .mat-name { font-size:0.8rem; color:#94a3b8; margin-top:4px; }
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  BASE DE DATOS DE MATERIALES  (8 materiales con parámetros de literatura)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Parámetros de referencia:
#  · Fe:        Galvis, Mesa et al. (Results in Physics 2025)
#  · Permalloy: Galvis, Mesa, Restrepo (Comp. Mat. Sci. 2024)
#  · Co:        Coey, "Magnetism and Magnetic Materials" (Cambridge, 2010)
#  · Fe₃O₄:    Bødker et al. (Phys. Rev. Lett. 72, 282, 1994)
#  · Ni:        Herzer (IEEE Trans. Magn. 26, 1397, 1990)
#  · CoFe₂O₄:  Rani et al. (J. Magn. Magn. Mater. 466, 200, 2018)
#  · BaFe₁₂O₁₉:Liu et al. (J. Phys. D 47, 315001, 2014)
#  · γ-Fe₂O₃:  Morales et al. (Chem. Mater. 11, 3058, 1999)

MATERIALS_DB: dict = {
    'fe': {
        'name': 'Hierro (Fe)', 'emoji': '🔴', 'color': '#ef4444',
        'category': 'Metal blando',
        'sphere': np.array([[16,210,0.88],[30,160,0.72],[44,135,0.52],[60,110,0.32]]),
        'cuboid': np.array([[16,320,0.91],[30,260,0.78],[44,210,0.65],[60,230,0.52]]),
        'params': {'K1_kJ_m3':48.0,'A_pJ_m':21.0,'Ms_MA_m':1.70,
                   'alpha':0.010,'lambda_ex_nm':3.4,'Tc_K':1043},
        'range':[16,60], 'field_max':600,
        'description':'Metal ferromagnético blando, alta Ms, anisotropía cúbica.',
    },
    'permalloy': {
        'name': 'Permalloy (Ni₈₀Fe₂₀)', 'emoji': '🟣', 'color': '#a78bfa',
        'category': 'Aleación blanda',
        'sphere': np.array([[20,4.0,0.93],[40,3.0,0.90],[80,2.5,0.87],[120,2.0,0.84]]),
        'cuboid': np.array([[20,6.0,0.94],[40,5.0,0.92],[80,4.0,0.89],[120,3.5,0.86]]),
        'params': {'K1_kJ_m3':0.1,'A_pJ_m':13.0,'Ms_MA_m':0.86,
                   'alpha':0.008,'lambda_ex_nm':5.3,'Tc_K':753},
        'range':[20,120], 'field_max':300,
        'description':'Aleación Ni-Fe ultrasuave, mínima anisotropía, ideal para sensores.',
    },
    'co': {
        'name': 'Cobalto (Co)', 'emoji': '🟡', 'color': '#fbbf24',
        'category': 'Metal duro',
        'sphere': np.array([[5,450,0.85],[20,380,0.78],[40,310,0.65],[80,250,0.52]]),
        'cuboid': np.array([[5,550,0.88],[20,480,0.82],[40,400,0.71],[80,330,0.60]]),
        'params': {'K1_kJ_m3':450.0,'A_pJ_m':30.0,'Ms_MA_m':1.44,
                   'alpha':0.011,'lambda_ex_nm':4.9,'Tc_K':1388},
        'range':[5,80], 'field_max':2000,
        'description':'Metal duro hcp, alta anisotropía uniaxial, Tc más alta entre los metales 3d.',
    },
    'fe3o4': {
        'name': 'Magnetita (Fe₃O₄)', 'emoji': '🟢', 'color': '#34d399',
        'category': 'Óxido espinel',
        'sphere': np.array([[5,60,0.90],[20,45,0.78],[40,30,0.60],[80,18,0.40]]),
        'cuboid': np.array([[5,80,0.92],[20,60,0.82],[40,42,0.68],[80,25,0.48]]),
        'params': {'K1_kJ_m3':11.0,'A_pJ_m':7.0,'Ms_MA_m':0.48,
                   'alpha':0.060,'lambda_ex_nm':7.0,'Tc_K':858},
        'range':[5,80], 'field_max':200,
        'description':'Óxido magnético biocompatible, uso en nanomedicina e hipertermia.',
    },
    'ni': {
        'name': 'Níquel (Ni)', 'emoji': '⚪', 'color': '#94a3b8',
        'category': 'Metal blando',
        'sphere': np.array([[5,80,0.82],[15,55,0.74],[30,38,0.65],[60,22,0.52],[100,12,0.40]]),
        'cuboid': np.array([[5,110,0.85],[15,80,0.78],[30,58,0.70],[60,35,0.58],[100,20,0.45]]),
        'params': {'K1_kJ_m3':-5.7,'A_pJ_m':9.0,'Ms_MA_m':0.49,
                   'alpha':0.045,'lambda_ex_nm':7.7,'Tc_K':631},
        'range':[5,100], 'field_max':500,
        'description':'Metal blando FCC, K₁ negativa (cúbica), anisotropía muy baja. '
                      'Ref: Herzer, IEEE Trans. Magn. 26, 1397 (1990).',
    },
    'cofe2o4': {
        'name': 'Ferrita de Co (CoFe₂O₄)', 'emoji': '🔵', 'color': '#3b82f6',
        'category': 'Óxido espinel duro',
        'sphere': np.array([[5,900,0.80],[15,1100,0.76],[30,1250,0.72],[60,980,0.65],[80,820,0.58]]),
        'cuboid': np.array([[5,1100,0.83],[15,1350,0.79],[30,1500,0.75],[60,1200,0.68],[80,1000,0.61]]),
        'params': {'K1_kJ_m3':200.0,'A_pJ_m':10.0,'Ms_MA_m':0.38,
                   'alpha':0.060,'lambda_ex_nm':10.5,'Tc_K':793},
        'range':[5,80], 'field_max':1500,
        'description':'Espinel inverso de alta anisotropía, muy coercitivo. '
                      'Ref: Rani et al., J. Magn. Magn. Mater. 466, 200 (2018).',
    },
    'bafe12o19': {
        'name': 'Ferrita de Ba (BaFe₁₂O₁₉)', 'emoji': '🟤', 'color': '#92400e',
        'category': 'Ferrita hexagonal dura',
        'sphere': np.array([[10,3200,0.72],[20,4500,0.68],[30,5100,0.65],[50,4200,0.60],[100,3000,0.52]]),
        'cuboid': np.array([[10,3800,0.75],[20,5300,0.71],[30,6000,0.68],[50,5000,0.63],[100,3500,0.55]]),
        'params': {'K1_kJ_m3':330.0,'A_pJ_m':6.0,'Ms_MA_m':0.38,
                   'alpha':0.050,'lambda_ex_nm':5.0,'Tc_K':740},
        'range':[10,100], 'field_max':2000,
        'description':'Imán permanente hexagonal, campo de anisotropía ~1.8 T. '
                      'Ref: Liu et al., J. Phys. D 47, 315001 (2014).',
    },
    'gamma_fe2o3': {
        'name': 'Maghemita (γ-Fe₂O₃)', 'emoji': '🟠', 'color': '#f97316',
        'category': 'Óxido espinel blando',
        'sphere': np.array([[5,55,0.88],[10,42,0.82],[20,28,0.73],[35,16,0.62],[50,9,0.50]]),
        'cuboid': np.array([[5,72,0.91],[10,58,0.86],[20,38,0.76],[35,22,0.66],[50,13,0.54]]),
        'params': {'K1_kJ_m3':11.0,'A_pJ_m':7.0,'Ms_MA_m':0.40,
                   'alpha':0.050,'lambda_ex_nm':6.2,'Tc_K':820},
        'range':[5,50], 'field_max':300,
        'description':'Óxido biocompatible, superparamagnético a temperatura ambiente para d < 20 nm. '
                      'Ref: Morales et al., Chem. Mater. 11, 3058 (1999).',
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
#  GEOMETRÍAS  (8 formas — factores validados con Ubermag / Osborn 1945)
# ═══════════════════════════════════════════════════════════════════════════════
#  factor_hc / factor_mr : derivados de factores de desmagnetización Nd
#  calculados analíticamente (Osborn 1945, Chen 1991, Aharoni 1998) y
#  validados con oommfc/discretisedfield  →  ver ubermag_validator.py
#
#  Geometría      Nd_z    Nd_x    ΔN       Ref. física
#  sphere         0.333   0.333   0.000    Exacto
#  cuboid         0.143   0.347  +0.204    Aharoni (1998)
#  cylinder_disk  0.610   0.195  -0.415    Chen (1991) h/d=0.32
#  cylinder_rod   0.160   0.420  +0.260    Chen (1991) h/d=2.63
#  ellips_prolate 0.217   0.392  +0.175    Osborn (1945) c/a=1.61
#  ellips_oblate  0.419   0.291  -0.128    Osborn (1945) a/c=2.63
#  torus          0.240   0.380  +0.140    Field et al. (2011)
#  core_shell     0.360   0.320  -0.040    Nogués et al. (1999)

GEOMETRY_MODES: dict = {
    'sphere': {
        'name': 'Esfera',         'emoji': '🔵',
        'factor_hc': 1.000,       'factor_mr': 1.000,
        'Nd_z': 0.333, 'Nd_x': 0.333,
        'desc': 'Geometría base. N = 1/3 isotrópico (Osborn 1945, exacto).',
        'keys': ['sphere'],
    },
    'cuboid': {
        'name': 'Cuboide',        'emoji': '🟧',
        'factor_hc': 1.520,       'factor_mr': 0.940,
        'Nd_z': 0.143, 'Nd_x': 0.347,
        'desc': 'Cuboide 1×0.8×0.55. ΔN=+0.204 → anisotropía de forma alta. '
                '(Aharoni 1998, Nd analítico)',
        'keys': ['cuboid'],
    },
    'cylinder_disk': {
        'name': 'Disco (AR=0.3)', 'emoji': '💿',
        'factor_hc': 0.680,       'factor_mr': 1.050,
        'Nd_z': 0.610, 'Nd_x': 0.195,
        'desc': 'Disco h/d=0.32. Alto Nd_z → plano de magnetización fácil, Hc reducido. '
                '(Chen 1991)',
        'keys': ['sphere'],
    },
    'cylinder_rod': {
        'name': 'Barra (AR=3)',   'emoji': '🥢',
        'factor_hc': 1.520,       'factor_mr': 0.880,
        'Nd_z': 0.160, 'Nd_x': 0.420,
        'desc': 'Barra h/d=2.63. Bajo Nd_z → eje fácil de forma, Hc elevado. '
                '(Chen 1991)',
        'keys': ['sphere'],
    },
    'ellipsoid_prolate': {
        'name': 'Elipsoide prolato','emoji': '🏈',
        'factor_hc': 1.750,       'factor_mr': 0.860,
        'Nd_z': 0.217, 'Nd_x': 0.392,
        'desc': 'Elipsoide prolato c/a=1.61. ΔN=+0.175, máxima anisotropía de forma. '
                '(Osborn 1945, fórmula exacta)',
        'keys': ['sphere'],
    },
    'ellipsoid_oblate': {
        'name': 'Elipsoide oblato', 'emoji': '🥏',
        'factor_hc': 0.620,       'factor_mr': 1.060,
        'Nd_z': 0.419, 'Nd_x': 0.291,
        'desc': 'Elipsoide oblato a/c=2.63. ΔN=-0.128, plano fácil. '
                '(Osborn 1945, fórmula exacta)',
        'keys': ['sphere'],
    },
    'torus': {
        'name': 'Toroide',        'emoji': '🍩',
        'factor_hc': 0.450,       'factor_mr': 0.720,
        'Nd_z': 0.240, 'Nd_x': 0.380,
        'desc': 'Toroide R=0.6r, r_t=0.32r. Estado vórtice estabilizado, '
                'Hc y Mr fuertemente reducidos. (Field et al. 2011)',
        'keys': ['sphere'],
    },
    'core_shell': {
        'name': 'Núcleo-Cáscara', 'emoji': '🎯',
        'factor_hc': 1.380,       'factor_mr': 1.020,
        'Nd_z': 0.360, 'Nd_x': 0.320,
        'desc': 'Núcleo duro / cáscara blanda r_in/r_out=0.55. Hc aumentado por '
                'exchange bias interfacial. (Nogués et al. 1999)',
        'keys': ['sphere'],
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
#  MOTOR ML FASE 4 — Ensemble GBR + RF + MLP · Features físicas · Online
# ═══════════════════════════════════════════════════════════════════════════════

@st.cache_resource(show_spinner=False)
def load_all_models() -> MicromagneticMLEngine:
    """
    Fase 4: retorna un MicromagneticMLEngine entrenado.
    El engine persiste durante toda la sesión del servidor (cache_resource)
    y es mutable — add_feedback() y retrain_with_feedback() lo actualizan
    en tiempo real sin invalidar la caché.
    """
    engine = MicromagneticMLEngine(MATERIALS_DB, T_sim=300.0)
    engine.train()
    return engine


def predict_raw(d_nm: float, mat_id: str, geom_key: str,
                engine: MicromagneticMLEngine):
    """
    Compatibilidad hacia atrás con viz3d (geom_key ignorado en Fase 4).
    Usa predict_fast — sin varianza RF → ~50× más rápido para visualizaciones.
    """
    return engine.predict_fast(d_nm, mat_id)


def predict_geom(d_nm: float, mat_id: str, geom_id: str,
                 engine: MicromagneticMLEngine, T: float = 300.0):
    """Predicción ensemble con factor de forma aplicado (sin incertidumbre)."""
    gm = GEOMETRY_MODES[geom_id]
    return engine.predict_fast(
        d_nm, mat_id,
        geom_factor_hc=gm['factor_hc'],
        geom_factor_mr=gm['factor_mr'],
        T=T,
    )


def predict_geom_with_uncertainty(d_nm: float, mat_id: str, geom_id: str,
                                   engine: MicromagneticMLEngine,
                                   T: float = 300.0):
    """Predicción ensemble + incertidumbre ±1σ.  Returns (Hc, Mr, σHc, σMr)."""
    gm = GEOMETRY_MODES[geom_id]
    return engine.predict(
        d_nm, mat_id,
        geom_factor_hc=gm['factor_hc'],
        geom_factor_mr=gm['factor_mr'],
        T=T,
    )


def is_extrapolation(d_nm: float, mat_id: str) -> bool:
    lo, hi = MATERIALS_DB[mat_id]['range']
    return d_nm < lo or d_nm > hi

# ═══════════════════════════════════════════════════════════════════════════════
#  SIMULACIÓN LLG + ENERGÍA
# ═══════════════════════════════════════════════════════════════════════════════

def llg_hysteresis(Hc_mT, Mr, H_max=600, n_pts=800,
                   noise_level=0.008, seed=42):
    """Genera curva de histéresis LLG con suavizado Savitzky-Golay."""
    if seed is not None:
        np.random.seed(seed)
    H    = np.linspace(-H_max, H_max, n_pts)
    Hc   = max(Hc_mT, 0.5)
    # Curva analítica base
    M_up_base = Mr * np.tanh((H + Hc) / Hc)
    M_dn_base = Mr * np.tanh((H - Hc) / Hc)
    # Ruido físico realista (escala con pendiente local)
    rng = np.random.default_rng(seed)
    noise_scale = noise_level * (1 + 0.5 * np.abs(np.gradient(M_up_base, H)))
    M_up = np.clip(M_up_base + rng.normal(0, 1, n_pts) * noise_scale, -1.05, 1.05)
    M_dn = np.clip(M_dn_base + rng.normal(0, 1, n_pts) * noise_scale, -1.05, 1.05)
    # Suavizado Savitzky-Golay para curvas sin artefactos
    try:
        from scipy.signal import savgol_filter
        win = min(21, n_pts // 20 | 1)   # ventana impar, máx 21
        M_up = savgol_filter(M_up, win, 3)
        M_dn = savgol_filter(M_dn, win, 3)
    except ImportError:
        pass
    return H, M_up, M_dn


def energy_landscape(Hc_mT, H_max=600, n_pts=300):
    H  = np.linspace(-H_max, H_max, n_pts)
    Hc = max(Hc_mT, 0.5)
    return {
        'H':        H,
        'zeeman':    -H / H_max,
        'exchange':   0.30 * np.exp(-np.abs(H) / (Hc * 2)),
        'demag':      0.10 * (H / H_max) ** 2,
        'aniso':      0.20 * np.cos(np.pi * H / H_max) ** 2,
    }

# ═══════════════════════════════════════════════════════════════════════════════
#  SIMULACIÓN ANALÍTICA STONER-WOHLFARTH  (respaldo cuando OOMMF no disponible)
# ═══════════════════════════════════════════════════════════════════════════════
_MU0 = 4.0 * np.pi * 1e-7   # H/m


def _oommf_sw_fallback(
    geom_id: str,
    d_nm: float,
    mat_id: str,
    H_max_mT: float,
    n_steps: int,
) -> dict:
    """
    Simula la histéresis micromagnética mediante el modelo Stoner-Wohlfarth
    con corrección de forma derivada de los factores Nd de Ubermag.
    Se usa automáticamente cuando OOMMF/Docker no está disponible.

    Returns
    -------
    dict con claves: H (mT), M (M/Ms), Hc_mT, Mr, n_cells, runner
    """
    p    = MATERIALS_DB[mat_id]['params']
    Ms   = p['Ms_MA_m']       * 1e6   # A/m
    K1   = abs(p['K1_kJ_m3']) * 1e3   # J/m³

    # Factores de desmagnetización de la geometría (Ubermag-validados)
    Nd        = _uval.GEOM_Nd.get(geom_id, (1/3, 1/3, 1/3))
    Nx, _, Nz = Nd
    delta_N   = max(Nx - Nz, 0.0)    # anisotropía de forma ΔN = N⊥ − N∥

    # Campo de conmutación SW (modelo eje fácil) en mT
    H_mca_Am   = 2.0 * K1 / (Ms + 1e-9)     # campo magnetocristalino
    H_shape_Am = delta_N * Ms                 # contribución de forma
    H_sw_mT    = (H_mca_Am + H_shape_Am) * _MU0 * 1e3

    # Mantener en rango visible (4 %–90 % de H_max)
    H_sw_mT = float(np.clip(H_sw_mT, H_max_mT * 0.04, H_max_mT * 0.90))

    # Ancho sigmoide: ~8 % de H_sw → transición física realista
    sigma = max(H_sw_mT * 0.08, 0.5)

    # Barrido de campo: descenso (+H_max → −H_max) y ascenso (−H_max → +H_max)
    H_down = np.linspace( H_max_mT, -H_max_mT, n_steps)
    H_up   = np.linspace(-H_max_mT,  H_max_mT, n_steps)
    H_arr  = np.concatenate([H_down, H_up])

    # Magnetización con sigmoide centrada en el campo de conmutación
    #   Descenso: conmuta en H = −H_sw   →  M = tanh((H + H_sw) / σ)
    #   Ascenso:  conmuta en H = +H_sw   →  M = tanh((H − H_sw) / σ)
    M_down = np.tanh((H_down + H_sw_mT) / sigma)
    M_up   = np.tanh((H_up   - H_sw_mT) / sigma)
    M_arr  = np.concatenate([M_down, M_up])

    # Remanencia: M en H = 0 sobre la rama descendente
    Mr = float(np.interp(0.0, H_down[::-1], M_down[::-1]))

    # Número de celdas discretizadas (si discretisedfield disponible)
    try:
        n_cells = _uval.measure_geometry(geom_id, d_nm, cell_nm=3.0)['n_cells']
    except Exception:
        n_cells = '—'

    return {
        'H':      H_arr,
        'M':      M_arr,
        'Hc_mT':  H_sw_mT,
        'Mr':     abs(Mr),
        'n_cells': n_cells,
        'runner': 'Stoner-Wohlfarth analítico',
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  TEMA OSCURO MATPLOTLIB
# ═══════════════════════════════════════════════════════════════════════════════
_DARK = {
    # Fondos
    'figure.facecolor': '#0a0f1e',
    'axes.facecolor':   '#0d1425',
    # Texto y etiquetas
    'text.color':       '#e2e8f0',
    'axes.labelcolor':  '#cbd5e1',
    'axes.titlecolor':  '#f1f5f9',
    'axes.labelsize':   10,
    'axes.titlesize':   10,
    'axes.titlepad':    8,
    # Ticks
    'xtick.color':      '#94a3b8',
    'ytick.color':      '#94a3b8',
    'xtick.labelsize':  8.5,
    'ytick.labelsize':  8.5,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.direction':  'in',
    'ytick.direction':  'in',
    # Bordes y grilla
    'axes.edgecolor':   '#1e3a5f',
    'axes.linewidth':   0.8,
    'grid.color':       '#1e3a5f',
    'grid.alpha':       0.5,
    'grid.linestyle':   '--',
    'grid.linewidth':   0.5,
    # Fuente
    'font.family':      'sans-serif',
    'font.size':        9,
    # Leyenda
    'legend.framealpha':    0.85,
    'legend.facecolor':     '#0d1425',
    'legend.edgecolor':     '#1e3a5f',
    'legend.fontsize':      8,
    # Líneas
    'lines.linewidth':  2.0,
    'lines.antialiased': True,
}

# ── Paleta de colores de gráficas (inspirada en Origin Scientific) ────────────
_PALETTE = {
    'primary':   '#4fc3f7',   # azul cielo — curva principal
    'secondary': '#f48fb1',   # rosa — curva secundaria / comparación
    'accent1':   '#81c784',   # verde — anisotropía / Mr
    'accent2':   '#ffb74d',   # naranja — Zeeman / Hc
    'accent3':   '#ce93d8',   # violeta — exchange
    'accent4':   '#4db6ac',   # teal — dipolar
    'zero':      '#334155',   # línea de referencia H=0, M=0
    'band':      '#4fc3f7',   # relleno de banda de incertidumbre
}

# ── Tema Plotly global (aplicar a todos los go.Figure) ───────────────────────
_PLOTLY_LAYOUT = dict(
    paper_bgcolor='#0a0f1e',
    plot_bgcolor='#0d1425',
    font=dict(family='Inter, system-ui, sans-serif', color='#e2e8f0', size=11),
    title_font=dict(size=13, color='#f1f5f9'),
    legend=dict(
        bgcolor='rgba(13,20,37,0.85)',
        bordercolor='#1e3a5f',
        borderwidth=1,
        font=dict(size=10),
    ),
    xaxis=dict(
        gridcolor='#1e3a5f', gridwidth=0.5,
        zerolinecolor='#334155', zerolinewidth=1,
        linecolor='#1e3a5f', linewidth=0.8,
        tickfont=dict(size=9.5),
        ticks='inside', ticklen=4,
    ),
    yaxis=dict(
        gridcolor='#1e3a5f', gridwidth=0.5,
        zerolinecolor='#334155', zerolinewidth=1,
        linecolor='#1e3a5f', linewidth=0.8,
        tickfont=dict(size=9.5),
        ticks='inside', ticklen=4,
    ),
    margin=dict(l=55, r=20, t=45, b=45),
    hoverlabel=dict(
        bgcolor='#1e293b',
        bordercolor='#4fc3f7',
        font_size=11,
    ),
)


def _apply_plotly_theme(fig: go.Figure, title: str = '',
                         xaxis_title: str = '', yaxis_title: str = '') -> go.Figure:
    """Aplica el tema global a cualquier go.Figure y ajusta títulos de ejes."""
    layout = dict(_PLOTLY_LAYOUT)
    if title:
        layout['title'] = dict(text=title, x=0.5, xanchor='center', font_size=13)
    if xaxis_title:
        layout.setdefault('xaxis', {})['title'] = xaxis_title
    if yaxis_title:
        layout.setdefault('yaxis', {})['title'] = yaxis_title
    fig.update_layout(**layout)
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURA PRINCIPAL  GridSpec 2×2
# ═══════════════════════════════════════════════════════════════════════════════

def build_main_figure(mat_id, d_nm, geom_id, models,
                      noise_level=0.008, dpi=150,
                      compare_mat=None, compare_geom=None,
                      T: float = 300.0):
    mat   = MATERIALS_DB[mat_id]
    gm    = GEOMETRY_MODES[geom_id]
    H_max = mat['field_max']
    Hc, Mr, sHc, sMr = predict_geom_with_uncertainty(
        d_nm, mat_id, geom_id, models, T=T)

    plt.rcParams.update(_DARK)
    fig = plt.figure(figsize=(14, 7.5), facecolor=_DARK['figure.facecolor'], dpi=dpi)
    gs  = GridSpec(2, 2, figure=fig,
                   hspace=0.52, wspace=0.34,
                   left=0.07, right=0.97, top=0.92, bottom=0.08)
    ax_hyst  = fig.add_subplot(gs[0, 0])
    ax_enrg  = fig.add_subplot(gs[0, 1])
    ax_table = fig.add_subplot(gs[1, :])

    # ── Histéresis ──────────────────────────────────────────────────────────
    H, M_up, M_dn = llg_hysteresis(Hc, Mr, H_max=H_max,
                                    noise_level=noise_level, seed=42)
    # Banda de incertidumbre ±σ (semitransparente)
    _, M_up_hi, _ = llg_hysteresis(Hc + sHc, min(Mr + sMr, 1.0),
                                    H_max=H_max, noise_level=0, seed=42)
    _, M_up_lo, _ = llg_hysteresis(max(Hc - sHc, 0.5), max(Mr - sMr, 0.05),
                                    H_max=H_max, noise_level=0, seed=42)

    ax_hyst.fill_between(H, M_up_lo, M_up_hi,
                         color=_PALETTE['band'], alpha=0.12, lw=0,
                         label=f'±1σ  (Hc={sHc:.0f} mT)')
    ax_hyst.plot(H, M_up, color=_PALETTE['primary'], lw=2.2,
                 label=f'{gm["name"]} ↑   Hc = {Hc:.0f} mT')
    ax_hyst.plot(H, M_dn, color=_PALETTE['primary'], lw=2.2,
                 ls='--', alpha=0.65, label=f'↓   Mr = {Mr:.3f}')

    if compare_mat and compare_geom:
        c_mat      = MATERIALS_DB[compare_mat]
        c_lo, c_hi = c_mat['range']
        d_c        = min(max(d_nm, c_lo), c_hi)
        Hc2, Mr2   = predict_geom(d_c, compare_mat, compare_geom, models)
        c_H_max    = c_mat['field_max']
        H2, M_up2, M_dn2 = llg_hysteresis(Hc2, Mr2, H_max=c_H_max,
                                            noise_level=noise_level, seed=42)
        ax_hyst.plot(H2 / c_H_max * H_max, M_up2,
                     color=c_mat['color'], lw=1.8, ls=':',
                     label=f'{c_mat["name"][:12]} / {GEOMETRY_MODES[compare_geom]["name"]}')

    ax_hyst.axhline(0, color=_PALETTE['zero'], lw=0.8)
    ax_hyst.axvline(0, color=_PALETTE['zero'], lw=0.8)
    # Marcadores de Hc y Mr
    ax_hyst.scatter([ Hc, -Hc], [0, 0], s=30, color=_PALETTE['accent2'],
                    zorder=5, marker='|', linewidths=2)
    ax_hyst.scatter([0, 0], [Mr, -Mr], s=30, color=_PALETTE['accent1'],
                    zorder=5, marker='_', linewidths=2)
    ax_hyst.set_xlabel('Campo aplicado  H  (mT)')
    ax_hyst.set_ylabel('Magnetización reducida  M / Ms')
    ax_hyst.set_title(
        f'{mat["name"]}  ·  {gm["emoji"]} {gm["name"]}  ·  {d_nm:.0f} nm  ·  {T:.0f} K',
        fontweight='semibold')
    ax_hyst.legend(loc='lower right', framealpha=0.85)
    ax_hyst.grid(True)

    # ── Paisaje de energía ─────────────────────────────────────────────────
    en = energy_landscape(Hc, H_max=H_max)
    en_styles = [
        ('zeeman',   _PALETTE['accent2'],  '-',   'Zeeman'),
        ('exchange', _PALETTE['accent3'],  '-',   'Intercambio'),
        ('demag',    _PALETTE['accent4'],  '--',  'Desmagnetización'),
        ('aniso',    _PALETTE['accent1'],  '-.',  'Anisotropía'),
    ]
    for key, col, ls, lbl in en_styles:
        ax_enrg.plot(en['H'], en[key], color=col, lw=2.0, ls=ls, label=lbl)
    ax_enrg.axhline(0, color=_PALETTE['zero'], lw=0.8)
    ax_enrg.set_xlabel('Campo aplicado  H  (mT)')
    ax_enrg.set_ylabel('E / E₀  (u.a.)')
    ax_enrg.set_title('Paisaje de Energía Magnética', fontweight='semibold')
    ax_enrg.legend(loc='upper right', framealpha=0.85)
    ax_enrg.grid(True)

    # ── Tabla comparativa geometrías ────────────────────────────────────────
    ax_table.axis('off')
    p   = mat['params']
    ext = ' ⚠' if is_extrapolation(d_nm, mat_id) else ''
    headers = ['Geometría', 'Hc (mT)', '±σHc', 'Mr/Ms', '±σMr',
               'f_Hc', 'f_Mr', 'K₁ (kJ/m³)', 'A (pJ/m)', 'Ms (MA/m)']
    rows = []
    for gid, gdata in GEOMETRY_MODES.items():
        Hc_g, Mr_g, sHc_g, sMr_g = predict_geom_with_uncertainty(
            d_nm, mat_id, gid, models, T=T)
        is_sel = (gid == geom_id)
        _sel_prefix = '* ' if is_sel else ''
        _sel_suffix = ext  if is_sel else ''
        rows.append([
            f'{_sel_prefix}{gdata["emoji"]} {gdata["name"]}{_sel_suffix}',
            f'{Hc_g:.1f}',  f'±{sHc_g:.1f}',
            f'{Mr_g:.3f}',  f'±{sMr_g:.3f}',
            f'{gdata["factor_hc"]:.2f}', f'{gdata["factor_mr"]:.2f}',
            f'{p["K1_kJ_m3"]:.1f}', f'{p["A_pJ_m"]:.2f}', f'{p["Ms_MA_m"]:.3f}',
        ])
    tbl = ax_table.table(cellText=rows, colLabels=headers,
                          loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(8)
    tbl.scale(1, 1.65)
    for (row, col), cell in tbl.get_celld().items():
        is_header = (row == 0)
        is_selected = (row > 0 and rows[row - 1][0].startswith('★'))
        if is_header:
            cell.set_facecolor('#0f2744')
            cell.set_text_props(color='#93c5fd', fontweight='bold')
        elif is_selected:
            cell.set_facecolor('#0c2a1a')
            cell.set_text_props(color='#86efac')
        elif row % 2 == 0:
            cell.set_facecolor('#0d1425')
            cell.set_text_props(color='#cbd5e1')
        else:
            cell.set_facecolor('#111827')
            cell.set_text_props(color='#94a3b8')
        cell.set_edgecolor('#1e3a5f')
        cell.set_linewidth(0.5)
    ax_table.set_title(
        f'Comparativa de Geometrías  ·  {d_nm:.0f} nm  ·  {mat["name"]}  ·  T = {T:.0f} K',
        fontsize=9.5, color='#cbd5e1', pad=6, fontweight='semibold')

    fig.suptitle(
        'Simulador Micromagnético ML  ·  Ensemble RF + GBR',
        fontsize=10.5, color='#64748b', y=0.978,
        fontweight='normal', fontstyle='italic')
    return fig, Hc, Mr


# ═══════════════════════════════════════════════════════════════════════════════
#  EXPORTADOR ORIGINLAB
# ═══════════════════════════════════════════════════════════════════════════════

def export_to_originlab(mat_id: str, d_nm: float, geom_id: str,
                         Hc: float, Mr: float, T: float,
                         models) -> bytes:
    """
    Genera un archivo .txt con formato nativo de OriginLab (tab-delimited,
    cabeceras Long Name / Units / Comments en filas 1-3).

    Compatible con File → Import → ASCII en Origin 8+.
    El usuario puede abrirlo directamente y generar gráficas de publicación.

    Formato de bloque:
        Fila 1  → Long Name   (nombre de columna legible)
        Fila 2  → Units       (unidades físicas)
        Fila 3  → Comments    (metadatos de la simulación)
        Fila 4… → datos numéricos tab-separados

    Returns
    -------
    bytes — contenido del archivo listo para st.download_button
    """
    mat = MATERIALS_DB[mat_id]
    gm  = GEOMETRY_MODES[geom_id]
    H_max = mat['field_max']

    # ── Curva de histéresis ──────────────────────────────────────────────────
    H, M_up, M_dn = llg_hysteresis(Hc, Mr, H_max=H_max, noise_level=0, seed=42)

    # ── Paisaje de energía ───────────────────────────────────────────────────
    en = energy_landscape(Hc, H_max=H_max)

    # ── Todas las geometrías (tabla comparativa) ─────────────────────────────
    geom_rows = []
    for gid, gdata in GEOMETRY_MODES.items():
        hc_g, mr_g, shc_g, smr_g = predict_geom_with_uncertainty(
            d_nm, mat_id, gid, models, T=T)
        geom_rows.append((gdata['name'], hc_g, shc_g, mr_g, smr_g,
                          gdata['factor_hc'], gdata['factor_mr']))

    meta = (f"Material={mat['name']}; Diameter={d_nm:.1f}nm; "
            f"Geometry={gm['name']}; T={T:.0f}K; "
            f"Hc={Hc:.2f}mT; Mr={Mr:.4f}; "
            f"K1={mat['params']['K1_kJ_m3']}kJ/m3; "
            f"A={mat['params']['A_pJ_m']}pJ/m; "
            f"Ms={mat['params']['Ms_MA_m']}MA/m")

    lines: list[str] = []

    # ─ Bloque 1: Histéresis ──────────────────────────────────────────────────
    lines += [
        '! OriginLab import file — Simulador Micromagnético ML',
        f'! Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
        f'! {meta}',
        '',
        '[Hysteresis Loop]',
        # Long Name  (fila 1)
        'H_applied\tM_Ms_ascending\tM_Ms_descending',
        # Units (fila 2)
        'mT\t\t',
        # Comments (fila 3)
        f'{meta}\t\t',
    ]
    for i in range(len(H)):
        lines.append(f'{H[i]:.4f}\t{M_up[i]:.6f}\t{M_dn[i]:.6f}')

    # ─ Bloque 2: Paisaje de energía ─────────────────────────────────────────
    lines += [
        '',
        '[Energy Landscape]',
        'H_applied\tE_Zeeman\tE_Exchange\tE_Demag\tE_Anisotropy',
        'mT\tu.a.\tu.a.\tu.a.\tu.a.',
        f'{meta}\t\t\t\t',
    ]
    H_en = en['H']
    for i in range(len(H_en)):
        lines.append(
            f'{H_en[i]:.4f}\t{en["zeeman"][i]:.6f}\t'
            f'{en["exchange"][i]:.6f}\t{en["demag"][i]:.6f}\t'
            f'{en["aniso"][i]:.6f}')

    # ─ Bloque 3: Tabla de geometrías ─────────────────────────────────────────
    lines += [
        '',
        '[Geometry Comparison]',
        'Geometry\tHc_mT\tsigma_Hc_mT\tMr_Ms\tsigma_Mr\tf_Hc\tf_Mr',
        '\tmT\tmT\t\t\t\t',
        f'd={d_nm:.1f}nm; T={T:.0f}K; Material={mat["name"]}\t\t\t\t\t\t',
    ]
    for row in geom_rows:
        lines.append(
            f'{row[0]}\t{row[1]:.2f}\t{row[2]:.2f}\t'
            f'{row[3]:.4f}\t{row[4]:.4f}\t{row[5]:.3f}\t{row[6]:.3f}')

    return '\n'.join(lines).encode('utf-8')


# ═══════════════════════════════════════════════════════════════════════════════
#  SESSION STATE
# ═══════════════════════════════════════════════════════════════════════════════
defaults = {
    'history':       [],
    'sim_done':      False,
    'mat_id':        'fe',
    'geom_id':       'sphere',
    'd_nm':          30,
    '_last_db_key':  '',
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

# ═══════════════════════════════════════════════════════════════════════════════
#  CARGA DE MODELOS
# ═══════════════════════════════════════════════════════════════════════════════
with st.spinner('⚙️ Entrenando modelos GBR para los 8 materiales…'):
    MODELS = load_all_models()

# ═══════════════════════════════════════════════════════════════════════════════
#  BARRA LATERAL — CONTROLES COMPLETOS
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('## 🔬 Simulador\nMicromagnético ML')
    st.caption('© 2026 SimuGOD. Todos los derechos reservados Jesus Cabezas - Arnol Pérez')
    st.divider()

    # ── Material ─────────────────────────────────────────────────────────────
    st.markdown('### 🧲 Material')
    mat_id = st.selectbox(
        'Material',
        list(MATERIALS_DB.keys()),
        index=list(MATERIALS_DB.keys()).index(st.session_state.mat_id),
        format_func=lambda x: f"{MATERIALS_DB[x]['emoji']}  {MATERIALS_DB[x]['name']}",
        label_visibility='collapsed',
        key='sb_material',
    )
    st.session_state.mat_id = mat_id
    mat   = MATERIALS_DB[mat_id]
    lo, hi = mat['range']

    with st.expander('ℹ️ Descripción y parámetros físicos'):
        st.caption(mat['description'])
        p = mat['params']
        st.markdown(f"""
| Parámetro | Valor |
|---|---|
| Categoría | {mat['category']} |
| K₁ (kJ/m³) | {p['K1_kJ_m3']} |
| A (pJ/m) | {p['A_pJ_m']} |
| Ms (MA/m) | {p['Ms_MA_m']} |
| α | {p['alpha']} |
| λₑₓ (nm) | {p['lambda_ex_nm']} |
| Tc (K) | {p['Tc_K']} |
| Rango ML | {lo}–{hi} nm |
        """)

    st.divider()

    # ── Geometría ─────────────────────────────────────────────────────────────
    st.markdown('### 📐 Geometría')
    geom_id = st.selectbox(
        'Geometría',
        list(GEOMETRY_MODES.keys()),
        index=list(GEOMETRY_MODES.keys()).index(st.session_state.geom_id),
        format_func=lambda x: f"{GEOMETRY_MODES[x]['emoji']}  {GEOMETRY_MODES[x]['name']}",
        label_visibility='collapsed',
        key='sb_geom',
    )
    st.session_state.geom_id = geom_id
    gm = GEOMETRY_MODES[geom_id]
    st.caption(gm['desc'])
    st.markdown(f'Factor Hc: **×{gm["factor_hc"]}** &nbsp;|&nbsp; Factor Mr: **×{gm["factor_mr"]}**')

    st.divider()

    # ── Tamaño ────────────────────────────────────────────────────────────────
    st.markdown('### 📏 Tamaño de partícula (nm)')
    d_nm = st.slider(
        'Tamaño',
        min_value=max(2, lo - 15), max_value=hi + 15,
        value=int(st.session_state.d_nm),
        step=1, label_visibility='collapsed', key='sb_size',
    )
    st.session_state.d_nm = d_nm
    if is_extrapolation(d_nm, mat_id):
        st.warning(f'⚠️ Extrapolación — fuera de [{lo}–{hi} nm]')

    st.divider()

    # ── Comparación ──────────────────────────────────────────────────────────
    st.markdown('### 🔄 Comparar')
    compare_enabled = st.toggle('Activar overlay')
    compare_mat, compare_geom = None, None
    if compare_enabled:
        c_opts = {k: v for k, v in MATERIALS_DB.items() if k != mat_id}
        compare_mat = st.selectbox(
            'Material 2',
            list(c_opts.keys()),
            format_func=lambda x: f"{MATERIALS_DB[x]['emoji']}  {MATERIALS_DB[x]['name']}",
            key='sb_cmat',
        )
        compare_geom = st.selectbox(
            'Geometría 2',
            list(GEOMETRY_MODES.keys()),
            format_func=lambda x: f"{GEOMETRY_MODES[x]['emoji']}  {GEOMETRY_MODES[x]['name']}",
            key='sb_cgeom',
        )

    st.divider()

    # ── 🌡 Temperatura de simulación ──────────────────────────────────────────
    st.markdown('### 🌡 Temperatura')
    _Tc_mat  = mat['params']['Tc_K']
    _T_max   = int(min(_Tc_mat - 1, 1400))

    # ── Toggle K / °C ─────────────────────────────────────────────────────────
    if 'T_unit_celsius' not in st.session_state:
        st.session_state['T_unit_celsius'] = False

    _unit_is_C = st.session_state['T_unit_celsius']

    # Rangos en ambas unidades (definidos antes del botón para usarlos al togglear)
    _K_min, _K_max   = 1, _T_max
    _C_min, _C_max   = _K_min - 273, _K_max - 273   # -272 °C … Tc-273 °C

    _btn_label = '🌡 Ver en Kelvin (K)' if _unit_is_C else '🌡 Ver en Celsius (°C)'
    if st.button(_btn_label, use_container_width=True, key='btn_toggle_tunit'):
        _cur_K = int(np.clip(st.session_state.get('T_sim', 300), _K_min, _K_max))
        if not _unit_is_C:
            # K → °C: sincronizar la key del slider °C antes del rerun
            st.session_state['sb_Tsim_C'] = int(np.clip(_cur_K - 273, _C_min, _C_max))
        else:
            # °C → K: sincronizar la key del slider K antes del rerun
            st.session_state['sb_Tsim'] = _cur_K
        st.session_state['T_unit_celsius'] = not _unit_is_C
        st.rerun()

    # ── Slider dinámico (la key pre-seteada arriba garantiza valor correcto) ──
    _T_cur_K = int(np.clip(st.session_state.get('T_sim', 300), _K_min, _K_max))

    if st.session_state['T_unit_celsius']:
        _c_init  = int(np.clip(_T_cur_K - 273, _C_min, _C_max))
        # Asegurar que la key existe y está en rango antes de renderizar
        if 'sb_Tsim_C' not in st.session_state:
            st.session_state['sb_Tsim_C'] = _c_init
        _T_C_raw = st.slider(
            'T de simulación (°C)',
            min_value=_C_min, max_value=_C_max,
            value=st.session_state['sb_Tsim_C'],
            step=10, key='sb_Tsim_C',
        )
        T_sim = int(np.clip(_T_C_raw + 273, _K_min, _K_max))
    else:
        if 'sb_Tsim' not in st.session_state:
            st.session_state['sb_Tsim'] = _T_cur_K
        T_sim = st.slider(
            'T de simulación (K)',
            min_value=_K_min, max_value=_K_max,
            value=st.session_state['sb_Tsim'],
            step=10, key='sb_Tsim',
        )

    st.session_state['T_sim'] = T_sim

    # ── Resumen dual siempre visible ──────────────────────────────────────────
    _T_C   = T_sim - 273.15
    _T_red = T_sim / _Tc_mat
    _Tc_C  = _Tc_mat - 273.15

    if st.session_state['T_unit_celsius']:
        st.caption(f'**{_T_C:.0f} °C**  ·  {T_sim} K · T/Tc = {_T_red:.3f}')
        st.caption(f'Tc ({mat["name"]}) = **{_Tc_C:.0f} °C** / {_Tc_mat} K')
    else:
        st.caption(f'**{T_sim} K**  ·  {_T_C:.0f} °C · T/Tc = {_T_red:.3f}')
        st.caption(f'Tc ({mat["name"]}) = **{_Tc_mat} K** / {_Tc_C:.0f} °C')

    if _T_red > 0.80:
        st.warning('⚠️ T > 0.8·Tc — propiedades muy reducidas')
    elif _T_red > 0.50:
        st.info('ℹ️ T > 0.5·Tc — efectos térmicos significativos')

    with st.expander('📉 Propiedades efectivas a T'):
        _tau  = float(np.clip(T_sim / _Tc_mat, 0.0, 0.9999))
        _ms_t = float(np.clip((1 - _tau**1.5)**(1/3), 0.0, 1.0))
        _k1_t = float(np.clip(_ms_t**(10/3), 0.0, 1.0))
        _a_t  = float(np.clip(_ms_t**2, 0.0, 1.0))
        _p    = mat['params']
        _T_label = f'{_T_C:.0f} °C' if st.session_state['T_unit_celsius'] else f'{T_sim} K'
        st.markdown(f"""
| Propiedad | T=0 K | T={_T_label} |
|---|---|---|
| Ms (MA/m) | {_p['Ms_MA_m']:.3f} | {_p['Ms_MA_m']*_ms_t:.3f} |
| K₁ (kJ/m³)| {_p['K1_kJ_m3']:.1f} | {_p['K1_kJ_m3']*_k1_t:.1f} |
| A (pJ/m)  | {_p['A_pJ_m']:.2f}  | {_p['A_pJ_m']*_a_t:.2f}  |
| Ms(T)/Ms₀ | 1.000 | {_ms_t:.4f} |
| K₁(T)/K₁₀| 1.000 | {_k1_t:.4f} |
""")
        st.caption('Bloch (1930) · Callen-Callen (1966) · Anderson scaling')

    st.divider()

    # ── Config avanzada ──────────────────────────────────────────────────────
    with st.expander('⚙️ Config. avanzada'):
        noise_level = st.slider('Ruido LLG', 0.0, 0.05, 0.008, 0.001,
                                 format='%.3f')
        export_dpi  = st.select_slider('DPI', [100, 150, 170, 200, 300], value=170)

    st.divider()

    # ── Botones ───────────────────────────────────────────────────────────────
    btn_sim     = st.button('▶ SIMULAR', use_container_width=True, type='primary')
    btn_animate = st.button('🎬 Animar por tamaño', use_container_width=True)
    btn_clear   = st.button('🗑 Limpiar historial', use_container_width=True)
    if btn_clear:
        st.session_state.history = []
        _db.clear_simulations()
        st.rerun()
    if btn_sim:
        st.session_state.sim_done = True

    st.divider()

    # ── 📂 Carga de datos OOMMF — siempre visible en sidebar ─────────────────
    st.markdown('### 📂 Datos OOMMF')

    # Resumen del dataset actual
    # _APP_DIR ya está en sys.path (inyectado al inicio), import siempre funciona
    import oommf_data_manager as _odm_sb

    _sb_data_dir = _odm_sb._DEFAULT_DATA_DIR   # ruta canónica oommf_data/
    _sb_n_h = _sb_n_e = _sb_n_nb = _sb_n_cal = 0
    _sb_summary_err = ''

    try:
        _sb_sum  = _odm_sb.dataset_summary(_sb_data_dir)
        _sb_n_h  = _sb_sum.get('n_hysteresis', 0)
        _sb_n_e  = _sb_sum.get('n_energies', 0)
        _sb_n_nb = _sb_sum.get('n_notebooks', 0)
        _sb_n_cal= _sb_sum.get('n_calibration', 0)
    except Exception as _e_sum:
        _sb_summary_err = str(_e_sum)

    # Badge de estado
    if _sb_n_h > 0 or _sb_n_e > 0:
        st.markdown(
            f'<div style="background:#134e4a;border:1px solid #14b8a6;'
            f'border-radius:8px;padding:8px 10px;font-size:0.78rem;color:#ccfbf1;">'
            f'✅ <b>{_sb_n_h}</b> histéresis &nbsp;·&nbsp; '
            f'<b>{_sb_n_e}</b> energías<br>'
            f'<span style="color:#5eead4;">📓 {_sb_n_nb} notebooks '
            f'· 🎯 {_sb_n_cal} calibraciones</span></div>',
            unsafe_allow_html=True,
        )
    elif _sb_summary_err:
        st.markdown(
            f'<div style="background:#2d1515;border:1px solid #ef4444;'
            f'border-radius:8px;padding:8px 10px;font-size:0.74rem;color:#fca5a5;">'
            f'⚠️ Error al leer datos:<br>'
            f'<code style="font-size:0.68rem">{_sb_summary_err[:120]}</code></div>',
            unsafe_allow_html=True,
        )
    else:
        _n_files = len(list(_os_sb.scandir(_sb_data_dir))) if _os_sb.path.isdir(_sb_data_dir) else 0
        st.markdown(
            f'<div style="background:#1c1917;border:1px solid #78350f;'
            f'border-radius:8px;padding:8px 10px;font-size:0.78rem;color:#fde68a;">'
            f'📂 Sin datos OOMMF ({_n_files} archivo{"s" if _n_files!=1 else ""} en carpeta)<br>'
            f'<span style="color:#a78bfa;">Sube archivos abajo ↓</span></div>',
            unsafe_allow_html=True,
        )
    _sb_has_data = True   # módulo siempre disponible; carga independiente del summary

    st.markdown('')

    # Uploader compacto
    _sb_files = st.file_uploader(
        'Arrastra aquí tus archivos',
        type=['txt', 'ipynb'],
        accept_multiple_files=True,
        key='sb_oommf_uploader',
        help='Archivos .txt (fd/mg) de energías o histéresis, o .ipynb de simulación',
        label_visibility='collapsed',
    )

    # Opciones de etiquetado (colapsadas para no saturar)
    with st.expander('⚙️ Opciones de carga'):
        _sb_mat = st.selectbox(
            'Material',
            ['— inferir —'] + list(MATERIALS_DB.keys()),
            format_func=lambda x: (
                '— inferir del archivo —' if x == '— inferir —'
                else f"{MATERIALS_DB[x]['emoji']} {MATERIALS_DB[x]['name']}"
            ),
            key='sb_up_mat',
        )
        _sb_d = st.number_input(
            'Diámetro (nm)', min_value=1.0, max_value=500.0,
            value=42.0, step=1.0, key='sb_up_d',
            help='Diámetro de la nanopartícula simulada',
        )
        _sb_geom_up = st.selectbox(
            'Geometría',
            list(GEOMETRY_MODES.keys()),
            format_func=lambda g: f"{GEOMETRY_MODES[g]['emoji']} {GEOMETRY_MODES[g]['name']}",
            key='sb_up_geom',
        )

    # Procesar archivos subidos
    if _sb_files and _sb_has_data:
        _sb_mat_id = None if _sb_mat == '— inferir —' else _sb_mat
        _any_hyst  = False
        for _sbf in _sb_files:
            _tmp_sb = f'/tmp/_sb_upload_{_sbf.name}'
            with open(_tmp_sb, 'wb') as _fh:
                _fh.write(_sbf.getbuffer())
            try:
                _r_sb = _odm_sb.ingest_uploaded_file(
                    src_path=_tmp_sb,
                    data_dir=_sb_data_dir,
                    mat_id=_sb_mat_id,
                    d_nm=float(_sb_d),
                    geom_id=_sb_geom_up,
                )
                if _r_sb.get('status') == 'ok':
                    _dtype_sb = _r_sb.get('dtype', '?')
                    _hp_sb    = _r_sb.get('hyst_params', {})
                    if _hp_sb:
                        st.success(
                            f'✅ **{_sbf.name}**\n\n'
                            f'Tipo: `{_dtype_sb}`\n\n'
                            f'Hc = **{_hp_sb["Hc_mT"]:.1f} mT** · '
                            f'Mr = **{_hp_sb["Mr_Ms"]:.4f}**'
                        )
                        if _r_sb.get('calibration_saved'):
                            st.caption(
                                f'🎯 Calibración guardada: '
                                f'{_sb_d:.0f} nm · {_sb_mat_id or "auto"}'
                            )
                        _any_hyst = True
                    else:
                        st.success(
                            f'✅ **{_sbf.name}**\n\nTipo: `{_dtype_sb}`'
                        )
                else:
                    st.error(f'❌ {_sbf.name}: {_r_sb.get("message","Error")}')
            except Exception as _e_sb:
                st.error(f'❌ {_sbf.name}: {_e_sb}')

        # Limpiar caches de visualización afectados
        for _ck in list(st.session_state.keys()):
            if _ck.startswith('fig_energy_real') or _ck.startswith('oommf_res_'):
                del st.session_state[_ck]

        if _any_hyst:
            st.info(
                '🔄 Ve a **⚡ Componentes de Energía** y activa el toggle '
                '"🔬 Mostrar datos reales OOMMF" para ver los datos recién cargados.'
            )

# ═══════════════════════════════════════════════════════════════════════════════
#  PANTALLA PRINCIPAL — HOME CARD  (Dashboard único)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Header ────────────────────────────────────────────────────────────────────
col_title, col_status = st.columns([3, 1])
with col_title:
    st.title('🔬 Simulador Micromagnético ML - Opcion Grado')
with col_status:
    status_text = '🟢 Simulación lista' if st.session_state.sim_done else '⚪ Sin simular'
    st.markdown(f"""
<div style='text-align:right;padding-top:1.2rem;font-size:0.8rem;color:#64748b;'>
  v4.0 · Fase 4 · Ensemble ML<br>{status_text}<br>
  {mat['emoji']} {mat['name']}<br>
  {gm['emoji']} {gm['name']} · {d_nm} nm
</div>""", unsafe_allow_html=True)

# ── HOME CARD — configuración actual + acción rápida ─────────────────────────
with st.container():
    st.markdown('---')
    c1, c2, c3, c4, c5 = st.columns(5)

    c1.markdown(f"""
<div style='background:#1e293b;border:1px solid {mat["color"]};border-radius:10px;
     padding:14px;text-align:center;'>
  <div style='font-size:2rem;'>{mat["emoji"]}</div>
  <div style='color:#f1f5f9;font-weight:600;font-size:0.9rem;'>{mat["name"]}</div>
  <div style='color:#64748b;font-size:0.75rem;'>{mat["category"]}</div>
</div>""", unsafe_allow_html=True)

    c2.markdown(f"""
<div style='background:#1e293b;border:1px solid #334155;border-radius:10px;
     padding:14px;text-align:center;'>
  <div style='font-size:2rem;'>{gm["emoji"]}</div>
  <div style='color:#f1f5f9;font-weight:600;font-size:0.9rem;'>{gm["name"]}</div>
  <div style='color:#64748b;font-size:0.75rem;'>f_Hc × {gm["factor_hc"]}</div>
</div>""", unsafe_allow_html=True)

    c3.markdown(f"""
<div style='background:#1e293b;border:1px solid #334155;border-radius:10px;
     padding:14px;text-align:center;'>
  <div style='font-size:2rem;'>📏</div>
  <div style='color:#38bdf8;font-weight:700;font-size:1.4rem;'>{d_nm} nm</div>
  <div style='color:#64748b;font-size:0.75rem;'>Rango ML: {lo}–{hi} nm</div>
</div>""", unsafe_allow_html=True)

    Hc_prev, Mr_prev, sHc_prev, sMr_prev = predict_geom_with_uncertainty(
        d_nm, mat_id, geom_id, MODELS, T=float(st.session_state.get('T_sim', 300)))
    c4.markdown(f"""
<div style='background:#1e293b;border:1px solid #334155;border-radius:10px;
     padding:14px;text-align:center;'>
  <div style='font-size:2rem;'>⚡</div>
  <div style='color:#38bdf8;font-weight:700;font-size:1.2rem;'>{Hc_prev:.0f} mT</div>
  <div style='color:#64748b;font-size:0.75rem;'>±{sHc_prev:.0f} mT · Ensemble</div>
</div>""", unsafe_allow_html=True)

    c5.markdown(f"""
<div style='background:#1e293b;border:1px solid #334155;border-radius:10px;
     padding:14px;text-align:center;'>
  <div style='font-size:2rem;'>🧲</div>
  <div style='color:#34d399;font-weight:700;font-size:1.2rem;'>{Mr_prev:.3f}</div>
  <div style='color:#64748b;font-size:0.75rem;'>±{sMr_prev:.3f} · Ensemble</div>
</div>""", unsafe_allow_html=True)

    st.markdown('---')

# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURA PRINCIPAL (calcula una vez)
# ═══════════════════════════════════════════════════════════════════════════════
if not st.session_state.sim_done:
    st.info('👈 Configura el material, geometría y tamaño en la barra lateral, '
            'luego presiona **▶ SIMULAR**.')
    st.stop()

t0 = time.perf_counter()
fig_main, Hc_val, Mr_val = build_main_figure(
    mat_id, d_nm, geom_id, MODELS,
    noise_level=noise_level, dpi=export_dpi,
    compare_mat=compare_mat if compare_enabled else None,
    compare_geom=compare_geom if compare_enabled else None,
    T=float(st.session_state.get('T_sim', 300)),
)
elapsed_ms = (time.perf_counter() - t0) * 1000

# ── Registro en historial + SQLite ────────────────────────────────────────────
entry = {
    'Hora': datetime.now().strftime('%H:%M:%S'),
    'Material': mat['name'], 'Geometría': gm['name'],
    'Tamaño (nm)': d_nm,
    'Hc (mT)': round(Hc_val, 1), 'Mr/Ms': round(Mr_val, 3),
    'Extrapolación': '⚠️' if is_extrapolation(d_nm, mat_id) else '✓',
}
_last_key = f'{mat_id}_{geom_id}_{d_nm}'
if (not st.session_state.history or
        st.session_state.history[-1].get('Tamaño (nm)') != d_nm or
        st.session_state.history[-1].get('Material') != mat['name'] or
        st.session_state.history[-1].get('Geometría') != gm['name']):
    st.session_state.history.append(entry)

if st.session_state.get('_last_db_key') != _last_key:
    _db.save_simulation(
        material=mat['name'], material_id=mat_id, size_nm=d_nm,
        geometry=geom_id,
        hc_sphere=Hc_val, mr_sphere=Mr_val,
        hc_cuboid=None,   mr_cuboid=None,
        noise_level=noise_level, field_max=mat['field_max'],
        extrapolation=is_extrapolation(d_nm, mat_id),
    )
    # ── Fase 4: feedback online — cada simulación mejora el modelo ───────────
    MODELS.add_feedback(mat_id, d_nm, Hc_val, Mr_val)
    st.session_state['_last_db_key'] = _last_key

# ═══════════════════════════════════════════════════════════════════════════════
#  BANNER DE DATOS OOMMF — aparece ANTES de los tabs
# ═══════════════════════════════════════════════════════════════════════════════
try:
    import oommf_data_manager as _odm_banner
    _bn_sum  = _odm_banner.dataset_summary()
    _bn_n_h  = _bn_sum.get('n_hysteresis', 0)
    _bn_n_e  = _bn_sum.get('n_energies',   0)
    _bn_n_nb = _bn_sum.get('n_notebooks',  0)
    _bn_n_cal= _bn_sum.get('n_calibration',0)
    _bn_dir  = _bn_sum.get('data_dir', 'oommf_data/')

    if _bn_n_h > 0 or _bn_n_e > 0:
        # ── Dataset cargado: mostrar resumen compacto ─────────────────────────
        with st.container():
            _bn_hc_r = _bn_sum.get('hc_range', (0, 0))
            _bn_mr_r = _bn_sum.get('mr_range', (0, 0))
            _bn_mats = ', '.join(_bn_sum.get('materials', ['?'])) or '—'
            _b1, _b2, _b3, _b4, _b5 = st.columns([2, 1, 1, 1, 1])
            _b1.markdown(
                f'<div style="background:#0d3329;border:1px solid #14b8a6;'
                f'border-radius:8px;padding:10px 14px;">'
                f'<span style="color:#14b8a6;font-weight:700;font-size:0.9rem;">'
                f'📂 Datos OOMMF activos</span><br>'
                f'<span style="color:#5eead4;font-size:0.78rem;">'
                f'Carpeta: <code>oommf_data/</code> · '
                f'Material(es): <b>{_bn_mats}</b></span></div>',
                unsafe_allow_html=True,
            )
            _b2.metric('Histéresis', _bn_n_h)
            _b3.metric('Energías',   _bn_n_e)
            _b4.metric('Notebooks',  _bn_n_nb)
            _b5.metric('Cal. ML',    _bn_n_cal)

            # Si hay histéresis, mostrar Hc/Mr del primer dataset
            if _bn_n_h > 0:
                _bn_hds = _odm_banner.scan_datasets(_bn_dir)['hysteresis']
                _hc_lbl = '  ·  '.join(
                    f"{h['filename'].split('.')[0]}: Hc={h.get('Hc_mT','?'):.1f} mT  Mr={h.get('Mr_Ms','?'):.4f}"
                    for h in _bn_hds
                )
                st.caption(f'🔬 {_hc_lbl}  ·  '
                           f'Hc rango [{_bn_hc_r[0]:.1f}–{_bn_hc_r[1]:.1f}] mT  '
                           f'Mr rango [{_bn_mr_r[0]:.4f}–{_bn_mr_r[1]:.4f}]')
    else:
        # ── Sin datos: invitación prominente a cargar ─────────────────────────
        st.markdown(
            '<div style="background:#1c1917;border:1px dashed #78350f;'
            'border-radius:10px;padding:14px 18px;margin-bottom:6px;">'
            '<span style="color:#fbbf24;font-weight:700;">📂 Sin datos OOMMF</span>'
            '&nbsp;&nbsp;'
            '<span style="color:#a3a3a3;font-size:0.85rem;">'
            'Sube archivos <code>.txt</code> (fd/mg) o <code>.ipynb</code> '
            'desde la barra lateral para activar la visualización de energías reales '
            'y la calibración ML.</span></div>',
            unsafe_allow_html=True,
        )
except Exception:
    pass

# ═══════════════════════════════════════════════════════════════════════════════
#  TABS DE RESULTADOS
# ═══════════════════════════════════════════════════════════════════════════════
(tab_sim, tab_compare, tab_params,
 tab_3d, tab_dashboard, tab_export, tab_uval) = st.tabs([
    '📊 Simulación',
    '🔄 Comparar',
    '📋 Parámetros ML',
    '🌐 3D Interactivo',
    '📈 Dashboard',
    '💾 Exportar',
    '🧲 Validación Ubermag',
])

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — SIMULACIÓN
# ─────────────────────────────────────────────────────────────────────────────
with tab_sim:
    if btn_animate:
        steps = list(range(lo, hi + 1, max(1, (hi - lo) // 25)))
        ph    = st.empty(); progress = st.progress(0)
        for i, s in enumerate(steps):
            f_a, _, _ = build_main_figure(mat_id, s, geom_id, MODELS,
                                           noise_level=noise_level, dpi=100)
            buf = io.BytesIO()
            f_a.savefig(buf, format='png', dpi=100,
                        bbox_inches='tight', facecolor='#0f172a')
            plt.close(f_a); buf.seek(0)
            ph.image(buf, use_column_width=True,
                     caption=f'🎬 {s} nm — {gm["name"]}')
            progress.progress((i + 1) / len(steps))
            time.sleep(0.10)
        progress.empty()
    else:
        st.pyplot(fig_main, use_container_width=True)

    st.divider()
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric(f'{gm["emoji"]} Hc (mT)',       f'{Hc_val:.1f}')
    m2.metric('Mr/Ms',                          f'{Mr_val:.3f}')
    m3.metric('Campo máximo (mT)',              mat['field_max'])
    m4.metric('Factor forma Hc',               f'×{gm["factor_hc"]}')
    m5.metric('⏱ Cómputo',                     f'{elapsed_ms:.0f} ms')

    # ── Exportar a OriginLab ───────────────────────────────────────────────────
    with st.expander('📤 Exportar para OriginLab', expanded=False):
        st.markdown(
            'Descarga los datos de simulación en formato **tab-delimited** '
            'compatible con **OriginLab 8+**.\n\n'
            'Incluye: histéresis (ambas ramas) · paisaje de energía · '
            'tabla comparativa de geometrías con incertidumbre ±σ.\n\n'
            '**Cómo importar en Origin:** `File → Import → ASCII` y '
            'activar *Read Long Name / Units / Comments from rows*.'
        )
        _T_cur = float(st.session_state.get('T_sim', 300))
        _origin_bytes = export_to_originlab(
            mat_id, d_nm, geom_id, Hc_val, Mr_val, _T_cur, MODELS)
        _origin_fname = (f'SimuMag_{mat["name"].replace(" ","_")}_'
                         f'{d_nm:.0f}nm_{geom_id}_{_T_cur:.0f}K.txt')
        st.download_button(
            label='⬇️  Descargar archivo OriginLab (.txt)',
            data=_origin_bytes,
            file_name=_origin_fname,
            mime='text/plain',
            use_container_width=True,
        )
        st.caption(
            '💡 En OriginLab puedes generar gráficas de publicación con '
            'el mismo dataset usando *Plot → Line*, *Plot → Scatter*, etc.'
        )

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — COMPARAR  (material + geometría)
# ─────────────────────────────────────────────────────────────────────────────
with tab_compare:
    st.subheader('Comparación Material × Geometría')
    cc1, cc2 = st.columns(2)
    all_mats = list(MATERIALS_DB.keys())
    all_geoms = list(GEOMETRY_MODES.keys())

    sel_a_mat  = cc1.selectbox('Material A', all_mats, index=0,
                                format_func=lambda x: f"{MATERIALS_DB[x]['emoji']}  {MATERIALS_DB[x]['name']}",
                                key='c_amat')
    sel_a_geom = cc1.selectbox('Geometría A', all_geoms, index=0,
                                format_func=lambda x: f"{GEOMETRY_MODES[x]['emoji']}  {GEOMETRY_MODES[x]['name']}",
                                key='c_ageom')
    sel_b_mat  = cc2.selectbox('Material B', all_mats, index=1,
                                format_func=lambda x: f"{MATERIALS_DB[x]['emoji']}  {MATERIALS_DB[x]['name']}",
                                key='c_bmat')
    sel_b_geom = cc2.selectbox('Geometría B', all_geoms, index=1,
                                format_func=lambda x: f"{GEOMETRY_MODES[x]['emoji']}  {GEOMETRY_MODES[x]['name']}",
                                key='c_bgeom')
    c_size = st.slider('Tamaño para comparar (nm)', 5, 150, 30, key='c_sz')

    plt.rcParams.update(_DARK)
    fig_cmp, axes_cmp = plt.subplots(1, 2, figsize=(14, 4.5), facecolor='#0f172a')
    for ax, sel_mat, sel_geom in zip(axes_cmp,
                                      [sel_a_mat, sel_b_mat],
                                      [sel_a_geom, sel_b_geom]):
        m    = MATERIALS_DB[sel_mat]
        gd   = GEOMETRY_MODES[sel_geom]
        lo_c, hi_c = m['range']
        d_c  = min(max(c_size, lo_c), hi_c)
        Hc_c, Mr_c = predict_geom(d_c, sel_mat, sel_geom, MODELS)
        H, M_up, M_dn = llg_hysteresis(Hc_c, Mr_c, H_max=m['field_max'], seed=42)
        ax.plot(H, M_up, color=m['color'], lw=1.8,
                label=f'{gd["name"]}  Hc={Hc_c:.0f} mT')
        ax.plot(H, M_dn, color=m['color'], lw=1.5, alpha=0.6, ls='--')
        ax.set_title(f'{m["name"]}  ·  {gd["emoji"]} {gd["name"]}  @  {d_c:.0f} nm',
                     fontsize=9)
        ax.set_xlabel('H (mT)'); ax.set_ylabel('M/Ms')
        ax.legend(fontsize=7); ax.grid(True)
        ax.axhline(0, color='#475569', lw=0.5)
    fig_cmp.tight_layout(pad=1.5)
    st.pyplot(fig_cmp, use_container_width=True)
    plt.close(fig_cmp)

    # Superposición normalizada
    st.subheader('Superposición normalizada H/H_max')
    fig_ov, ax_ov = plt.subplots(figsize=(12, 4), facecolor='#0f172a')
    ax_ov.set_facecolor('#1e293b')
    for sel_mat, sel_geom in [(sel_a_mat, sel_a_geom), (sel_b_mat, sel_b_geom)]:
        m    = MATERIALS_DB[sel_mat]
        gd   = GEOMETRY_MODES[sel_geom]
        lo_c, hi_c = m['range']
        d_c  = min(max(c_size, lo_c), hi_c)
        Hc_c, Mr_c = predict_geom(d_c, sel_mat, sel_geom, MODELS)
        H, M_up, M_dn = llg_hysteresis(Hc_c, Mr_c, H_max=m['field_max'], seed=42)
        H_norm = H / m['field_max']
        ax_ov.plot(H_norm, M_up, color=m['color'], lw=2.2,
                   label=f'{m["name"]} · {gd["name"]}  Hc={Hc_c:.0f} mT')
        ax_ov.plot(H_norm, M_dn, color=m['color'], lw=1.6, ls='--', alpha=0.65)
    ax_ov.axhline(0, color='#475569', lw=0.5)
    ax_ov.set_xlabel('H / H_max  (normalizado)'); ax_ov.set_ylabel('M / Ms')
    ax_ov.set_title('Comparación directa normalizada', fontsize=9)
    ax_ov.legend(fontsize=8); ax_ov.grid(True, alpha=0.35)
    fig_ov.tight_layout()
    st.pyplot(fig_ov, use_container_width=True)
    plt.close(fig_ov)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — PARÁMETROS ML FASE 4  (ensemble · features · online learning)
# ─────────────────────────────────────────────────────────────────────────────
with tab_params:
    st.subheader(f'📋 Parámetros ML Fase 4 — {mat["name"]}')

    # ── Subtabs del panel ML ──────────────────────────────────────────────────
    ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs([
        '📊 Métricas del Ensemble',
        '🔍 Importancia de Features',
        '📈 Curvas con Incertidumbre',
        '🔄 Aprendizaje Online',
    ])

    eng_metrics = MODELS.get_metrics(mat_id)

    # ── SUB-TAB 1: Métricas ───────────────────────────────────────────────────
    with ml_tab1:
        st.markdown('#### Comparación de modelos — R² (CV 5-fold) y RMSE')

        p = mat['params']
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        col_p1.metric('K₁ (kJ/m³)', p['K1_kJ_m3'])
        col_p1.metric('A (pJ/m)',    p['A_pJ_m'])
        col_p2.metric('Ms (MA/m)',   p['Ms_MA_m'])
        col_p2.metric('α (LLG)',     p['alpha'])
        col_p3.metric('λₑₓ (nm)',    p['lambda_ex_nm'])
        col_p3.metric('Tc (K)',      p['Tc_K'])
        col_p4.metric('Pts. entren.', eng_metrics.get('n_train', '—'))
        col_p4.metric('Feedback',    eng_metrics.get('n_feedback', 0))
        st.divider()

        # Tabla R²
        r2_hc = eng_metrics.get('r2_cv_hc', {})
        r2_mr = eng_metrics.get('r2_cv_mr', {})
        rmse_hc = eng_metrics.get('rmse_hc', {})
        rmse_mr = eng_metrics.get('rmse_mr', {})
        w_hc    = eng_metrics.get('weights_hc', np.array([1/3]*3))
        w_mr    = eng_metrics.get('weights_mr', np.array([1/3]*3))

        rows_metrics = []
        for i, name in enumerate(MODELS.MODEL_NAMES):
            rows_metrics.append({
                'Modelo':       name,
                'R² CV  Hc':    round(r2_hc.get(name, 0), 4),
                'R² CV  Mr':    round(r2_mr.get(name, 0), 4),
                'RMSE Hc (mT)': round(rmse_hc.get(name, 0), 2),
                'RMSE Mr':      round(rmse_mr.get(name, 0), 5),
                'Peso Hc':      f'{w_hc[i]:.3f}',
                'Peso Mr':      f'{w_mr[i]:.3f}',
            })
        st.dataframe(pd.DataFrame(rows_metrics),
                     use_container_width=True, hide_index=True)

        # Gráfica R² por modelo
        plt.rcParams.update(_DARK)
        fig_r2, axes_r2 = plt.subplots(1, 2, figsize=(11, 3.5), facecolor='#0f172a')
        model_colors = ['#38bdf8', '#34d399', '#f472b6']
        for ax, metric_dict, title in zip(
            axes_r2,
            [r2_hc, r2_mr],
            ['R² CV — Hc (campo coercitivo)', 'R² CV — Mr/Ms (remanencia)'],
        ):
            ax.set_facecolor('#1e293b')
            names = list(metric_dict.keys())
            vals  = list(metric_dict.values())
            bars  = ax.bar(names, vals, color=model_colors[:len(names)], width=0.5)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2,
                        bar.get_height() + 0.005, f'{v:.3f}',
                        ha='center', va='bottom', fontsize=9, color='#f1f5f9')
            ax.set_ylim(0, 1.05)
            ax.set_ylabel('R²'); ax.set_title(title, fontsize=9)
            ax.grid(True, alpha=0.3, axis='y')
        fig_r2.tight_layout()
        st.pyplot(fig_r2, use_container_width=True)
        plt.close(fig_r2)

        st.divider()
        st.markdown('**Barrido de tamaños — predicciones Ensemble por geometría**')
        sizes_sweep = sorted(set(
            list(range(lo, hi + 1, max(1, (hi - lo) // 10))) + [hi]))
        sizes_sweep_arr = np.array(sizes_sweep, dtype=float)
        # 1 batch call → luego escalar por factor de geometría (antes 8×n calls)
        Hc_sw_base, _ = MODELS.predict_batch(sizes_sweep_arr, mat_id)
        rows_sw = []
        for i, s in enumerate(sizes_sweep):
            row_s = {'Tamaño (nm)': s,
                     'Extrapol.': '⚠️' if is_extrapolation(s, mat_id) else '✓'}
            for gid, gdata in GEOMETRY_MODES.items():
                Hc_g = float(Hc_sw_base[i]) * GEOMETRY_MODES[gid]['factor_hc']
                row_s[f'Hc {gdata["emoji"]}'] = round(Hc_g, 1)
            rows_sw.append(row_s)
        st.dataframe(pd.DataFrame(rows_sw),
                     use_container_width=True, hide_index=True)
        st.info(
            f'📍 Selección: **{d_nm} nm** · **{gm["name"]}** · '
            f'Hc = **{Hc_val:.1f} mT** · Mr/Ms = **{Mr_val:.3f}**'
        )

    # ── SUB-TAB 2: Importancia de features ───────────────────────────────────
    with ml_tab2:
        st.markdown('#### Importancia de features — GradientBoostingRegressor')
        st.caption(
            'Fracción del "information gain" total atribuido a cada feature. '
            'Features físicamente motivados: el modelo aprende qué parámetros '
            'dominan la coercitividad y la remanencia.'
        )
        fi = MODELS.feature_importance(mat_id)
        if fi:
            plt.rcParams.update(_DARK)
            fig_fi, axes_fi = plt.subplots(1, 2, figsize=(12, 4), facecolor='#0f172a')
            fi_colors = ['#38bdf8','#fb923c','#34d399','#f472b6',
                         '#fbbf24','#a78bfa','#f87171']
            for ax, key, title in zip(
                axes_fi,
                ['hc', 'mr'],
                ['Importancia → Hc (campo coercitivo)',
                 'Importancia → Mr/Ms (remanencia)'],
            ):
                ax.set_facecolor('#1e293b')
                imps   = fi[key]
                names  = fi['names']
                idx    = np.argsort(imps)[::-1]
                sorted_names = [names[i] for i in idx]
                sorted_imps  = imps[idx]
                bars = ax.barh(sorted_names, sorted_imps,
                               color=[fi_colors[i % len(fi_colors)] for i in idx])
                for bar, v in zip(bars, sorted_imps):
                    ax.text(v + 0.003, bar.get_y() + bar.get_height() / 2,
                            f'{v:.3f}', va='center', fontsize=8, color='#f1f5f9')
                ax.set_xlabel('Importancia relativa')
                ax.set_title(title, fontsize=9)
                ax.grid(True, alpha=0.3, axis='x')
            fig_fi.suptitle(
                f'GBR Feature Importance — {mat["name"]}',
                fontsize=10, color='#64748b')
            fig_fi.tight_layout()
            st.pyplot(fig_fi, use_container_width=True)
            plt.close(fig_fi)

            st.markdown('**Descripción de los 7 features**')
            feat_desc = {
                'd (nm)':        'Diámetro de partícula — feature base',
                'd / λₑₓ':       'Tamaño normalizado a longitud de intercambio (crítico para SPM)',
                'log₁₀(d)':      'Escala logarítmica — captura leyes de potencia',
                'K₁V / k_BT':    'Barrera energética de anisotropía vs. energía térmica',
                'Ms (MA/m)':     'Magnetización de saturación del material',
                'α (LLG)':       'Parámetro de amortiguamiento Gilbert',
                'T / Tc':        'Temperatura reducida (0 = 0 K, 1 = Curie)',
            }
            for fname, fdesc in feat_desc.items():
                st.markdown(f'  - **{fname}**: {fdesc}')

    # ── SUB-TAB 3: Curvas con bandas de incertidumbre ─────────────────────────
    with ml_tab3:
        st.markdown('#### Hc vs Tamaño — comparación de modelos y banda ±1σ')
        st.caption(
            'Líneas = predicciones individuales (GBR, RF, MLP). '
            'Línea gruesa = Ensemble ponderado. '
            'Banda sombreada = ±1σ estimado por varianza del RandomForest.'
        )

        # Vectorizado: una pasada sklearn por modelo en lugar de 80 predict() en bucle
        sweep_data = MODELS.predict_all_models_sweep(mat_id, n_pts=50)
        sizes_full = sweep_data['sizes']

        plt.rcParams.update(_DARK)
        fig_unc, axes_unc = plt.subplots(1, 2, figsize=(14, 4.5), facecolor='#0f172a')
        m_colors = {'GBR': '#38bdf8', 'RF': '#34d399', 'MLP': '#f472b6'}

        for ax, target, ylabel, std_key in zip(
            axes_unc,
            ['Hc', 'Mr'],
            ['Hc (mT)', 'Mr / Ms'],
            ['Hc_std', 'Mr_std'],
        ):
            ax.set_facecolor('#1e293b')
            ens_arr = sweep_data['Ensemble'][target]
            std_arr = sweep_data[std_key]

            # Banda ±1σ
            ax.fill_between(sizes_full,
                            ens_arr - std_arr, ens_arr + std_arr,
                            alpha=0.18, color='#f1f5f9', label='±1σ (RF)')

            # Modelos individuales
            for nm, col in m_colors.items():
                ax.plot(sizes_full, sweep_data[nm][target], color=col,
                        lw=1.2, ls='--', alpha=0.75, label=nm)

            # Ensemble
            ax.plot(sizes_full, ens_arr, color='#f1f5f9',
                    lw=2.5, label='Ensemble')

            # Ancla: punto actual
            ax.axvline(d_nm, color='#fbbf24', lw=1.0, ls=':', alpha=0.8)
            ax.axvspan(lo, hi, alpha=0.07, color='#38bdf8')

            ax.set_xlabel('Tamaño (nm)'); ax.set_ylabel(ylabel)
            ax.set_title(f'{ylabel} vs Tamaño — {mat["name"]}', fontsize=9)
            ax.legend(fontsize=7, loc='upper right'); ax.grid(True, alpha=0.3)

        fig_unc.tight_layout()
        st.pyplot(fig_unc, use_container_width=True)
        plt.close(fig_unc)

        # Gráfica Hc vs tamaño — todas las geometrías (como antes)
        st.divider()
        st.markdown('**Hc vs Tamaño — todas las geometrías (Ensemble)**')
        # Resolución moderada: max 25 puntos (antes podía llegar a 86 → 688 calls)
        sizes_geom = np.linspace(max(2, lo - 5), hi + 5,
                                 min(25, max(10, (hi - lo) // 4)))
        # 1 batch call para base Hc, luego escalar por factor de geometría
        Hc_base_geom, _ = MODELS.predict_batch(sizes_geom, mat_id)
        fig_hc, ax_hc = plt.subplots(figsize=(12, 4), facecolor='#0f172a')
        ax_hc.set_facecolor('#1e293b')
        geom_colors = ['#38bdf8','#fb923c','#34d399','#f472b6',
                       '#fbbf24','#a78bfa','#6ee7b7','#f87171']
        for (gid, gdata), col in zip(GEOMETRY_MODES.items(), geom_colors):
            hc_vals = Hc_base_geom * GEOMETRY_MODES[gid]['factor_hc']
            ax_hc.plot(sizes_geom, hc_vals, color=col, lw=1.6,
                       label=f'{gdata["emoji"]} {gdata["name"]}')
        ax_hc.axvline(d_nm, color='#f1f5f9', lw=1, ls=':', alpha=0.7)
        ax_hc.axvspan(lo, hi, alpha=0.07, color='#38bdf8')
        ax_hc.set_xlabel('Tamaño (nm)'); ax_hc.set_ylabel('Hc (mT)')
        ax_hc.set_title(
            f'Hc vs Tamaño — {mat["name"]} — todas las geometrías', fontsize=9)
        ax_hc.legend(fontsize=7, ncol=2)
        ax_hc.grid(True, alpha=0.35)
        fig_hc.tight_layout()
        st.pyplot(fig_hc, use_container_width=True)
        plt.close(fig_hc)

    # ── SUB-TAB 4: Aprendizaje Online ─────────────────────────────────────────
    with ml_tab4:
        st.markdown('#### 🔄 Aprendizaje Online — Feedback de Simulaciones')
        st.markdown(
            'Cada simulación agrega un punto de feedback al motor ML. '
            'Al reentrenar, estos puntos tienen **prioridad ×20** sobre los '
            'datos sintéticos, permitiendo que el modelo se especialice en '
            'los rangos que más usas.'
        )

        fb_counts = MODELS.feedback_counts
        total_fb  = MODELS.total_feedback

        fcol1, fcol2, fcol3 = st.columns(3)
        fcol1.metric('Total feedback pts.',   total_fb)
        fcol2.metric(f'Feedback {mat["name"]}', fb_counts.get(mat_id, 0))
        fcol3.metric('Materiales con feedback',
                     sum(1 for v in fb_counts.values() if v > 0))

        if total_fb > 0:
            st.divider()
            # Tabla de feedback por material
            fb_rows = [{'Material': MATERIALS_DB[mid]['name'],
                        'Pts. Feedback': cnt,
                        'Icono': MATERIALS_DB[mid]['emoji']}
                       for mid, cnt in fb_counts.items() if cnt > 0]
            st.dataframe(pd.DataFrame(fb_rows),
                         use_container_width=True, hide_index=True)

        st.divider()
        btn_retrain = st.button(
            '🔁 Reentrenar con feedback acumulado',
            use_container_width=True, type='primary',
            disabled=(total_fb == 0),
            help='Incorpora todos los puntos de feedback al entrenamiento. '
                 'Los datos de simulaciones previas mejoran las predicciones.',
        )
        if btn_retrain:
            with st.spinner('Reentrenando ensemble GBR + RF + MLP…'):
                MODELS.retrain_with_feedback()
            st.success(
                f'✅ Modelos reentrenados con {total_fb} puntos de feedback. '
                f'Predicciones actualizadas automáticamente.'
            )
            st.rerun()

        st.divider()
        st.markdown('**¿Cómo funciona el aprendizaje online?**')
        st.markdown('''
1. Cada vez que presionas **▶ SIMULAR**, el resultado (Hc, Mr) se registra como punto de feedback.
2. El feedback incluye los **7 features físicos** del punto simulado.
3. Al pulsar **Reentrenar**, el engine reconstruye el dataset con:
   - Datos sintéticos base (interpolados de literatura)
   - Puntos de feedback con **peso ×20** (alta prioridad)
4. Los tres modelos (GBR, RF, MLP) se reentrenan y los pesos del ensemble se recalculan.
5. Las predicciones mejoran en los rangos y materiales que más usas.
        ''')
        st.info(
            '💡 **Tip**: Simula el mismo material con varios tamaños para '
            'que el modelo aprenda la forma real de la curva Hc(d) para ese material.'
        )

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — 3D INTERACTIVO
# ─────────────────────────────────────────────────────────────────────────────
with tab_3d:
    st.subheader('🌐 Visualizaciones 3D Interactivas')
    st.caption('Powered by Plotly · Haz zoom, rota y explora con el mouse')

    viz_sel = st.radio(
        'Visualización',
        ['🧊 Geometría 3D Vóxel',
         '🗺️ Mapa 2D de Magnetización',
         '⚡ Componentes de Energía',
         '🏔 Superficie de Energía  E(H, d)',
         '🧲 Vectores de Magnetización',
         '🌡 Mapa de Calor  Hc(material, nm)',
         '🔵 Polar de Anisotropía  E_K(θ)',
         '📚 Stack 3D de Histéresis'],
        horizontal=False,
    )

    # adaptar predict para viz3d — usa predict_fast (sin varianza RF → rápido)
    def _predict_for_viz(d, mid, geom_key, models):
        return models.predict_fast(d, mid)

    # ── Helper: botones de exportación para figuras Plotly ────────────────────
    def _export_plotly(fig, fname_base):
        """Descarga HTML interactivo + intento PNG (requiere kaleido)."""
        c_html, c_png = st.columns(2)
        html_bytes = fig.to_html(full_html=True, include_plotlyjs='cdn').encode()
        c_html.download_button(
            '⬇ Descargar HTML interactivo',
            data=html_bytes,
            file_name=f'{fname_base}.html',
            mime='text/html',
            use_container_width=True,
        )
        try:
            png_bytes = fig.to_image(format='png', width=1400, height=800, scale=2)
            c_png.download_button(
                '⬇ Descargar PNG (alta res)',
                data=png_bytes,
                file_name=f'{fname_base}.png',
                mime='image/png',
                use_container_width=True,
            )
        except Exception:
            c_png.caption('📌 PNG: instala `kaleido` para activar  ·  `pip install kaleido`')

    # ── Geometría 3D Vóxel ────────────────────────────────────────────────────
    if '🧊' in viz_sel:
        st.markdown(f'**Geometría voxelizada — {gm["emoji"]} {gm["name"]}  @  {d_nm} nm**')
        st.markdown(
            'Cada punto es un vóxel de la nanopartícula. '
            'Color: 🔴 = mz/Ms ≈ +1 (eje fácil ↑)  ·  🔵 = mz/Ms ≈ −1 (↓). '
            'Inspirado en visualizaciones de MuMax3.'
        )
        n_vox = st.slider('Resolución de vóxeles', 14, 30, 20, step=2, key='vox_n')
        _vox_key = f'fig_vox_{geom_id}_{d_nm}_{n_vox}'
        if _vox_key not in st.session_state:
            with st.spinner('Voxelizando geometría…'):
                st.session_state[_vox_key] = _viz3d.voxel_geometry_3d(
                    geom_id, d_nm, GEOMETRY_MODES, n=n_vox)
        fig_vox = st.session_state[_vox_key]
        st.plotly_chart(fig_vox, use_container_width=True)
        _export_plotly(fig_vox, f'voxel_{geom_id}_{d_nm:.0f}nm')

    # ── Mapa 2D de Magnetización ──────────────────────────────────────────────
    elif '🗺️' in viz_sel:
        st.markdown(f'**Mapa 2D de mz — {mat["name"]}  @  {d_nm} nm**')
        st.markdown(
            'Cuatro estados magnéticos en el corte transversal XY. '
            'Color: 🔴 = +Ms, 🔵 = −Ms. Flechas = dirección de M en el plano. '
            'Inspirado en Fig. 4 de Galvis, Mesa et al. (Results in Physics, 2025).'
        )
        n_grid_map = st.slider('Resolución de grilla', 18, 48, 28, step=2, key='map_grid')
        _map_key = f'fig_map_{mat_id}_{d_nm}_{n_grid_map}'
        if _map_key not in st.session_state:
            with st.spinner('Calculando mapas 2D…'):
                st.session_state[_map_key] = _viz3d.magnetization_map_2d(
                    mat_id, d_nm, MODELS, MATERIALS_DB, _predict_for_viz,
                    n_grid=n_grid_map)
        fig_map = st.session_state[_map_key]
        st.plotly_chart(fig_map, use_container_width=True)
        _export_plotly(fig_map, f'mapa2d_{mat_id}_{d_nm:.0f}nm')

    # ── Componentes de Energía (4 paneles) ────────────────────────────────────
    elif '⚡' in viz_sel:
        st.markdown(f'**Componentes de Energía vs H — {mat["name"]}**')
        st.markdown(
            'Cuatro contribuciones energéticas para distintos tamaños de partícula. '
            'Magnitudes calculadas con parámetros físicos del material. '
            'Inspirado en Fig. 3 de Galvis, Mesa et al. (Results in Physics, 2025).'
        )

        # ── Toggle: datos reales OOMMF ────────────────────────────────────────
        _show_real = False
        if _REAL_DATA_OK:
            _show_real = st.toggle(
                '🔬 Mostrar datos reales OOMMF (2 esferas Fe, r=21 nm, separación=6 nm)',
                value=False, key='energy_show_real',
            )

        if _show_real and _REAL_DATA_OK:
            # ── Visualización datos reales de 12nm.ipynb ─────────────────────
            st.info(
                '**Datos reales OOMMF** · Sistema: 2 esferas de Fe (r=21 nm, sep=6 nm) '
                'en caja 114×42×42 nm · Barrido ±400 mT · Runner: ExeOOMMFRunner · '
                'Fuente: `12nm.ipynb` (Galvis, Mesa et al.)'
            )
            _rd_key = 'fig_energy_real_oommf'
            if _rd_key not in st.session_state:
                with st.spinner('Cargando datos reales OOMMF…'):
                    _hist   = _ref_data.load_hysteresis()
                    _energ  = _ref_data.load_energies()

                    _fig_real = go.Figure()

                    # ── Subplots: 5 paneles (histeresis + 4 energías) ─────────
                    from plotly.subplots import make_subplots
                    _fig_real = make_subplots(
                        rows=3, cols=2,
                        subplot_titles=[
                            'Ciclo de Histéresis  (M/Ms)',
                            'Energía de Anisotropía  (J)',
                            'Energía Zeeman  (J)',
                            'Energía Dipolar  (J)',
                            'Energía de Intercambio  (J)',
                            '',
                        ],
                        vertical_spacing=0.12,
                        horizontal_spacing=0.10,
                    )

                    _clr_desc = '#38bdf8'
                    _clr_asc  = '#fb923c'

                    # Panel 1: Histéresis
                    _fig_real.add_trace(go.Scatter(
                        x=_hist['fd_desc'], y=_hist['mg_desc'],
                        mode='lines+markers', name='M/Ms ↓', marker_size=4,
                        line=dict(color=_clr_desc, width=2),
                    ), row=1, col=1)
                    _fig_real.add_trace(go.Scatter(
                        x=_hist['fd_asc'], y=_hist['mg_asc'],
                        mode='lines+markers', name='M/Ms ↑', marker_size=4,
                        line=dict(color=_clr_asc, width=2),
                    ), row=1, col=1)
                    # Hc markers
                    _Hc_r, _Mr_r = _ref_data.extract_hc_mr()
                    _fig_real.add_vline(x= _Hc_r, line_dash='dot',
                        line_color='#f43f5e', line_width=1.5, row=1, col=1)
                    _fig_real.add_vline(x=-_Hc_r, line_dash='dot',
                        line_color='#f43f5e', line_width=1.5, row=1, col=1)

                    # Panel 2: Anisotropía
                    _ean = _energ['anisotropia']
                    _fig_real.add_trace(go.Scatter(
                        x=_ean['fd_desc'], y=_ean['mg_desc'],
                        mode='lines+markers', name='E_anis ↓', marker_size=4,
                        line=dict(color=_clr_desc, width=2), showlegend=False,
                    ), row=1, col=2)
                    _fig_real.add_trace(go.Scatter(
                        x=_ean['fd_asc'], y=_ean['mg_asc'],
                        mode='lines+markers', name='E_anis ↑', marker_size=4,
                        line=dict(color=_clr_asc, width=2), showlegend=False,
                    ), row=1, col=2)

                    # Panel 3: Zeeman
                    _ez = _energ['zeeman']
                    _fig_real.add_trace(go.Scatter(
                        x=_ez['fd_desc'], y=_ez['mg_desc'],
                        mode='lines+markers', name='E_Z ↓', marker_size=4,
                        line=dict(color=_clr_desc, width=2), showlegend=False,
                    ), row=2, col=1)
                    _fig_real.add_trace(go.Scatter(
                        x=_ez['fd_asc'], y=_ez['mg_asc'],
                        mode='lines+markers', name='E_Z ↑', marker_size=4,
                        line=dict(color=_clr_asc, width=2), showlegend=False,
                    ), row=2, col=1)

                    # Panel 4: Dipolar
                    _ed = _energ['dipolar']
                    _fig_real.add_trace(go.Scatter(
                        x=_ed['fd_desc'], y=_ed['mg_desc'],
                        mode='lines+markers', name='E_dip ↓', marker_size=4,
                        line=dict(color=_clr_desc, width=2), showlegend=False,
                    ), row=2, col=2)
                    _fig_real.add_trace(go.Scatter(
                        x=_ed['fd_asc'], y=_ed['mg_asc'],
                        mode='lines+markers', name='E_dip ↑', marker_size=4,
                        line=dict(color=_clr_asc, width=2), showlegend=False,
                    ), row=2, col=2)

                    # Panel 5: Intercambio
                    _ex = _energ['intercambio']
                    _fig_real.add_trace(go.Scatter(
                        x=_ex['fd_desc'], y=_ex['mg_desc'],
                        mode='lines+markers', name='E_ex ↓', marker_size=4,
                        line=dict(color=_clr_desc, width=2), showlegend=False,
                    ), row=3, col=1)
                    _fig_real.add_trace(go.Scatter(
                        x=_ex['fd_asc'], y=_ex['mg_asc'],
                        mode='lines+markers', name='E_ex ↑', marker_size=4,
                        line=dict(color=_clr_asc, width=2), showlegend=False,
                    ), row=3, col=1)

                    # Layout global
                    _fig_real.update_layout(
                        height=850,
                        paper_bgcolor='#0f172a',
                        plot_bgcolor='#1e293b',
                        font=dict(color='#f1f5f9', size=11),
                        title=dict(
                            text='Datos reales OOMMF — 2 esferas Fe · r=21 nm · sep=6 nm · ±400 mT',
                            font=dict(color='#f1f5f9', size=14),
                        ),
                        legend=dict(
                            bgcolor='#1e293b', bordercolor='#334155',
                            title=dict(text='Rama'),
                        ),
                        margin=dict(t=80, b=40, l=60, r=30),
                    )
                    # Ejes en todos los subplots
                    for _ax in _fig_real.layout:
                        if _ax.startswith('xaxis') or _ax.startswith('yaxis'):
                            _fig_real.layout[_ax].update(
                                gridcolor='#334155', zerolinecolor='#475569',
                                color='#94a3b8',
                            )
                    # Etiquetas x
                    for _r, _c in [(1,1),(1,2),(2,1),(2,2),(3,1)]:
                        _fig_real.update_xaxes(title_text='H (mT)', row=_r, col=_c)
                    # Etiquetas y
                    _ylabels = {(1,1):'M/Ms',(1,2):'E (J)',(2,1):'E (J)',
                                (2,2):'E (J)',(3,1):'E (J)'}
                    for (_r,_c), _lbl in _ylabels.items():
                        _fig_real.update_yaxes(title_text=_lbl, row=_r, col=_c)

                    st.session_state[_rd_key] = _fig_real

            fig_real_oommf = st.session_state[_rd_key]
            st.plotly_chart(fig_real_oommf, use_container_width=True)
            _export_plotly(fig_real_oommf, 'energia_real_oommf_2esferas_fe')

            # Métricas extraídas de los datos reales
            st.markdown('##### Valores extraídos de los datos reales')
            _rp = _ref_data.REFERENCE_PARAMS
            _rm1, _rm2, _rm3, _rm4, _rm5 = st.columns(5)
            _rm1.metric('Hc (mT)',         f"{_rp['Hc_mT']:.1f}")
            _rm2.metric('Mr / Ms',          f"{_rp['Mr_Ms']:.4f}")
            _rm3.metric('H_max (mT)',       f"{_rp['H_max_mT']:.0f}")
            _rm4.metric('Radio (nm)',        f"{_rp['radius_nm']:.0f}")
            _rm5.metric('Separación (nm)',   f"{_rp['separation_nm']:.0f}")
            st.caption(
                f"Material: **{_rp['material']}**  ·  "
                f"Ms = {_rp['Ms_Am']/1e6:.2f} MA/m  ·  "
                f"K₁ = {_rp['K1_Jm3']/1e3:.0f} kJ/m³  ·  "
                f"A = {_rp['A_Jm']*1e12:.1f} pJ/m  ·  "
                f"Celda = {_rp['cell_nm']:.0f} nm  ·  "
                f"Fuente: `{_rp['source_nb']}`"
            )

        else:
            # ── Vista ML: energía calculada para múltiples tamaños ────────────
            n_curvas = st.slider('Número de tamaños', 3, 10, 5, key='energy_n')
            _eng_key = f'fig_energy_{mat_id}_{n_curvas}'
            if _eng_key not in st.session_state:
                with st.spinner('Calculando componentes de energía…'):
                    st.session_state[_eng_key] = _viz3d.energy_components_4panel(
                        mat_id, MODELS, MATERIALS_DB, _predict_for_viz, n_sizes=n_curvas)
            fig_energy4 = st.session_state[_eng_key]
            st.plotly_chart(fig_energy4, use_container_width=True)
            _export_plotly(fig_energy4, f'energia_{mat_id}_{d_nm:.0f}nm')

    elif '🏔' in viz_sel:
        st.markdown(f'**Superficie E(H, d) — {mat["name"]}**')
        st.markdown('El eje X es el campo H, eje Y el tamaño y eje Z la energía total normalizada.')
        _surf_key = f'fig_surf_{mat_id}'
        if _surf_key not in st.session_state:
            with st.spinner('Calculando superficie 3D…'):
                st.session_state[_surf_key] = _viz3d.surface_energy_3d(
                    mat_id, MODELS, MATERIALS_DB, _predict_for_viz)
        fig_surf = st.session_state[_surf_key]
        st.plotly_chart(fig_surf, use_container_width=True)
        _export_plotly(fig_surf, f'superficie_e_{mat_id}_{d_nm:.0f}nm')

    elif '🧲' in viz_sel:
        st.markdown(f'**Vectores M sobre esfera — {mat["name"]}  @  {d_nm} nm**')
        st.markdown('Conos coloreados por Mz: 🔴 = +Ms, 🔵 = –Ms.')
        _vec_key = f'fig_vecs_{mat_id}_{d_nm}'
        if _vec_key not in st.session_state:
            with st.spinner('Generando vectores…'):
                st.session_state[_vec_key] = _viz3d.magnetization_vectors(
                    mat_id, d_nm, MODELS, MATERIALS_DB, _predict_for_viz)
        fig_vecs = st.session_state[_vec_key]
        st.plotly_chart(fig_vecs, use_container_width=True)
        _export_plotly(fig_vecs, f'vectores_m_{mat_id}_{d_nm:.0f}nm')

    elif '🌡' in viz_sel:
        st.markdown('**Mapa de Calor Hc — todos los materiales × todos los tamaños**')
        geom_hm = st.radio('Geometría base', ['sphere', 'cuboid'],
                            format_func=lambda x: 'Esfera' if x == 'sphere' else 'Cuboide',
                            horizontal=True, key='hm_geom')
        _hm_key = f'fig_hm_{geom_hm}'
        if _hm_key not in st.session_state:
            with st.spinner('Calculando mapa de calor…'):
                st.session_state[_hm_key] = _viz3d.hc_heatmap(
                    MODELS, MATERIALS_DB, _predict_for_viz, geom=geom_hm)
        fig_hm = st.session_state[_hm_key]
        st.plotly_chart(fig_hm, use_container_width=True)
        _export_plotly(fig_hm, f'heatmap_hc_{geom_hm}')

    elif '🔵' in viz_sel:
        st.markdown('**Energía de Anisotropía E_K(θ) — comparación de materiales**')
        st.markdown('E_K = K₁·sin²(θ), normalizado a K₁_max. Mayor área = mayor barrera de anisotropía.')
        mats_polar = st.multiselect(
            'Materiales',
            list(MATERIALS_DB.keys()), default=list(MATERIALS_DB.keys()),
            format_func=lambda x: f"{MATERIALS_DB[x]['emoji']} {MATERIALS_DB[x]['name']}",
            key='polar_mats',
        )
        if mats_polar:
            fig_polar = _viz3d.polar_anisotropy(MATERIALS_DB, mat_ids=mats_polar)
            st.plotly_chart(fig_polar, use_container_width=True)
            _export_plotly(fig_polar, f'polar_anisotropia')
        else:
            st.info('Selecciona al menos un material.')

    elif '📚' in viz_sel:
        st.markdown(f'**Stack 3D de Histéresis — {mat["name"]}**')
        st.markdown('Cada plano es un tamaño diferente. Color: azul → tamaño mínimo · naranja → máximo.')
        n_lazos = st.slider('Número de lazos', 4, 14, 6, key='stack_n')
        _stack_key = f'fig_stack_{mat_id}_{n_lazos}'
        if _stack_key not in st.session_state:
            with st.spinner('Generando stack 3D…'):
                st.session_state[_stack_key] = _viz3d.hysteresis_3d_stack(
                    mat_id, MODELS, MATERIALS_DB, _predict_for_viz,
                    llg_fn=llg_hysteresis, n_sizes=n_lazos)
        fig_stack = st.session_state[_stack_key]
        st.plotly_chart(fig_stack, use_container_width=True)
        _export_plotly(fig_stack, f'stack3d_{mat_id}')

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 5 — DASHBOARD  (SQLite + sesión)
# ─────────────────────────────────────────────────────────────────────────────
with tab_dashboard:
    st.subheader('📈 Dashboard — Base de Datos SQLite + Sesión')

    db_rows  = _db.get_all_simulations()
    db_stats = _db.get_stats()

    source_tab, hist_tab = st.tabs(['🗄 SQLite (persistente)', '📋 Sesión actual'])

    with source_tab:
        if not db_rows:
            st.info('No hay registros todavía.')
        else:
            df_db = pd.DataFrame(db_rows)
            kp1,kp2,kp3,kp4,kp5 = st.columns(5)
            kp1.metric('Total simulaciones',  db_stats.get('total', 0))
            kp2.metric('Materiales únicos',   db_stats.get('unique_materials', 0))
            kp3.metric('Tamaño mínimo (nm)',  f"{db_stats.get('min_size', 0):.1f}")
            kp4.metric('Tamaño máximo (nm)',  f"{db_stats.get('max_size', 0):.1f}")
            kp5.metric('Extrapolaciones',     db_stats.get('total_extrapolations', 0))
            st.divider()
            if len(df_db) > 1:
                st.markdown('**Evolución histórica de Hc**')
                plt.rcParams.update(_DARK)
                fig_db2, ax_db2 = plt.subplots(figsize=(12, 3), facecolor='#0f172a')
                ax_db2.set_facecolor('#1e293b')
                for mat_name in df_db['material'].unique():
                    sub   = df_db[df_db['material'] == mat_name].dropna(subset=['hc_sphere'])
                    color = next((v['color'] for v in MATERIALS_DB.values()
                                  if v['name'] == mat_name), '#38bdf8')
                    if not sub.empty:
                        ax_db2.plot(range(len(sub)), sub['hc_sphere'].values,
                                    'o-', color=color, lw=1.8, ms=4, label=mat_name)
                ax_db2.set_xlabel('# Registro'); ax_db2.set_ylabel('Hc (mT)')
                ax_db2.set_title('Historial SQLite — Hc', fontsize=9)
                ax_db2.legend(fontsize=7); ax_db2.grid(True, alpha=0.35)
                plt.tight_layout()
                st.pyplot(fig_db2, use_container_width=True)
                plt.close(fig_db2)
            st.divider()
            cols_show = ['id','timestamp','material','size_nm','geometry',
                         'hc_sphere','mr_sphere','extrapolation']
            st.dataframe(df_db[cols_show], use_container_width=True, hide_index=True)
            if st.button('🗑 Limpiar base de datos SQLite', key='clr_db'):
                _db.clear_simulations()
                st.rerun()

    with hist_tab:
        hist = st.session_state.history
        if not hist:
            st.info('No hay simulaciones en la sesión actual.')
        else:
            df_hist = pd.DataFrame(hist)
            kp1,kp2,kp3,kp4 = st.columns(4)
            kp1.metric('Simulaciones',      len(df_hist))
            kp2.metric('Materiales únicos', df_hist['Material'].nunique())
            kp3.metric('Min tamaño (nm)',   df_hist['Tamaño (nm)'].min())
            kp4.metric('Max tamaño (nm)',   df_hist['Tamaño (nm)'].max())
            st.divider()
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
            st.divider()
            st.markdown('**Distribución por material**')
            counts = df_hist['Material'].value_counts()
            plt.rcParams.update(_DARK)
            fig_pie, ax_pie = plt.subplots(figsize=(6, 3.5), facecolor='#0f172a')
            ax_pie.set_facecolor('#0f172a')
            colors_pie = [next((v['color'] for v in MATERIALS_DB.values()
                                if v['name'] == nm), '#38bdf8')
                          for nm in counts.index]
            ax_pie.pie(counts.values, labels=counts.index, colors=colors_pie,
                       autopct='%1.0f%%',
                       textprops={'color':'#f1f5f9','fontsize':8},
                       wedgeprops={'edgecolor':'#0f172a','linewidth':2})
            ax_pie.set_title('Por material', fontsize=9, color='#f1f5f9')
            fig_pie.tight_layout()
            st.pyplot(fig_pie, use_container_width=True)
            plt.close(fig_pie)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 6 — EXPORTAR
# ─────────────────────────────────────────────────────────────────────────────
with tab_export:
    st.subheader('💾 Exportar Resultados')

    # ── PDF ────────────────────────────────────────────────────────────────
    st.markdown('#### 📄 Reporte PDF Científico')
    st.markdown('Portada · parámetros físicos · predicciones ML · ecuaciones · referencias.')
    if st.button('🖨 Generar Reporte PDF', use_container_width=True, type='primary'):
        with st.spinner('Generando PDF…'):
            pdf_bytes = _report.generate_report(
                mat_id=mat_id, mat_name=mat['name'], d_nm=d_nm,
                preds={'sphere': (Hc_val, Mr_val)},
                mat_params=mat['params'], mat_range=mat['range'],
                field_max=mat['field_max'], fig_main=None, fig_energy=None,
                noise_level=noise_level,
                extrapolation=is_extrapolation(d_nm, mat_id),
                history_rows=_db.get_all_simulations()[:30],
            )
        fname_pdf = f'reporte_{mat_id}_{geom_id}_{d_nm:.0f}nm.pdf'
        st.download_button(
            label=f'⬇ Descargar {fname_pdf}', data=pdf_bytes,
            file_name=fname_pdf, mime='application/pdf',
            use_container_width=True)
        _db.log_report(mat_id, d_nm, fname_pdf)
        st.success(f'✅ {fname_pdf}')

    st.divider()
    ex1, ex2 = st.columns(2)

    with ex1:
        st.markdown('#### 🖼 Figura PNG')
        buf_png = io.BytesIO()
        fig_main.savefig(buf_png, format='png', dpi=export_dpi,
                         bbox_inches='tight', facecolor='#0f172a')
        buf_png.seek(0)
        fname_png = f'fig_{mat_id}_{geom_id}_{d_nm}nm.png'
        st.download_button(f'⬇ {fname_png}', buf_png, fname_png, 'image/png',
                           use_container_width=True)
        st.caption(f'DPI: {export_dpi}')

    with ex2:
        st.markdown('#### 📄 Histéresis CSV')
        H_csv, M_up_csv, M_dn_csv = llg_hysteresis(
            Hc_val, Mr_val, H_max=mat['field_max'],
            noise_level=noise_level, seed=42)
        df_csv = pd.DataFrame({'H (mT)': H_csv,
                                'M_up/Ms': M_up_csv, 'M_dn/Ms': M_dn_csv})
        fname_csv = f'histeresis_{mat_id}_{geom_id}_{d_nm}nm.csv'
        st.download_button(f'⬇ {fname_csv}',
                           df_csv.to_csv(index=False).encode(),
                           fname_csv, 'text/csv', use_container_width=True)
        st.caption(f'500 pts · H ∈ [−{mat["field_max"]}, +{mat["field_max"]}] mT')

    st.divider()
    st.markdown('#### 📊 Historial SQLite (CSV)')
    if db_rows:
        st.download_button(
            '⬇ historial_completo.csv',
            pd.DataFrame(db_rows).to_csv(index=False).encode(),
            'historial_completo.csv', 'text/csv', use_container_width=True)
    else:
        st.info('Sin registros todavía.')

    st.divider()
    st.markdown('#### 🗂 Parámetros del material (JSON)')
    mat_json = {mat_id: {k: v for k, v in mat.items()
                          if k not in ('sphere', 'cuboid')}}
    def _np_serial(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        return o
    st.download_button(
        f'⬇ params_{mat_id}.json',
        json.dumps(mat_json, default=_np_serial, indent=2, ensure_ascii=False).encode(),
        f'params_{mat_id}.json', 'application/json', use_container_width=True)

plt.close(fig_main)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 7 — VALIDACIÓN UBERMAG  (discretisedfield + micromagneticmodel + oommfc)
# ─────────────────────────────────────────────────────────────────────────────
with tab_uval:
    st.subheader('🧲 Validación de Geometrías — Ubermag / OOMMF')
    st.markdown(
        'Validación física de las 8 geometrías utilizando el framework científico '
        '**Ubermag** (discretisedfield + micromagneticmodel + oommfc). '
        'Los factores de forma se calculan a partir de los factores de '
        'desmagnetización analíticos (Osborn 1945, Chen 1991, Aharoni 1998) '
        'y se comparan con los valores usados en el simulador.'
    )

    # ── Subtabs de validación ─────────────────────────────────────────────────
    uv_tab1, uv_tab2, uv_tab3, uv_tab4 = st.tabs([
        '🔬 Geometría (discretisedfield)',
        '📐 Factores de Forma (Stoner-Wohlfarth)',
        '🌐 Radar de Anisotropía',
        '🖥️ Simulación OOMMF',
    ])

    # Crear validador (cacheado en session_state)
    if 'ubermag_validator' not in st.session_state:
        d_val = st.session_state.get('d_nm', 20.0)
        st.session_state['ubermag_validator'] = _uval.UbermagValidator(
            MATERIALS_DB, GEOMETRY_MODES, d_test_nm=d_val)

    validator = st.session_state['ubermag_validator']

    # ── Sub-tab 1: Geometría ──────────────────────────────────────────────────
    with uv_tab1:
        st.markdown('#### Validación geométrica con `discretisedfield`')
        st.caption(
            'Cada geometría se construye como un mesh 3D discretizado (celdas de 2.5 nm). '
            'Se mide el volumen discret izado y se compara con el volumen analítico de la esfera. '
            'Los factores Nd se calculan con fórmulas analíticas exactas.'
        )

        d_val_input = st.slider(
            'Diámetro de prueba (nm)', 10, 60, 20, step=5, key='uval_d')
        cell_val    = st.select_slider(
            'Tamaño de celda (nm)', [1.5, 2.0, 2.5, 3.0], value=2.5, key='uval_cell')

        _geom_key = f'uval_geom_{d_val_input}_{cell_val}'
        if _geom_key not in st.session_state:
            with st.spinner('Construyendo meshes con discretisedfield…'):
                v2 = _uval.UbermagValidator(
                    MATERIALS_DB, GEOMETRY_MODES, d_test_nm=d_val_input)
                st.session_state[_geom_key] = {
                    'metrics': v2.validate_geometry(cell_nm=cell_val),
                    'fig':     v2.plot_geometry_metrics(),
                    'table':   None,
                    'validator': v2,
                }
        gd = st.session_state[_geom_key]

        st.plotly_chart(gd['fig'], use_container_width=True)

        # Tabla de métricas geométricas
        import pandas as pd
        rows_g = []
        for gid, m in gd['metrics'].items():
            rows_g.append({
                'Geometría':      _uval.GEOM_LABELS[gid].split('(')[0].strip(),
                'Ref. física':    _uval.GEOM_LABELS[gid].split('(')[-1].replace(')','')
                                  if '(' in _uval.GEOM_LABELS[gid] else '—',
                'Celdas válidas': m['n_cells'],
                'V disc. (nm³)':  f"{m['V_nm3']:.1f}",
                'V / V_esfera':   f"{m['V_rel']:.3f}",
                'N_z (analítico)':f"{m.get('Nd_z', 0.333):.3f}",
                'N_x (analítico)':f"{m.get('Nd_x', 0.333):.3f}",
                'ΔN':             f"{m.get('Nd_aniso', 0):.4f}",
            })
        st.dataframe(pd.DataFrame(rows_g), use_container_width=True, hide_index=True)

        st.info(
            '📌 **Nd analítico** calculado con: Osborn (1945) para elipsoides, '
            'Chen (1991) para cilindros, Aharoni (1998) para cuboides, '
            'Field et al. (2011) para toroide, Nogués et al. (1999) para núcleo-cáscara.'
        )

    # ── Sub-tab 2: Factores Stoner-Wohlfarth ─────────────────────────────────
    with uv_tab2:
        st.markdown('#### Factores de forma — App vs. Stoner-Wohlfarth (Ubermag)')
        st.caption(
            'Los factores del simulador se comparan con los valores calculados '
            'por el modelo de Stoner-Wohlfarth usando los Nd analíticos. '
            'La anisotropía de forma ΔN·Ms modifica el campo de conmutación '
            'respecto a la esfera de referencia.'
        )

        mat_sw = st.selectbox(
            'Material de referencia para factores SW',
            list(MATERIALS_DB.keys()),
            format_func=lambda x: f"{MATERIALS_DB[x]['emoji']} {MATERIALS_DB[x]['name']}",
            key='uval_mat_sw',
        )

        _sw_key = f'uval_sw_{mat_sw}'
        if _sw_key not in st.session_state:
            with st.spinner('Calculando factores Stoner-Wohlfarth…'):
                v_sw = _uval.UbermagValidator(MATERIALS_DB, GEOMETRY_MODES)
                sw_res = v_sw.compute_sw_factors(mat_sw)
                st.session_state[_sw_key] = {
                    'fig':     v_sw.plot_shape_factors(),
                    'factors': sw_res,
                }
        sw_data = st.session_state[_sw_key]
        st.plotly_chart(sw_data['fig'], use_container_width=True)

        # Tabla de factores
        rows_sw = []
        for gid, f in sw_data['factors'].items():
            rows_sw.append({
                'Geometría':       _uval.GEOM_LABELS[gid].split('(')[0].strip(),
                'ΔN':              f"{f['delta_Nd']:+.3f}",
                'factor_hc (app)': f['factor_hc_app'],
                'factor_hc (SW)':  f['factor_hc_sw'],
                'Δ hc %':          f"{f['hc_error_pct']}%",
                'factor_mr (app)': f['factor_mr_app'],
                'factor_mr (SW)':  f['factor_mr_sw'],
            })
        df_sw = pd.DataFrame(rows_sw)
        st.dataframe(df_sw, use_container_width=True, hide_index=True)

        st.markdown(
            '**Nota:** El modelo SW asume rotación coherente (válido para '
            'd < d_crit monodominio). Para partículas grandes, la inversión '
            'por movimiento de paredes reduce el Hc real < H_sw.'
        )

        # Mostrar parámetros del material seleccionado
        p_sw = MATERIALS_DB[mat_sw]['params']
        Ms   = p_sw['Ms_MA_m'] * 1e6
        K1   = abs(p_sw['K1_kJ_m3']) * 1e3
        H_mca_mT = round(2 * K1 / Ms * 1e3 / (4*np.pi*1e-7) / 1e3, 1) if Ms > 0 else 0
        H_shape_mT = round((1/3) * Ms * 4*np.pi*1e-7 * 1e3, 1)
        st.info(
            f'**{MATERIALS_DB[mat_sw]["name"]}** — '
            f'Ms = {p_sw["Ms_MA_m"]} MA/m · K₁ = {p_sw["K1_kJ_m3"]} kJ/m³ · '
            f'H_K,mca ≈ {H_mca_mT} mT · '
            f'H_shape(esfera) ≈ {H_shape_mT} mT'
        )

    # ── Sub-tab 3: Radar de anisotropía ───────────────────────────────────────
    with uv_tab3:
        st.markdown('#### Radar de Anisotropía de Forma  ΔN = N_⊥ – N_∥')
        st.caption(
            'ΔN > 0: eje fácil de forma (aumenta Hc). '
            'ΔN < 0: plano fácil de forma (reduce Hc). '
            'ΔN = 0: isótropo (esfera).'
        )
        if 'uval_radar_fig' not in st.session_state:
            v_rad = _uval.UbermagValidator(MATERIALS_DB, GEOMETRY_MODES)
            st.session_state['uval_radar_fig'] = v_rad.plot_Nd_radar()
        st.plotly_chart(
            st.session_state['uval_radar_fig'], use_container_width=True)

        # Tabla de Nd completa
        rows_nd = []
        for gid, Nd in _uval.GEOM_Nd.items():
            rows_nd.append({
                'Geometría':   _uval.GEOM_LABELS[gid].split('(')[0].strip(),
                'Referencia':  _uval.GEOM_LABELS[gid].split('(')[-1].replace(')','')
                               if '(' in _uval.GEOM_LABELS[gid] else '—',
                'N_x':   f"{Nd[0]:.4f}",
                'N_y':   f"{Nd[1]:.4f}",
                'N_z':   f"{Nd[2]:.4f}",
                'ΔN':    f"{Nd[0]-Nd[2]:+.4f}",
                'Traza': f"{sum(Nd):.4f}",
            })
        st.dataframe(pd.DataFrame(rows_nd), use_container_width=True, hide_index=True)
        st.caption('Traza = N_x + N_y + N_z = 1 (conservación del flujo magnético).')

    # ── Sub-tab 4: Simulación OOMMF ───────────────────────────────────────────
    with uv_tab4:
        st.markdown('#### 🖥️ Simulación OOMMF completa vía `oommfc`')

        # ── Comprobación de disponibilidad ───────────────────────────────────
        _oommf_avail = False
        try:
            import oommfc as _oc
            _oc.oommf.DockerOOMMFRunner(image='ubermag/oommf')
            _oommf_avail = True
        except Exception:
            pass

        if _oommf_avail:
            st.success(
                '✅ **oommfc** disponible · Docker con `ubermag/oommf` detectado — '
                'se ejecutará simulación OOMMF completa.'
            )
        else:
            st.info(
                'ℹ️ **OOMMF/Docker** no detectado en este entorno — '
                'se usará simulación analítica **Stoner-Wohlfarth** con corrección '
                'de forma Ubermag como respaldo.  '
                'Para activar OOMMF: `docker pull ubermag/oommf`'
            )

        st.divider()

        # ── Controles de la simulación ───────────────────────────────────────
        _col_s1, _col_s2, _col_s3 = st.columns([1, 1, 1])
        with _col_s1:
            _sim_geom = st.selectbox(
                'Geometría',
                list(GEOMETRY_MODES.keys()),
                format_func=lambda g: f"{GEOMETRY_MODES[g]['emoji']}  {GEOMETRY_MODES[g]['name']}",
                key='oommf_geom',
            )
            _sim_mat = st.selectbox(
                'Material',
                list(MATERIALS_DB.keys()),
                format_func=lambda m: (
                    f"{MATERIALS_DB[m]['emoji']}  {MATERIALS_DB[m]['name']}"
                ),
                key='oommf_mat',
            )
        with _col_s2:
            _sim_d    = st.slider('Diámetro (nm)',   5,  60,  20,  1,  key='oommf_d')
            _sim_Hmax = st.slider('H máximo (mT)', 100, 800, 300, 50, key='oommf_Hmax')
        with _col_s3:
            _sim_steps = st.slider('Pasos de campo',  20, 80, 40, 10, key='oommf_steps')
            _sim_cell  = st.slider('Celda (nm)',        2,  5,  3,  1, key='oommf_cell')

        _run_sim = st.button('▶ Ejecutar Simulación', type='primary',
                             key='btn_run_oommf')

        _sim_key = (f'oommf_res_{_sim_geom}_{_sim_mat}_{_sim_d}'
                    f'_{_sim_Hmax}_{_sim_steps}_{_sim_cell}')

        # Recalcular si el usuario presionó el botón
        if _run_sim:
            st.session_state.pop(_sim_key, None)

        if _run_sim or _sim_key in st.session_state:

            if _sim_key not in st.session_state:
                with st.spinner('⏳ Ejecutando simulación micromagnética…'):
                    _sim_res = None

                    # Intento real con OOMMF Docker
                    if _oommf_avail:
                        _sim_res = _uval.run_oommf_hysteresis(
                            geom_id=_sim_geom,
                            d_nm=float(_sim_d),
                            mat_params=MATERIALS_DB[_sim_mat]['params'],
                            H_max_mT=float(_sim_Hmax),
                            n_steps=_sim_steps,
                            cell_nm=float(_sim_cell),
                        )

                    # Respaldo analítico Stoner-Wohlfarth
                    if _sim_res is None:
                        _sim_res = _oommf_sw_fallback(
                            _sim_geom, float(_sim_d), _sim_mat,
                            float(_sim_Hmax), _sim_steps,
                        )

                st.session_state[_sim_key] = _sim_res

            _res = st.session_state[_sim_key]

            # ── Métricas de resultado ─────────────────────────────────────────
            st.markdown('##### Resultados de la simulación')
            _mc1, _mc2, _mc3, _mc4 = st.columns(4)
            _mc1.metric('Hc simulado (mT)',  f"{_res['Hc_mT']:.1f}")
            _mc2.metric('Mr / Ms',           f"{_res['Mr']:.3f}")
            _mc3.metric('Celdas activas',    _res.get('n_cells', '—'))
            _mc4.metric('Motor',             _res.get('runner', '—'))

            # ── Lazo de histéresis interactivo ────────────────────────────────
            _H  = _res['H']
            _M  = _res['M']
            _nh = len(_H) // 2

            # Remanencia en H=0 (rama descendente)
            _Mr_H0 = float(np.interp(0.0, _H[:_nh][::-1], _M[:_nh][::-1]))

            _fig_h = go.Figure()

            # Rama descendente
            _fig_h.add_trace(go.Scatter(
                x=_H[:_nh], y=_M[:_nh],
                mode='lines', name='H ↓ bajada',
                line=dict(color='#38bdf8', width=2.5),
                hovertemplate='H=%{x:.1f} mT<br>M/Ms=%{y:.4f}<extra>bajada</extra>',
            ))

            # Rama ascendente
            _fig_h.add_trace(go.Scatter(
                x=_H[_nh:], y=_M[_nh:],
                mode='lines', name='H ↑ subida',
                line=dict(color='#fb923c', width=2.5),
                hovertemplate='H=%{x:.1f} mT<br>M/Ms=%{y:.4f}<extra>subida</extra>',
            ))

            # Punto de remanencia Mr
            _fig_h.add_trace(go.Scatter(
                x=[0], y=[_Mr_H0],
                mode='markers+text',
                marker=dict(color='#a78bfa', size=11, symbol='circle',
                            line=dict(color='#fff', width=1.5)),
                text=[f'Mr = {_Mr_H0:.3f}'],
                textposition='top right',
                textfont=dict(color='#a78bfa', size=11),
                name='Remanencia (H=0)',
            ))

            # Referencia: ejes y campo coercitivo
            _fig_h.add_hline(y=0, line_dash='dash', line_color='#475569', line_width=1)
            _fig_h.add_vline(x=0, line_dash='dash', line_color='#475569', line_width=1)
            _fig_h.add_vline(
                x= _res['Hc_mT'], line_dash='dot',
                line_color='#f43f5e', line_width=1.5,
                annotation_text=f"+Hc = {_res['Hc_mT']:.1f} mT",
                annotation_position='top right',
                annotation_font_color='#f43f5e',
            )
            _fig_h.add_vline(
                x=-_res['Hc_mT'], line_dash='dot',
                line_color='#f43f5e', line_width=1.5,
                annotation_text=f"−Hc = {_res['Hc_mT']:.1f} mT",
                annotation_position='top left',
                annotation_font_color='#f43f5e',
            )

            _mat_name  = MATERIALS_DB[_sim_mat]['name']
            _geom_name = f"{GEOMETRY_MODES[_sim_geom]['emoji']} {GEOMETRY_MODES[_sim_geom]['name']}"
            _apply_plotly_theme(
                _fig_h,
                title=f'Lazo de Histéresis — {_geom_name}  ·  {_mat_name}  ·  {_sim_d} nm',
                xaxis_title='Campo aplicado  H  (mT)',
                yaxis_title='Magnetización reducida  M / Ms',
            )
            _fig_h.update_layout(
                height=450,
                yaxis=dict(range=[-1.18, 1.18]),
            )
            st.plotly_chart(_fig_h, use_container_width=True)

            # ── Comparación ML predicho vs. Simulado ─────────────────────────
            st.markdown('##### Comparación: ML predicho vs. Simulado')
            _Hc_ml, _Mr_ml = MODELS.predict_fast(
                float(_sim_d), _sim_mat,
                geom_factor_hc=GEOMETRY_MODES[_sim_geom]['factor_hc'],
                geom_factor_mr=GEOMETRY_MODES[_sim_geom]['factor_mr'],
            )
            _cmp_df = pd.DataFrame({
                'Fuente':   ['Simulación (OOMMF/SW)', 'ML Ensemble (predicho)'],
                'Hc (mT)':  [f"{_res['Hc_mT']:.1f}", f"{_Hc_ml:.1f}"],
                'Mr / Ms':  [f"{_res['Mr']:.3f}",     f"{_Mr_ml:.3f}"],
                'Motor':    [_res.get('runner', '—'), 'GBR + RF + MLP'],
            })
            st.dataframe(_cmp_df, use_container_width=True, hide_index=True)

            _Hc_err = abs(_res['Hc_mT'] - _Hc_ml) / max(_res['Hc_mT'], 1.0) * 100
            _Mr_err = abs(_res['Mr']    - _Mr_ml)  / max(_res['Mr'],    0.01) * 100
            st.caption(
                f'Diferencia relativa — '
                f'Hc: **{_Hc_err:.1f} %**  ·  Mr: **{_Mr_err:.1f} %**  '
                f'(modelo entrenado sobre histéresis LLG + datos OOMMF de referencia)'
            )

            # ── Validación contra datos reales (12nm.ipynb) ───────────────────
            if _REAL_DATA_OK and _ref_data is not None:
                _Hc_real, _Mr_real = _ref_data.extract_hc_mr()
                with st.expander(
                    '📊 Validación contra datos reales OOMMF (2 esferas Fe · 12nm.ipynb)',
                    expanded=(_sim_mat == 'fe')
                ):
                    st.info(
                        '**Referencia real**: simulación con ExeOOMMFRunner de 2 esferas '
                        f'de Fe (r=21 nm, sep=6 nm, celda=3 nm, ±400 mT).  '
                        f'Hc_real = **{_Hc_real:.1f} mT**  ·  Mr_real = **{_Mr_real:.4f}**'
                    )
                    _val_df = pd.DataFrame({
                        'Fuente': [
                            '🔬 OOMMF real (12nm.ipynb)',
                            f'⚙️ Simulación actual ({_res.get("runner","SW")})',
                            '🤖 ML Ensemble',
                        ],
                        'Hc (mT)': [
                            f'{_Hc_real:.1f}',
                            f"{_res['Hc_mT']:.1f}",
                            f'{_Hc_ml:.1f}',
                        ],
                        'Mr / Ms': [
                            f'{_Mr_real:.4f}',
                            f"{_res['Mr']:.4f}",
                            f'{_Mr_ml:.4f}',
                        ],
                        'Δ Hc vs OOMMF real': [
                            '0.0 %',
                            f"{abs(_res['Hc_mT']-_Hc_real)/max(_Hc_real,1)*100:.1f} %",
                            f'{abs(_Hc_ml-_Hc_real)/max(_Hc_real,1)*100:.1f} %',
                        ],
                        'Δ Mr vs OOMMF real': [
                            '0.000',
                            f"{abs(_res['Mr']-_Mr_real):.4f}",
                            f'{abs(_Mr_ml-_Mr_real):.4f}',
                        ],
                    })
                    st.dataframe(_val_df, use_container_width=True, hide_index=True)

                    # Mini-gráfica: histéresis real + SW actual superpuestas
                    _hist_r = _ref_data.load_hysteresis()
                    _fig_val = go.Figure()
                    _fig_val.add_trace(go.Scatter(
                        x=_hist_r['fd_desc'], y=_hist_r['mg_desc'],
                        mode='lines', name='OOMMF real ↓',
                        line=dict(color='#34d399', width=2.5),
                    ))
                    _fig_val.add_trace(go.Scatter(
                        x=_hist_r['fd_asc'], y=_hist_r['mg_asc'],
                        mode='lines', name='OOMMF real ↑',
                        line=dict(color='#34d399', width=2.5, dash='dash'),
                    ))
                    _nh2 = len(_res['H']) // 2
                    _fig_val.add_trace(go.Scatter(
                        x=_res['H'][:_nh2], y=_res['M'][:_nh2],
                        mode='lines', name=f'Simulación ({_res.get("runner","SW")}) ↓',
                        line=dict(color='#38bdf8', width=2, dash='dot'),
                    ))
                    _fig_val.add_trace(go.Scatter(
                        x=_res['H'][_nh2:], y=_res['M'][_nh2:],
                        mode='lines', name=f'Simulación ({_res.get("runner","SW")}) ↑',
                        line=dict(color='#fb923c', width=2, dash='dot'),
                    ))
                    _apply_plotly_theme(
                        _fig_val,
                        title='Histéresis: OOMMF real vs. simulación actual',
                        xaxis_title='Campo aplicado  H  (mT)',
                        yaxis_title='Magnetización reducida  M / Ms',
                    )
                    _fig_val.update_layout(
                        height=360,
                        yaxis=dict(range=[-1.18, 1.18]),
                    )
                    _fig_val.add_hline(y=0, line_dash='dash',
                                       line_color=_PALETTE['zero'], line_width=0.8)
                    _fig_val.add_vline(x=0, line_dash='dash',
                                       line_color=_PALETTE['zero'], line_width=0.8)
                    st.plotly_chart(_fig_val, use_container_width=True)

                    _rp = _ref_data.REFERENCE_PARAMS
                    st.caption(
                        f"Sistema real: {_rp['material']} · "
                        f"Ms={_rp['Ms_Am']/1e6:.2f} MA/m · "
                        f"K₁={_rp['K1_Jm3']/1e3:.0f} kJ/m³ · "
                        f"A={_rp['A_Jm']*1e12:.1f} pJ/m · "
                        f"Caja {_rp['box_nm'][0]}×{_rp['box_nm'][1]}×{_rp['box_nm'][2]} nm · "
                        f"Celda={_rp['cell_nm']:.0f} nm"
                    )

            # ── Código Ubermag reproducible (12nm.ipynb) ─────────────────────
            if _REAL_DATA_OK and _ref_data is not None:
                with st.expander('📓 Código Ubermag reproducible — `12nm.ipynb`'):
                    st.markdown(
                        'Código Python/Ubermag que reproduce la simulación de referencia. '
                        'Requiere `oommfc`, `discretisedfield`, `micromagneticmodel` y OOMMF instalado.'
                    )
                    st.code(_ref_data.NOTEBOOK_CODE, language='python')
                    st.caption(
                        '**Parámetros clave**: Ms=1.7 MA/m · K=48 kJ/m³ (cúbica) · '
                        'A=2.1×10⁻¹¹ J/m · r=21 nm · sep=6 nm · celda=3 nm · '
                        'Driver: TimeDriver t=5 ns/paso'
                    )

            # ── Parámetros del sistema simulado ───────────────────────────────
            with st.expander('📋 Parámetros del sistema micromagnético'):
                try:
                    import discretisedfield as _df
                    import micromagneticmodel as _mm
                    _mesh, _field, _sys = _uval.build_ubermag_field(
                        _sim_geom, float(_sim_d),
                        MATERIALS_DB[_sim_mat]['params'],
                        cell_nm=float(_sim_cell),
                    )
                    _n_valid = int((_field.norm.array > 0).sum())
                    _sys_str = (
                        f'System: {_sys.name}\n'
                        f'  Mesh:  {list(_mesh.n)} celdas × {_sim_cell} nm = '
                        f'{_mesh.n[0]*_sim_cell} × {_mesh.n[1]*_sim_cell}'
                        f' × {_mesh.n[2]*_sim_cell} nm\n'
                        f'  Celdas válidas: {_n_valid} / {int(_mesh.n.prod())}'
                        f'  ({100*_n_valid/_mesh.n.prod():.1f} %)\n'
                        f'  Energy terms: Exchange + Demag + UniaxialAnisotropy + Zeeman\n'
                        f'    A  = {MATERIALS_DB[_sim_mat]["params"]["A_pJ_m"]} pJ/m\n'
                        f'    K1 = {MATERIALS_DB[_sim_mat]["params"]["K1_kJ_m3"]} kJ/m³\n'
                        f'    Ms = {MATERIALS_DB[_sim_mat]["params"]["Ms_MA_m"]} MA/m\n'
                        f'  M inicial: +z (estado saturado)\n'
                        f'\nDriver:\n'
                        f'  MinDriver (relajación por paso H)\n'
                        f'  Barrido: +{_sim_Hmax} → −{_sim_Hmax} → +{_sim_Hmax} mT '
                        f'({_sim_steps} pasos)\n'
                    )
                    st.code(_sys_str, language='text')
                except Exception as _e:
                    st.info(f'Parámetros no disponibles (discretisedfield requerido): {_e}')

        st.divider()

        # ── Tabla de factores validados con Ubermag ───────────────────────────
        st.markdown('##### Factores validados con Ubermag (usados en el simulador)')
        _rows_vf = []
        for _gid, _vf in _uval.VALIDATED_FACTORS.items():
            _rows_vf.append({
                'Geometría':         _uval.GEOM_LABELS[_gid].split('(')[0].strip(),
                'factor_hc':         _vf['factor_hc'],
                'factor_mr':         _vf['factor_mr'],
                'N_z':               _vf['Nd_z'],
                'N_x':               _vf['Nd_x'],
                'Referencia física':  _vf['ref'],
            })
        st.dataframe(
            pd.DataFrame(_rows_vf), use_container_width=True, hide_index=True)
        st.success(
            '✅ Los factores mostrados arriba son los valores **actualmente usados** '
            'en `GEOMETRY_MODES` del simulador, derivados de cálculos analíticos '
            'validados con el stack Ubermag.'
        )

        st.divider()

        # ══════════════════════════════════════════════════════════════════════
        # ── 📂 Gestión dinámica de datos OOMMF ────────────────────────────────
        # ══════════════════════════════════════════════════════════════════════
        st.markdown('#### 📂 Datos OOMMF — Carga y Entrenamiento Dinámico')
        st.markdown(
            'El simulador detecta automáticamente nuevos archivos en `oommf_data/`. '
            'Sube archivos `.txt` (fd/mg) o `.ipynb` para ampliar el dataset de '
            'calibración y mejorar las predicciones ML.'
        )

        # ── Estado actual del dataset ─────────────────────────────────────────
        if _ref_data is not None:
            _dssum = _ref_data.dataset_summary()
        else:
            _dssum = {}

        _ds1, _ds2, _ds3, _ds4, _ds5 = st.columns(5)
        _ds1.metric('Ciclos de histéresis',  _dssum.get('n_hysteresis', 0))
        _ds2.metric('Series de energía',     _dssum.get('n_energies',   0))
        _ds3.metric('Notebooks (.ipynb)',     _dssum.get('n_notebooks',  0))
        _ds4.metric('Puntos calibración ML', _dssum.get('n_calibration', 0))
        _ds5.metric('Materiales detectados', len(_dssum.get('materials', [])))

        # ── Browser de datasets ───────────────────────────────────────────────
        with st.expander('🔍 Explorar datasets disponibles', expanded=False):
            if _REAL_DATA_OK and _ref_data is not None:
                import oommf_data_manager as _odm
                _all_ds = _odm.scan_datasets(_dssum.get('data_dir'))

                # Histéresis
                if _all_ds['hysteresis']:
                    st.markdown('**Ciclos de histéresis detectados**')
                    _h_rows = []
                    for _h in _all_ds['hysteresis']:
                        _h_rows.append({
                            'Archivo':    _h['filename'],
                            'Puntos':     _h['n_points'],
                            'Hc (mT)':   f"{_h.get('Hc_mT', '?'):.1f}",
                            'Mr / Ms':   f"{_h.get('Mr_Ms', '?'):.4f}",
                            'H_max (mT)': f"{_h.get('H_max_mT', '?'):.0f}",
                        })
                    st.dataframe(pd.DataFrame(_h_rows),
                                 use_container_width=True, hide_index=True)

                # Energías
                if _all_ds['energies']:
                    st.markdown('**Series de energía detectadas**')
                    _e_rows = [
                        {'Archivo': e['filename'], 'Tipo': e['dtype'],
                         'Label': e['label'], 'Puntos': e['n_points']}
                        for e in _all_ds['energies']
                    ]
                    st.dataframe(pd.DataFrame(_e_rows),
                                 use_container_width=True, hide_index=True)

                # Notebooks
                if _all_ds['notebooks']:
                    st.markdown('**Notebooks detectados**')
                    _nb_rows = [
                        {
                            'Archivo':  nb.get('source_nb', '?'),
                            'Material': nb.get('material_guess', '?'),
                            'Ms (MA/m)': f"{nb.get('Ms_Am', 0)/1e6:.2f}",
                            'K (kJ/m³)': f"{nb.get('K1_Jm3', 0)/1e3:.0f}",
                            'r (nm)':    f"{nb.get('radius_nm', '?')}",
                            'sep (nm)':  f"{nb.get('separation_nm', '?')}",
                            'celda (nm)': f"{nb.get('cell_nm', '?')}",
                            'Runner':    nb.get('runner', '?'),
                        }
                        for nb in _all_ds['notebooks']
                    ]
                    st.dataframe(pd.DataFrame(_nb_rows),
                                 use_container_width=True, hide_index=True)

                # Puntos de calibración
                _cal_pts = _odm.load_calibration_db()
                if _cal_pts:
                    st.markdown('**Puntos de calibración ML guardados**')
                    st.dataframe(
                        pd.DataFrame(_cal_pts),
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.caption(
                        'No hay puntos de calibración guardados. '
                        'Sube un ciclo de histéresis con material y diámetro '
                        'especificados para generar calibración.'
                    )
            else:
                st.info('Datos OOMMF no disponibles. Sube archivos para activar.')

        # ── Carga de nuevos archivos ──────────────────────────────────────────
        with st.expander('⬆️ Cargar nuevos datos OOMMF / Ubermag', expanded=False):
            st.markdown(
                'Sube archivos `.txt` (formato `fd  mg`, tab-separado) o '
                '`.ipynb` de Jupyter para ampliar el dataset. '
                'El sistema clasifica y parsea automáticamente cada archivo.'
            )
            _up_col1, _up_col2 = st.columns([2, 1])
            with _up_col1:
                _uploaded = st.file_uploader(
                    'Seleccionar archivo(s)',
                    type=['txt', 'ipynb'],
                    accept_multiple_files=True,
                    key='oommf_uploader',
                    help='fd = campo aplicado (mT) · mg = cantidad medida',
                )
            with _up_col2:
                _up_mat = st.selectbox(
                    'Material (opcional)',
                    ['— auto —'] + list(MATERIALS_DB.keys()),
                    format_func=lambda x: (
                        '— inferir del archivo —' if x == '— auto —'
                        else f"{MATERIALS_DB[x]['emoji']} {MATERIALS_DB[x]['name']}"
                    ),
                    key='up_mat',
                )
                _up_d = st.number_input(
                    'Diámetro partícula (nm)',
                    min_value=1.0, max_value=200.0,
                    value=42.0, step=1.0, key='up_d',
                    help='Usado para registrar el punto de calibración ML',
                )
                _up_geom = st.selectbox(
                    'Geometría',
                    list(GEOMETRY_MODES.keys()),
                    format_func=lambda g: f"{GEOMETRY_MODES[g]['emoji']}  {GEOMETRY_MODES[g]['name']}",
                    key='up_geom',
                )

            if _uploaded:
                _up_mat_id = None if _up_mat == '— auto —' else _up_mat
                _ingest_results = []
                for _uf in _uploaded:
                    # Guardar temporalmente
                    _tmp = f'/tmp/_oommf_upload_{_uf.name}'
                    with open(_tmp, 'wb') as _fh:
                        _fh.write(_uf.getbuffer())
                    # Ingestar
                    if _REAL_DATA_OK and _ref_data is not None:
                        _r = _ref_data.ingest_file(
                            src_path=_tmp,
                            mat_id=_up_mat_id,
                            d_nm=float(_up_d),
                            geom_id=_up_geom,
                        )
                    else:
                        import oommf_data_manager as _odm2
                        _r = _odm2.ingest_uploaded_file(
                            src_path=_tmp,
                            data_dir=_dssum.get('data_dir'),
                            mat_id=_up_mat_id,
                            d_nm=float(_up_d),
                            geom_id=_up_geom,
                        )
                    _ingest_results.append((_uf.name, _r))

                    # Invalidar cache de energía real
                    for _k in list(st.session_state.keys()):
                        if _k.startswith('fig_energy_real'):
                            del st.session_state[_k]

                for _fname, _r in _ingest_results:
                    if _r.get('status') == 'ok':
                        dtype  = _r.get('dtype', '?')
                        msg    = _r.get('message', '')
                        hp     = _r.get('hyst_params', {})
                        if hp:
                            st.success(
                                f'✅ **{_fname}** cargado · Tipo: `{dtype}` · '
                                f'Hc = **{hp["Hc_mT"]:.1f} mT** · '
                                f'Mr = **{hp["Mr_Ms"]:.4f}**'
                            )
                            if _r.get('calibration_saved'):
                                st.info(
                                    f'🎯 Punto de calibración guardado: '
                                    f'{_up_d:.0f} nm · {_up_mat_id or "auto"} · '
                                    f'{_up_geom}'
                                )
                        else:
                            st.success(f'✅ **{_fname}** · Tipo: `{dtype}` · {msg}')
                    else:
                        st.error(f'❌ **{_fname}**: {_r.get("message","Error")}')

        # ── Predicción con calibración OOMMF ─────────────────────────────────
        with st.expander(
            '🎯 Predicción ML con calibración de datos reales',
            expanded=False,
        ):
            st.markdown(
                'Combina el ensemble ML con puntos de calibración OOMMF reales '
                'usando interpolación gaussiana en el espacio de parámetros.'
            )
            _pcal_mat = st.selectbox(
                'Material',
                list(MATERIALS_DB.keys()),
                format_func=lambda m: (
                    f"{MATERIALS_DB[m]['emoji']} {MATERIALS_DB[m]['name']}"
                ),
                key='pcal_mat',
            )
            _pcal_d   = st.slider('Diámetro (nm)', 5, 100, 42, key='pcal_d')
            _pcal_g   = st.selectbox(
                'Geometría', list(GEOMETRY_MODES.keys()),
                format_func=lambda g: f"{GEOMETRY_MODES[g]['emoji']}  {GEOMETRY_MODES[g]['name']}",
                key='pcal_geom',
            )

            _Hc_cal, _Mr_cal, _cal_on = MODELS.predict_with_calibration(
                d_nm   = float(_pcal_d),
                mat_id = _pcal_mat,
                geom_id = _pcal_g,
                geom_factor_hc = GEOMETRY_MODES[_pcal_g]['factor_hc'],
                geom_factor_mr = GEOMETRY_MODES[_pcal_g]['factor_mr'],
                T = float(st.session_state.get('T_sim', 300)),
            )
            _Hc_base, _Mr_base = MODELS.predict_fast(
                float(_pcal_d), _pcal_mat,
                GEOMETRY_MODES[_pcal_g]['factor_hc'],
                GEOMETRY_MODES[_pcal_g]['factor_mr'],
                T=float(st.session_state.get('T_sim', 300)),
            )
            _pc1, _pc2, _pc3, _pc4 = st.columns(4)
            _pc1.metric('Hc ML puro (mT)',       f'{_Hc_base:.1f}')
            _pc2.metric('Mr ML puro',             f'{_Mr_base:.3f}')
            _pc3.metric(
                'Hc calibrado (mT)', f'{_Hc_cal:.1f}',
                delta=f'{_Hc_cal-_Hc_base:+.1f}' if _cal_on else None,
            )
            _pc4.metric(
                'Mr calibrado', f'{_Mr_cal:.3f}',
                delta=f'{_Mr_cal-_Mr_base:+.3f}' if _cal_on else None,
            )
            if _cal_on:
                st.success(
                    '🎯 Calibración OOMMF activa — predicción ajustada con '
                    'datos reales cercanos en el espacio de parámetros.'
                )
            else:
                st.caption(
                    'Sin calibración activa para este (material, diámetro, geometría). '
                    'Sube datos de histéresis OOMMF para activarla.'
                )
