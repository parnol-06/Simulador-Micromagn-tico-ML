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
from temperature_model import (
    to_kelvin,
    from_kelvin,
    apply_temperature_to_hc_mr,
    reduced_magnetization,
)
from materials_db import MATERIALS_DB, GEOMETRY_MODES

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
    page_title="SimuGOD — Micromagnetic ML Simulator",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
/* ── Scientific UI System — SimuGOD v4 ─────────────────────────────────────── */

/* Fonts */
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500&family=Inter:wght@400;500;600&display=swap');

/* Root variables */
:root {
  --bg:      #0d1117;
  --bg2:     #161b22;
  --bg3:     #1c2128;
  --border:  #30363d;
  --border2: #21262d;
  --text1:   #e6edf3;
  --text2:   #8b949e;
  --text3:   #484f58;
  --accent:  #2f81f7;
  --ok:      #3fb950;
  --warn:    #d29922;
  --danger:  #f85149;
  --mono:    'JetBrains Mono', 'Consolas', 'Courier New', monospace;
  --sans:    'Inter', 'Segoe UI', system-ui, sans-serif;
}

/* App backgrounds */
.stApp                              { background: var(--bg) !important; }
.stSidebar > div:first-child        { background: var(--bg2) !important;
                                      border-right: 1px solid var(--border) !important; }
section[data-testid="stSidebar"]    { background: var(--bg2) !important; }

/* Global typography */
body, .stApp { font-family: var(--sans) !important; }
h1  { font-family: var(--sans) !important; font-size: 14px !important;
      font-weight: 600 !important; color: var(--text1) !important;
      letter-spacing: 0.04em !important; }
h2  { font-family: var(--sans) !important; font-size: 12px !important;
      font-weight: 600 !important; color: var(--text2) !important;
      text-transform: uppercase; letter-spacing: 0.09em !important; }
h3  { font-family: var(--sans) !important; font-size: 11px !important;
      font-weight: 600 !important; color: var(--text3) !important;
      text-transform: uppercase; letter-spacing: 0.1em !important; }
h4  { font-family: var(--sans) !important; font-size: 12px !important;
      font-weight: 600 !important; color: var(--text2) !important; }
h5  { font-family: var(--sans) !important; font-size: 11px !important;
      font-weight: 600 !important; color: var(--text3) !important; }
p   { color: var(--text2) !important; font-size: 13px !important; }

/* Metric values — monospaced + prominent */
[data-testid="stMetricValue"] {
  font-family: var(--mono) !important;
  font-size: 22px !important;
  font-weight: 500 !important;
  color: var(--text1) !important;
  letter-spacing: -0.5px !important;
}
[data-testid="stMetricLabel"] {
  font-family: var(--mono) !important;
  font-size: 10px !important;
  font-weight: 400 !important;
  color: var(--text3) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.07em !important;
}
[data-testid="stMetricDelta"] {
  font-family: var(--mono) !important;
  font-size: 11px !important;
  color: var(--text3) !important;
}

/* Tabs — monospaced, minimal */
.stTabs [data-baseweb="tab-list"] {
  background: transparent !important;
  border-bottom: 1px solid var(--border) !important;
  gap: 0 !important;
}
.stTabs [data-baseweb="tab"] {
  font-family: var(--mono) !important;
  font-size: 11px !important;
  font-weight: 400 !important;
  color: var(--text3) !important;
  padding: 8px 16px !important;
  background: transparent !important;
  border: none !important;
  border-bottom: 2px solid transparent !important;
  letter-spacing: 0.02em !important;
}
.stTabs [data-baseweb="tab"]:hover {
  color: var(--text2) !important;
  background: transparent !important;
}
.stTabs [aria-selected="true"] {
  color: var(--text1) !important;
  border-bottom: 2px solid var(--accent) !important;
  background: transparent !important;
}
.stTabs [data-baseweb="tab-panel"] {
  background: transparent !important;
  padding-top: 20px !important;
}

/* Buttons — tool-style, not app-style */
.stButton > button {
  font-family: var(--mono) !important;
  font-size: 11px !important;
  font-weight: 500 !important;
  letter-spacing: 0.06em !important;
  background: var(--bg3) !important;
  color: var(--text2) !important;
  border: 1px solid var(--border) !important;
  border-radius: 4px !important;
  box-shadow: none !important;
  transition: border-color .12s, color .12s;
  text-transform: uppercase;
}
.stButton > button:hover {
  background: var(--bg3) !important;
  border-color: var(--accent) !important;
  color: var(--text1) !important;
  box-shadow: none !important;
}
/* Primary button (type='primary') */
.stButton > button[kind="primary"],
.stButton > button[data-testid*="primary"] {
  background: rgba(47,129,247,.08) !important;
  border-color: rgba(47,129,247,.4) !important;
  color: var(--accent) !important;
}
.stButton > button[kind="primary"]:hover {
  background: rgba(47,129,247,.15) !important;
}

/* Inputs / number_input / text_input */
.stNumberInput input,
.stTextInput  input {
  font-family: var(--mono) !important;
  font-size: 13px !important;
  background: var(--bg) !important;
  border: 1px solid var(--border) !important;
  border-radius: 3px !important;
  color: var(--text1) !important;
  box-shadow: none !important;
}
.stNumberInput input:focus,
.stTextInput  input:focus {
  border-color: var(--accent) !important;
  box-shadow: none !important;
}
.stNumberInput [data-testid="stNumberInputStepDown"],
.stNumberInput [data-testid="stNumberInputStepUp"] {
  background: var(--bg3) !important;
  border-color: var(--border) !important;
  color: var(--text2) !important;
}

/* Selectbox */
.stSelectbox > div > div {
  font-family: var(--mono) !important;
  font-size: 12px !important;
  background: var(--bg) !important;
  border: 1px solid var(--border) !important;
  border-radius: 3px !important;
  color: var(--text1) !important;
}

/* Radio buttons */
.stRadio > label {
  font-family: var(--mono) !important;
  font-size: 11px !important;
  color: var(--text3) !important;
  text-transform: uppercase;
  letter-spacing: 0.06em;
}
.stRadio [role="radiogroup"] label {
  font-family: var(--mono) !important;
  font-size: 12px !important;
  color: var(--text2) !important;
}

/* Toggle / checkbox */
.stCheckbox label, .stToggle label {
  font-family: var(--mono) !important;
  font-size: 11px !important;
  color: var(--text2) !important;
}
.stToggle [data-testid="stWidgetLabel"] {
  font-family: var(--mono) !important;
  font-size: 11px !important;
  color: var(--text2) !important;
}

/* Sidebar labels */
.stSidebar label,
.stSidebar .stMarkdown p,
.stSidebar .stCaption { color: var(--text2) !important; }

/* Dividers */
hr { border-color: var(--border2) !important; margin: 10px 0 !important; }

/* Expanders */
.streamlit-expanderHeader,
[data-testid="stExpander"] summary {
  font-family: var(--mono) !important;
  font-size: 10px !important;
  font-weight: 600 !important;
  color: var(--text3) !important;
  text-transform: uppercase;
  letter-spacing: 0.08em;
  background: var(--bg3) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 3px !important;
}

/* Alert / info / warning boxes */
.stAlert { border-radius: 4px !important; box-shadow: none !important; }
[data-testid="stAlert"] p {
  font-family: var(--mono) !important;
  font-size: 11px !important;
}

/* DataFrames / tables */
.stDataFrame { font-family: var(--mono) !important; font-size: 11px !important; }
.stDataFrame thead th { color: var(--text3) !important; font-size: 10px !important; }

/* Download button */
[data-testid="stDownloadButton"] button {
  font-family: var(--mono) !important;
  font-size: 11px !important;
  background: var(--bg3) !important;
  border: 1px solid var(--border) !important;
  color: var(--text2) !important;
  border-radius: 3px !important;
  letter-spacing: 0.05em;
  text-transform: uppercase;
}
[data-testid="stDownloadButton"] button:hover {
  border-color: var(--ok) !important;
  color: var(--ok) !important;
}

/* st.metric — override con sistema tipográfico propio */
[data-testid="stMetric"] {
  background: var(--bg3) !important;
  border: 1px solid var(--border2) !important;
  border-radius: 4px !important;
  padding: 10px 14px !important;
}
[data-testid="stMetricLabel"] p,
[data-testid="stMetricLabel"] {
  font-family: var(--mono) !important;
  font-size: 9px !important;
  font-weight: 600 !important;
  color: var(--text3) !important;
  text-transform: uppercase !important;
  letter-spacing: 0.08em !important;
}
[data-testid="stMetricValue"],
[data-testid="stMetricValue"] p {
  font-family: var(--mono) !important;
  font-size: 18px !important;
  font-weight: 500 !important;
  color: var(--text1) !important;
  letter-spacing: -0.3px !important;
}
[data-testid="stMetricDelta"] {
  font-family: var(--mono) !important;
  font-size: 10px !important;
}

/* Caption text */
.stCaption { font-family: var(--mono) !important; font-size: 10px !important;
             color: var(--text3) !important; }

/* Spinner */
.stSpinner p { font-family: var(--mono) !important; font-size: 11px !important;
               color: var(--text3) !important; }

/* Scrollbar */
::-webkit-scrollbar       { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--text3); }

/* ── Custom scientific classes ───────────────────────────────────────────── */

/* Compact header bar */
.sci-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 8px 0 12px;
  border-bottom: 1px solid var(--border2);
  margin-bottom: 16px;
}
.sci-header-name {
  font-family: var(--mono); font-size: 13px; font-weight: 600;
  color: var(--text1); letter-spacing: 0.08em;
}
.sci-header-version {
  font-family: var(--mono); font-size: 10px; color: var(--text3);
  margin-left: 10px;
}
.sci-header-ctx {
  font-family: var(--mono); font-size: 11px; color: var(--text3);
}
.sci-status {
  display: inline-flex; align-items: center; gap: 5px;
  font-family: var(--mono); font-size: 10px;
  padding: 3px 8px; border-radius: 3px;
}
.sci-status-dot { width: 5px; height: 5px; border-radius: 50%; }
.sci-status-idle  { background: rgba(72,79,88,.25); color: var(--text3); }
.sci-status-idle .sci-status-dot { background: var(--text3); }
.sci-status-ready { background: rgba(63,185,80,.1); color: var(--ok); }
.sci-status-ready .sci-status-dot { background: var(--ok); }
.sci-status-run   { background: rgba(47,129,247,.1); color: var(--accent); }
.sci-status-run   .sci-status-dot { background: var(--accent);
                                     animation: sci-pulse 1.2s infinite; }
@keyframes sci-pulse { 0%,100%{opacity:1} 50%{opacity:.25} }

/* Metric row (replaces 5-card home grid) */
.sci-metric-row {
  display: grid; grid-template-columns: repeat(5, 1fr);
  border: 1px solid var(--border2); border-radius: 4px;
  overflow: hidden; margin-bottom: 16px;
}
.sci-metric-cell {
  padding: 12px 14px;
  border-right: 1px solid var(--border2);
}
.sci-metric-cell:last-child { border-right: none; }
.sci-metric-label {
  font-family: var(--mono); font-size: 9px; font-weight: 600;
  color: var(--text3); text-transform: uppercase; letter-spacing: 0.1em;
  margin-bottom: 4px;
}
.sci-metric-value {
  font-family: var(--mono); font-size: 20px; font-weight: 500;
  color: var(--text1); letter-spacing: -0.5px; line-height: 1;
  margin-bottom: 3px;
}
.sci-metric-unit {
  font-family: var(--mono); font-size: 10px; color: var(--text3);
}
.sci-metric-sub {
  font-family: var(--mono); font-size: 9px; color: var(--text3);
  margin-top: 3px;
}
.sci-metric-cell.accent .sci-metric-value { color: var(--accent); }
.sci-metric-cell.ok     .sci-metric-value { color: var(--ok); }
.sci-metric-cell.warn   .sci-metric-value { color: var(--warn); }

/* Sidebar section label */
.sb-section {
  font-family: var(--mono); font-size: 9px; font-weight: 600;
  color: var(--text3); text-transform: uppercase; letter-spacing: 0.1em;
  padding: 6px 0 5px;
  border-bottom: 1px solid var(--border2);
  margin-bottom: 8px;
  margin-top: 4px;
}

/* Inline physics warning — no emoji */
.sci-warn {
  background: rgba(210,153,34,.07); border: 1px solid rgba(210,153,34,.25);
  border-left: 3px solid var(--warn); border-radius: 3px;
  padding: 7px 10px; margin: 8px 0;
  font-family: var(--mono); font-size: 10px; color: var(--warn);
  letter-spacing: 0.01em; line-height: 1.5;
}
.sci-info {
  background: rgba(47,129,247,.06); border: 1px solid rgba(47,129,247,.2);
  border-left: 3px solid var(--accent); border-radius: 3px;
  padding: 7px 10px; margin: 8px 0;
  font-family: var(--mono); font-size: 10px; color: var(--accent);
  line-height: 1.5;
}
.sci-ok {
  background: rgba(63,185,80,.06); border: 1px solid rgba(63,185,80,.2);
  border-left: 3px solid var(--ok); border-radius: 3px;
  padding: 7px 10px; margin: 8px 0;
  font-family: var(--mono); font-size: 10px; color: var(--ok);
  line-height: 1.5;
}

/* ── Simulation results 2×2 card grid ────────────────────────────────────── */

/* Outer grid */
.sim-grid {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 12px;
  margin-bottom: 14px;
}

/* Individual card */
.sim-card {
  background: var(--bg2);
  border: 1px solid var(--border2);
  border-radius: 4px;
  overflow: hidden;
}

/* Card header strip */
.sim-card-header {
  display: flex; align-items: center; justify-content: space-between;
  padding: 7px 12px 6px;
  border-bottom: 1px solid var(--border2);
  background: var(--bg3);
}
.sim-card-title {
  font-family: var(--mono); font-size: 10px; font-weight: 600;
  color: var(--text2); text-transform: uppercase; letter-spacing: 0.08em;
}
.sim-card-tag {
  font-family: var(--mono); font-size: 9px;
  color: var(--text3); letter-spacing: 0.04em;
}

/* Card body — wraps the Streamlit chart */
.sim-card-body {
  padding: 8px 10px 4px;
}

/* Bottom stats row inside a card */
.sim-card-stats {
  display: flex; gap: 16px;
  padding: 6px 12px 8px;
  border-top: 1px solid var(--border2);
}
.sim-stat-item {
  display: flex; flex-direction: column;
}
.sim-stat-label {
  font-family: var(--mono); font-size: 8px;
  color: var(--text3); text-transform: uppercase; letter-spacing: 0.07em;
}
.sim-stat-value {
  font-family: var(--mono); font-size: 13px; font-weight: 500;
  color: var(--text1); line-height: 1.2;
}

/* Material list item */
.mat-row {
  display: flex; justify-content: space-between; align-items: center;
  padding: 5px 8px; border-radius: 3px; margin-bottom: 2px;
  background: var(--bg3); border: 1px solid var(--border2);
  font-family: var(--mono); font-size: 12px; cursor: default;
}
.mat-row.selected {
  background: rgba(47,129,247,.08);
  border-color: rgba(47,129,247,.35);
}
.mat-row-sym   { color: var(--text1); font-weight: 500; }
.mat-row-name  { color: var(--text3); font-size: 10px; }

/* Section title inside tabs */
.sci-section-title {
  font-family: var(--mono); font-size: 10px; font-weight: 600;
  color: var(--text3); text-transform: uppercase; letter-spacing: 0.09em;
  padding-bottom: 8px; border-bottom: 1px solid var(--border2);
  margin-bottom: 16px;
}
</style>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  BASE DE DATOS DE MATERIALES Y GEOMETRÍAS  →  materials_db.py
# ═══════════════════════════════════════════════════════════════════════════════
# MATERIALS_DB y GEOMETRY_MODES se importan desde materials_db.py (fuente única
# de verdad, compartida con micromagnetic_simulator_v2.py).
# Ver: from materials_db import MATERIALS_DB, GEOMETRY_MODES  (arriba)

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


# ─────────────────────────────────────────────────────────────────────────────
#  Wrappers con corrección térmica post-ML  (temperature_model.py)
# ─────────────────────────────────────────────────────────────────────────────

def adapt_to_temperature(
    Hc_ref_mT: float,
    Mr_ref: float,
    *,
    d_nm: float,
    mat_id: str,
    T_K: float,
    use_temp_correction: bool = True,
) -> tuple:
    """Aplica corrección Bloch/Callen-Callen + SPM al par (Hc, Mr) predicho por ML.

    Si ``use_temp_correction`` es False devuelve los valores ML sin modificar.

    Returns
    -------
    tuple : (Hc_T [mT], Mr_T [0-1], barrier | None)
    """
    if not use_temp_correction:
        return float(Hc_ref_mT), float(Mr_ref), None

    p    = MATERIALS_DB[mat_id]['params']
    Tc   = float(p.get('Tc_K', 1.0))
    if float(T_K) >= Tc:
        return 0.0, 0.0, 0.0

    Hc_T, Mr_T, barrier = apply_temperature_to_hc_mr(
        Hc_ref_mT = float(Hc_ref_mT),
        Mr_ref    = float(Mr_ref),
        d_nm      = float(d_nm),
        Ms_MA_m   = float(p.get('Ms_MA_m', 1.0)),
        K1_kJ_m3  = float(p.get('K1_kJ_m3', 10.0)),
        Tc_K      = Tc,
        T_K       = float(T_K),
    )
    return float(Hc_T), float(Mr_T), float(barrier)


def predict_geom_temp(
    d_nm: float,
    mat_id: str,
    geom_id: str,
    engine: MicromagneticMLEngine,
    *,
    T_K: float,
    use_temp_correction: bool = True,
) -> tuple:
    """Predicción con corrección térmica post-ML (sin incertidumbre).

    El ML predice en T_sim=300 K y la física escala el resultado a T_K.

    Returns
    -------
    tuple : (Hc_T [mT], Mr_T)
    """
    gm = GEOMETRY_MODES[geom_id]
    Hc_ref, Mr_ref = engine.predict_fast(
        d_nm, mat_id,
        geom_factor_hc=gm['factor_hc'],
        geom_factor_mr=gm['factor_mr'],
        T=engine.T_sim,
    )
    Hc_T, Mr_T, _ = adapt_to_temperature(
        Hc_ref, Mr_ref,
        d_nm=d_nm, mat_id=mat_id,
        T_K=T_K, use_temp_correction=use_temp_correction,
    )
    return Hc_T, Mr_T


def predict_geom_with_uncertainty_temp(
    d_nm: float,
    mat_id: str,
    geom_id: str,
    engine: MicromagneticMLEngine,
    *,
    T_K: float,
    use_temp_correction: bool = True,
) -> tuple:
    """Predicción ensemble + incertidumbre ±1σ con corrección térmica post-ML.

    Returns
    -------
    tuple : (Hc_T, Mr_T, σHc, σMr, barrier)
    """
    gm = GEOMETRY_MODES[geom_id]
    Hc_ref, Mr_ref, sHc, sMr = engine.predict(
        d_nm, mat_id,
        geom_factor_hc=gm['factor_hc'],
        geom_factor_mr=gm['factor_mr'],
        T=engine.T_sim,
    )
    Hc_T, Mr_T, barrier = adapt_to_temperature(
        Hc_ref, Mr_ref,
        d_nm=d_nm, mat_id=mat_id,
        T_K=T_K, use_temp_correction=use_temp_correction,
    )
    # Escalar incertidumbre con la misma razón de reducción
    _ratio = (Hc_T / Hc_ref) if Hc_ref > 1e-9 else 0.0
    sHc_T = float(sHc) * _ratio
    sMr_T = float(sMr) * _ratio
    return Hc_T, Mr_T, sHc_T, sMr_T, barrier


def predict_batch_temp(
    sizes_nm: list,
    mat_id: str,
    engine: MicromagneticMLEngine,
    *,
    T_K: float,
    geom_id: str = 'sphere',
    use_temp_correction: bool = True,
) -> list:
    """Predicción en lote con corrección térmica.

    Returns
    -------
    list[tuple] : lista de (d_nm, Hc_T, Mr_T)
    """
    results = []
    for d in sizes_nm:
        Hc_T, Mr_T = predict_geom_temp(
            d, mat_id, geom_id, engine,
            T_K=T_K, use_temp_correction=use_temp_correction,
        )
        results.append((float(d), Hc_T, Mr_T))
    return results


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
    # Fondos — alineados con el CSS: --bg / --bg2 / --bg3
    'figure.facecolor': '#0d1117',   # --bg
    'axes.facecolor':   '#161b22',   # --bg2
    # Texto y etiquetas — --text1 / --text2
    'text.color':       '#e6edf3',
    'axes.labelcolor':  '#8b949e',
    'axes.titlecolor':  '#e6edf3',
    'axes.labelsize':   10,
    'axes.titlesize':   10,
    'axes.titlepad':    8,
    # Ticks — --text2 / --text3
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'xtick.labelsize':  8.5,
    'ytick.labelsize':  8.5,
    'xtick.major.size': 4,
    'ytick.major.size': 4,
    'xtick.direction':  'in',
    'ytick.direction':  'in',
    # Bordes y grilla — --border / --border2
    'axes.edgecolor':   '#30363d',
    'axes.linewidth':   0.6,
    'grid.color':       '#21262d',
    'grid.alpha':       0.6,
    'grid.linestyle':   '--',
    'grid.linewidth':   0.5,
    # Fuente — monospace para consistencia con el sistema tipográfico
    'font.family':      'monospace',
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
                         xaxis_title: str = '', yaxis_title: str = '',
                         height: int = 0) -> go.Figure:
    """Apply global theme to any go.Figure and set axis/chart titles."""
    layout = dict(_PLOTLY_LAYOUT)
    if title:
        layout['title'] = dict(text=title, x=0.5, xanchor='center', font_size=13)
    if xaxis_title:
        layout.setdefault('xaxis', {})['title'] = xaxis_title
    if yaxis_title:
        layout.setdefault('yaxis', {})['title'] = yaxis_title
    if height:
        layout['height'] = height
    fig.update_layout(**layout)
    return fig

# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURA PRINCIPAL  GridSpec 2×2
# ═══════════════════════════════════════════════════════════════════════════════

def build_main_figure(mat_id, d_nm, geom_id, models,
                      noise_level=0.008, dpi=150,
                      compare_mat=None, compare_geom=None,
                      T: float = 300.0,
                      use_temp_correction: bool = True):
    mat   = MATERIALS_DB[mat_id]
    gm    = GEOMETRY_MODES[geom_id]
    H_max = mat['field_max']
    Hc, Mr, sHc, sMr, _barrier = predict_geom_with_uncertainty_temp(
        d_nm, mat_id, geom_id, models,
        T_K=T, use_temp_correction=use_temp_correction)

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
        Hc2, Mr2   = predict_geom_temp(d_c, compare_mat, compare_geom, models,
                                      T_K=T, use_temp_correction=use_temp_correction)
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
    ax_hyst.set_ylabel('Reduced magnetization  M / Ms')
    ax_hyst.set_title(
        f'{mat["name"]}  ·  {gm["name"]}  ·  {d_nm:.0f} nm  ·  {T:.0f} K',
        fontweight='semibold')
    ax_hyst.legend(loc='lower right', framealpha=0.85)
    ax_hyst.grid(True)

    # ── Paisaje de energía ─────────────────────────────────────────────────
    en = energy_landscape(Hc, H_max=H_max)
    en_styles = [
        ('zeeman',   _PALETTE['accent2'],  '-',   'Zeeman'),
        ('exchange', _PALETTE['accent3'],  '-',   'Exchange'),
        ('demag',    _PALETTE['accent4'],  '--',  'Demagnetization'),
        ('aniso',    _PALETTE['accent1'],  '-.',  'Anisotropy'),
    ]
    for key, col, ls, lbl in en_styles:
        ax_enrg.plot(en['H'], en[key], color=col, lw=2.0, ls=ls, label=lbl)
    ax_enrg.axhline(0, color=_PALETTE['zero'], lw=0.8)
    ax_enrg.set_xlabel('Applied field  H  (mT)')
    ax_enrg.set_ylabel('E / E₀  (a.u.)')
    ax_enrg.set_title('Magnetic Energy Landscape', fontweight='semibold')
    ax_enrg.legend(loc='upper right', framealpha=0.85)
    ax_enrg.grid(True)

    # ── Tabla comparativa geometrías ────────────────────────────────────────
    ax_table.axis('off')
    p   = mat['params']
    ext = ' ⚠' if is_extrapolation(d_nm, mat_id) else ''
    headers = ['Geometry', 'Hc (mT)', '±σHc', 'Mr/Ms', '±σMr',
               'f_Hc', 'f_Mr', 'K₁ (kJ/m³)', 'A (pJ/m)', 'Ms (MA/m)']
    rows = []
    for gid, gdata in GEOMETRY_MODES.items():
        Hc_g, Mr_g, sHc_g, sMr_g, _ = predict_geom_with_uncertainty_temp(
            d_nm, mat_id, gid, models, T_K=T, use_temp_correction=use_temp_correction)
        is_sel = (gid == geom_id)
        _sel_prefix = '* ' if is_sel else ''
        _sel_suffix = ext  if is_sel else ''
        rows.append([
            f'{_sel_prefix}{gdata["name"]}{_sel_suffix}',
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
        f'Geometry Comparison  ·  {d_nm:.0f} nm  ·  {mat["name"]}  ·  T = {T:.0f} K',
        fontsize=9.5, color='#cbd5e1', pad=6, fontweight='semibold')

    fig.suptitle(
        'Simulador Micromagnético ML  ·  Ensemble RF + GBR',
        fontsize=10.5, color='#64748b', y=0.978,
        fontweight='normal', fontstyle='italic')
    return fig, Hc, Mr


# ═══════════════════════════════════════════════════════════════════════════════
#  CHART BUILDERS INDIVIDUALES  (para el dashboard 2×2)
# ═══════════════════════════════════════════════════════════════════════════════

def build_hysteresis_card(mat_id, d_nm, geom_id, Hc, Mr, sHc, sMr, T,
                           noise_level=0.008):
    """Histéresis interactiva Plotly con banda ±1σ y marcadores Hc/Mr."""
    mat   = MATERIALS_DB[mat_id]
    H_max = mat['field_max']

    H, M_up, M_dn = llg_hysteresis(Hc, Mr, H_max=H_max,
                                    noise_level=noise_level, seed=42)
    _, M_up_hi, _ = llg_hysteresis(Hc + sHc, min(Mr + sMr, 1.0),
                                    H_max=H_max, noise_level=0, seed=42)
    _, M_up_lo, _ = llg_hysteresis(max(Hc - sHc, 0.5), max(Mr - sMr, 0.05),
                                    H_max=H_max, noise_level=0, seed=42)

    _C  = _PALETTE['primary']   # azul principal
    _C2 = _PALETTE['accent2']   # naranja — Hc
    _C3 = _PALETTE['accent1']   # verde  — Mr

    fig = go.Figure()

    # Banda ±1σ (fill entre lo y hi de la rama ascendente)
    fig.add_trace(go.Scatter(
        x=np.concatenate([H, H[::-1]]),
        y=np.concatenate([M_up_hi, M_up_lo[::-1]]),
        fill='toself',
        fillcolor='rgba(79,195,247,0.12)',
        line=dict(width=0),
        name='±1σ band',
        hoverinfo='skip',
    ))

    # Rama ascendente
    fig.add_trace(go.Scatter(
        x=H, y=M_up,
        mode='lines',
        line=dict(color=_C, width=2.5),
        name=f'M↑  Hc={Hc:.1f} mT',
        hovertemplate='H = %{x:.1f} mT<br>M/Ms = %{y:.3f}<extra>↑ branch</extra>',
    ))

    # Rama descendente
    fig.add_trace(go.Scatter(
        x=H, y=M_dn,
        mode='lines',
        line=dict(color=_C, width=2.0, dash='dash'),
        name=f'M↓  Mr={Mr:.3f}',
        hovertemplate='H = %{x:.1f} mT<br>M/Ms = %{y:.3f}<extra>↓ branch</extra>',
    ))

    # Marcadores Hc (coercividad)
    fig.add_trace(go.Scatter(
        x=[Hc, -Hc], y=[0, 0],
        mode='markers',
        marker=dict(color=_C2, size=10, symbol='line-ns-open', line_width=2.5),
        name=f'Hc = ±{Hc:.1f} mT',
        hovertemplate='Hc = %{x:.1f} mT<extra>Coercivity</extra>',
    ))

    # Marcadores Mr (remanencia)
    fig.add_trace(go.Scatter(
        x=[0, 0], y=[Mr, -Mr],
        mode='markers',
        marker=dict(color=_C3, size=10, symbol='line-ew-open', line_width=2.5),
        name=f'Mr = {Mr:.3f}',
        hovertemplate='Mr/Ms = %{y:.3f}<extra>Remanence</extra>',
    ))

    # Líneas de referencia H=0, M=0
    fig.add_hline(y=0, line_color='#334155', line_width=0.8)
    fig.add_vline(x=0, line_color='#334155', line_width=0.8)

    _apply_plotly_theme(fig, xaxis_title='H (mT)', yaxis_title='M / Ms')
    fig.update_layout(
        height=340,
        margin=dict(l=52, r=16, t=16, b=44),
        legend=dict(orientation='h', yanchor='bottom', y=1.01,
                    xanchor='left', x=0, font_size=9),
        hovermode='x unified',
    )
    return fig


def build_energy_card(Hc, mat_id):
    """Paisaje de energía magnética interactivo Plotly."""
    mat   = MATERIALS_DB[mat_id]
    H_max = mat['field_max']
    en    = energy_landscape(Hc, H_max=H_max)

    en_styles = [
        ('zeeman',   _PALETTE['accent2'], 'solid',  'Zeeman'),
        ('exchange', _PALETTE['accent3'], 'solid',  'Exchange'),
        ('demag',    _PALETTE['accent4'], 'dash',   'Demag'),
        ('aniso',    _PALETTE['accent1'], 'dot',    'Anisotropy'),
    ]

    fig = go.Figure()
    for key, col, dash, lbl in en_styles:
        fig.add_trace(go.Scatter(
            x=en['H'], y=en[key],
            mode='lines',
            line=dict(color=col, width=2.2, dash=dash),
            name=lbl,
            hovertemplate=f'H = %{{x:.1f}} mT<br>{lbl} = %{{y:.3f}} E₀<extra>{lbl}</extra>',
        ))

    fig.add_hline(y=0, line_color='#334155', line_width=0.8)

    _apply_plotly_theme(fig, xaxis_title='H (mT)', yaxis_title='E / E₀ (a.u.)')
    fig.update_layout(
        height=340,
        margin=dict(l=52, r=16, t=16, b=44),
        legend=dict(orientation='h', yanchor='bottom', y=1.01,
                    xanchor='left', x=0, font_size=9),
        hovermode='x unified',
    )
    return fig


def build_magnetization_sweep_card(mat_id, geom_id, d_nm, models, T,
                                    use_temp_correction=True):
    """Curva Hc vs diámetro interactiva Plotly con punto actual resaltado."""
    mat    = MATERIALS_DB[mat_id]
    lo, hi = mat['range']
    sizes  = np.linspace(lo, hi, 60)
    Hc_b, _ = models.predict_batch(sizes, mat_id)
    gm     = GEOMETRY_MODES[geom_id]
    factor = gm['factor_hc']
    Hc_curve = Hc_b * factor

    Hc_cur, _, _, _, _ = predict_geom_with_uncertainty_temp(
        d_nm, mat_id, geom_id, models, T_K=T,
        use_temp_correction=use_temp_correction)

    fig = go.Figure()

    # Área de rango válido
    fig.add_vrect(x0=lo, x1=hi,
                  fillcolor=mat['color'], opacity=0.05,
                  layer='below', line_width=0)

    # Línea vertical del punto actual
    fig.add_vline(x=d_nm, line_color='#f1f5f9',
                  line_width=1.0, line_dash='dot', opacity=0.5)

    # Curva Hc(d)
    fig.add_trace(go.Scatter(
        x=sizes, y=Hc_curve,
        mode='lines',
        line=dict(color=mat['color'], width=2.5),
        name='Hc(d)',
        hovertemplate='d = %{x:.1f} nm<br>Hc = %{y:.1f} mT<extra>Hc(d)</extra>',
    ))

    # Punto actual
    fig.add_trace(go.Scatter(
        x=[d_nm], y=[Hc_cur],
        mode='markers+text',
        marker=dict(color='#f1f5f9', size=11,
                    line=dict(color=mat['color'], width=2)),
        text=[f'{d_nm:.0f} nm'],
        textposition='top right',
        textfont=dict(size=9, color='#e6edf3'),
        name=f'{d_nm:.0f} nm → {Hc_cur:.1f} mT',
        hovertemplate=(f'd = {d_nm:.1f} nm<br>'
                       f'Hc = {Hc_cur:.1f} mT<extra>Current point</extra>'),
    ))

    _apply_plotly_theme(fig, xaxis_title='Diameter (nm)', yaxis_title='Hc (mT)')
    fig.update_layout(
        height=340,
        margin=dict(l=52, r=16, t=16, b=44),
        legend=dict(orientation='h', yanchor='bottom', y=1.01,
                    xanchor='left', x=0, font_size=9),
        hovermode='x unified',
    )
    return fig


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
        hc_g, mr_g, shc_g, smr_g, _ = predict_geom_with_uncertainty_temp(
            d_nm, mat_id, gid, models, T_K=T, use_temp_correction=use_temp_correction)
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
with st.spinner('⚙️ Training GBR models for all 8 materials…'):
    MODELS = load_all_models()

# ═══════════════════════════════════════════════════════════════════════════════
#  BARRA LATERAL — CONTROLES COMPLETOS
# ═══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
<div style="padding:10px 0 6px;">
  <div style="font-family:var(--mono);font-size:13px;font-weight:600;
              color:var(--text1);letter-spacing:0.08em;">SIMUGOD</div>
  <div style="font-family:var(--mono);font-size:9px;color:var(--text3);
              margin-top:2px;">
Micromagnetic ML Simulator · © 2026 Arnol Perez and Jesus Cabezas. All rights reserved.
</div>
""", unsafe_allow_html=True)
    st.divider()

    # ── MATERIAL ──────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">MATERIAL</div>', unsafe_allow_html=True)
    mat_id = st.selectbox(
        'Material',
        list(MATERIALS_DB.keys()),
        index=list(MATERIALS_DB.keys()).index(st.session_state.mat_id),
        format_func=lambda x: (
            f"{x.upper():12}  {MATERIALS_DB[x]['name'].split('(')[0].strip()}"
        ),
        label_visibility='collapsed',
        key='sb_material',
    )
    st.session_state.mat_id = mat_id
    mat   = MATERIALS_DB[mat_id]
    lo, hi = mat['range']

    with st.expander('PARAMETERS'):
        p = mat['params']
        st.markdown(f"""
<div style="font-family:var(--mono);font-size:10px;color:var(--text3);line-height:2;">
  K₁&nbsp;&nbsp;{p['K1_kJ_m3']} kJ/m³<br>
  A&nbsp;&nbsp;&nbsp;{p['A_pJ_m']} pJ/m<br>
  Ms&nbsp;&nbsp;{p['Ms_MA_m']} MA/m<br>
  α&nbsp;&nbsp;&nbsp;{p['alpha']}<br>
  λₑₓ&nbsp;{p['lambda_ex_nm']} nm<br>
  Tc&nbsp;&nbsp;{p['Tc_K']} K<br>
  ML&nbsp;&nbsp;{lo}–{hi} nm
</div>""", unsafe_allow_html=True)

    st.divider()

    # ── GEOMETRY ──────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">GEOMETRY</div>', unsafe_allow_html=True)
    geom_id = st.selectbox(
        'Geometry',
        list(GEOMETRY_MODES.keys()),
        index=list(GEOMETRY_MODES.keys()).index(st.session_state.geom_id),
        format_func=lambda x: f"{GEOMETRY_MODES[x]['name']}  (Nd={GEOMETRY_MODES[x].get('Nd','')})",
        label_visibility='collapsed',
        key='sb_geom',
    )
    st.session_state.geom_id = geom_id
    gm = GEOMETRY_MODES[geom_id]
    st.markdown(f"""
<div style="font-family:var(--mono);font-size:10px;color:var(--text3);
            padding:4px 0 2px;line-height:1.8;">
  f_Hc = ×{gm["factor_hc"]} &nbsp;·&nbsp; f_Mr = ×{gm["factor_mr"]}
</div>""", unsafe_allow_html=True)

    st.divider()

    # ── PARAMETERS ────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">PARAMETERS</div>', unsafe_allow_html=True)
    d_nm = st.number_input(
        'd — diameter (nm)',
        min_value=float(max(2, lo - 15)),
        max_value=float(hi + 15),
        value=float(st.session_state.d_nm),
        step=1.0,
        key='sb_size',
    )
    st.session_state.d_nm = d_nm
    if is_extrapolation(d_nm, mat_id):
        st.markdown(
            f'<div class="sci-warn">extrapolation — outside ML range [{lo}–{hi} nm]</div>',
            unsafe_allow_html=True,
        )

    st.divider()

    # ── OVERLAY ───────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">OVERLAY COMPARE</div>', unsafe_allow_html=True)
    compare_enabled = st.toggle('Enable overlay', key='sb_overlay')
    compare_mat, compare_geom = None, None
    if compare_enabled:
        c_opts = {k: v for k, v in MATERIALS_DB.items() if k != mat_id}
        compare_mat = st.selectbox(
            'Material B',
            list(c_opts.keys()),
            format_func=lambda x: (
                f"{x.upper():12}  {MATERIALS_DB[x]['name'].split('(')[0].strip()}"
            ),
            key='sb_cmat',
        )
        compare_geom = st.selectbox(
            'Geometry B',
            list(GEOMETRY_MODES.keys()),
            format_func=lambda x: GEOMETRY_MODES[x]['name'],
            key='sb_cgeom',
        )

    st.divider()

    # ── TEMPERATURE ───────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">TEMPERATURE</div>', unsafe_allow_html=True)
    _Tc_mat = mat['params']['Tc_K']
    _T_max  = int(min(_Tc_mat - 1, 1400))

    # Rangos en ambas unidades
    _K_min, _K_max = 1,          _T_max
    _C_min, _C_max = _K_min - 273, _K_max - 273   # −272 °C … Tc-273 °C

    # ── Callbacks para sincronizar K ↔ °C sin perder el valor ─────────────────
    def _on_temp_unit_change():
        """Convierte el valor actual a la nueva unidad antes del re-render."""
        cur_K  = float(st.session_state.get('T_sim', 300))
        new_unit = st.session_state.get('sb_temp_unit', 'K')
        if new_unit == '°C':
            st.session_state['sb_temp_value'] = float(
                np.clip(cur_K - 273.15, _C_min, _C_max))
        else:
            st.session_state['sb_temp_value'] = float(
                np.clip(cur_K, _K_min, _K_max))

    def _on_temp_value_change():
        """Actualiza T_sim en Kelvin cada vez que cambia el number_input."""
        raw   = float(st.session_state.get('sb_temp_value', 300))
        unit  = st.session_state.get('sb_temp_unit', 'K')
        T_k   = (raw + 273.15) if unit == '°C' else raw
        st.session_state['T_sim'] = float(np.clip(T_k, _K_min, _K_max))

    # Inicializar keys si es la primera ejecución
    if 'sb_temp_unit' not in st.session_state:
        st.session_state['sb_temp_unit']  = 'K'
    if 'sb_temp_value' not in st.session_state:
        st.session_state['sb_temp_value'] = float(
            st.session_state.get('T_sim', 300))

    # ── Selector de unidad (radio horizontal) ─────────────────────────────────
    _cur_unit = st.radio(
        'Unidad de temperatura',
        ['K', '°C'],
        index=0 if st.session_state['sb_temp_unit'] == 'K' else 1,
        horizontal=True,
        key='sb_temp_unit',
        on_change=_on_temp_unit_change,
        label_visibility='collapsed',
    )

    # ── number_input con botones + / − ────────────────────────────────────────
    if _cur_unit == '°C':
        _ni_min, _ni_max = float(_C_min), float(_C_max)
        _ni_label = f'Temperature (°C)  [min {_C_min} · max {_C_max}]'
    else:
        _ni_min, _ni_max = float(_K_min), float(_K_max)
        _ni_label = f'Temperature (K)  [min {_K_min} · max {_K_max}]'

    _raw_val = st.number_input(
        _ni_label,
        min_value=_ni_min,
        max_value=_ni_max,
        value=float(np.clip(st.session_state['sb_temp_value'], _ni_min, _ni_max)),
        step=1.0,
        key='sb_temp_value',
        on_change=_on_temp_value_change,
        label_visibility='visible',
    )

    # Sincronizar T_sim desde el valor actual
    _T_K_raw = (_raw_val + 273.15) if _cur_unit == '°C' else _raw_val
    T_sim    = int(np.clip(_T_K_raw, _K_min, _K_max))
    st.session_state['T_sim'] = T_sim

    # ── Thermal state badge ───────────────────────────────────────────────────
    _T_C   = T_sim - 273.15
    _T_red = float(T_sim) / float(_Tc_mat)

    # Magnetización reducida y estado del régimen
    _ms_t = float(np.clip((1 - _T_red**1.5)**(1/3), 0.0, 1.0)) if _T_red < 1.0 else 0.0

    # Barrera SPM estimada con Hc rápido (sin corrección térmica) y diámetro actual
    _d_cur  = float(st.session_state.get('sb_size', 20))
    _Hc_est, _ = MODELS.predict_fast(_d_cur, mat_id)
    _Hc_est = float(_Hc_est) * gm.get('factor_hc', 1.0)
    _p_mat  = mat['params']
    _mu0    = 4.0 * np.pi * 1e-7
    _Ms_T   = _p_mat['Ms_MA_m'] * _ms_t
    _r_m    = (_d_cur / 2.0) * 1e-9
    _V_m3   = (4.0 / 3.0) * np.pi * _r_m**3
    _Hc_Am  = (_Hc_est * _ms_t**(7/3) * 1e-3) / _mu0
    _K_eff  = 0.5 * _mu0 * _Ms_T * 1e6 * _Hc_Am
    _kB     = 1.380649e-23
    _Eb     = max(0.0, _K_eff * _V_m3)
    _barrier= _Eb / (_kB * max(1.0, T_sim)) if _Eb > 0 else 0.0

    # Elegir clase y texto según régimen
    if _T_red >= 1.0:
        _th_cls, _th_label, _th_regime = 'sci-warn', f'T ≥ Tc = {_Tc_mat} K', 'PARAMAGNETIC'
    elif _barrier < 25.0 and _d_cur < 30:
        _th_cls, _th_label, _th_regime = 'sci-warn', f'Eb/kBT = {_barrier:.1f} < 25', 'SPM RISK'
    elif _T_red > 0.80:
        _th_cls, _th_label, _th_regime = 'sci-warn', f'T/Tc = {_T_red:.3f}', 'HIGH T'
    elif _T_red > 0.50:
        _th_cls, _th_label, _th_regime = 'sci-info', f'T/Tc = {_T_red:.3f}', 'MODERATE T'
    else:
        _th_cls, _th_label, _th_regime = 'sci-ok',   f'T/Tc = {_T_red:.3f}', 'STABLE'

    st.markdown(f"""
<div class="{_th_cls}" style="display:flex;justify-content:space-between;
     align-items:baseline;padding:6px 10px;margin:6px 0 4px;">
  <span style="font-weight:600;letter-spacing:0.06em">{_th_regime}</span>
  <span style="font-size:9px;opacity:.8">{_th_label}</span>
</div>
<div style="font-family:var(--mono);font-size:9px;color:var(--text3);
            padding:2px 0 4px;line-height:1.9;">
  ms(T) = {_ms_t:.4f} &nbsp;&middot;&nbsp;
  {T_sim}&nbsp;K&nbsp;/&nbsp;{_T_C:.0f}&nbsp;&deg;C &nbsp;&middot;&nbsp;
  Tc = {_Tc_mat}&nbsp;K
  {'&nbsp;&middot;&nbsp; Eb/kBT = ' + f'{_barrier:.1f}' if _barrier > 0 else ''}
</div>""", unsafe_allow_html=True)

    with st.expander('EFFECTIVE PROPERTIES AT T'):
        _tau  = float(np.clip(T_sim / _Tc_mat, 0.0, 0.9999))
        _ms_t = float(np.clip((1 - _tau**1.5)**(1/3), 0.0, 1.0))
        _k1_t = float(np.clip(_ms_t**(10/3), 0.0, 1.0))
        _a_t  = float(np.clip(_ms_t**2, 0.0, 1.0))
        _p    = mat['params']
        _T_label = f'{_T_C:.0f} °C' if st.session_state.get('sb_temp_unit', 'K') == '°C' else f'{T_sim} K'
        st.markdown(f"""
<div style="font-family:var(--mono);font-size:10px;color:var(--text3);line-height:2.1;">
  Ms&nbsp;&nbsp;&nbsp;{_p['Ms_MA_m']:.3f} → {_p['Ms_MA_m']*_ms_t:.3f} MA/m<br>
  K₁&nbsp;&nbsp;&nbsp;{_p['K1_kJ_m3']:.1f} → {_p['K1_kJ_m3']*_k1_t:.1f} kJ/m³<br>
  A&nbsp;&nbsp;&nbsp;&nbsp;{_p['A_pJ_m']:.2f} → {_p['A_pJ_m']*_a_t:.2f} pJ/m<br>
  ms(T) = {_ms_t:.4f} &nbsp;·&nbsp; k1(T) = {_k1_t:.4f}
</div>""", unsafe_allow_html=True)
        st.markdown(
            '<div style="font-family:var(--mono);font-size:9px;color:var(--text3);'
            'margin-top:6px;">Bloch (1930) · Callen-Callen (1966)</div>',
            unsafe_allow_html=True)

    # ── Corrección térmica post-ML ────────────────────────────────────────────
    if 'use_temp_correction' not in st.session_state:
        st.session_state['use_temp_correction'] = True
    use_temp_correction = st.toggle(
        'Thermal correction (Bloch / SPM / Tc)',
        value=st.session_state['use_temp_correction'],
        key='use_temp_correction',
        help=(
            'Applies Bloch/Callen-Callen laws and the Néel SPM criterion '
            'to the ML prediction. Disable to see raw model output '
            'without temperature scaling.'
        ),
    )

    st.divider()

    # ── ADVANCED ─────────────────────────────────────────────────────────────
    with st.expander('ADVANCED'):
        noise_level = st.slider('LLG noise', 0.0, 0.05, 0.008, 0.001,
                                 format='%.3f')
        export_dpi  = st.select_slider('Export DPI', [100, 150, 170, 200, 300], value=170)

    st.divider()

    # ── ACTIONS ──────────────────────────────────────────────────────────────
    btn_sim     = st.button('RUN SIMULATION', use_container_width=True, type='primary')
    btn_animate = st.button('ANIMATE BY SIZE', use_container_width=True)
    btn_clear   = st.button('CLEAR HISTORY',   use_container_width=True)
    if btn_clear:
        st.session_state.history = []
        _db.clear_simulations()
        st.rerun()
    if btn_sim:
        st.session_state.sim_done = True

    st.divider()

    # ── OOMMF DATA ────────────────────────────────────────────────────────────
    st.markdown('<div class="sb-section">OOMMF DATA</div>', unsafe_allow_html=True)

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

    # Status de datos
    if _sb_n_h > 0 or _sb_n_e > 0:
        st.markdown(
            f'<div class="sci-ok">'
            f'{_sb_n_h} hysteresis · {_sb_n_e} energies<br>'
            f'{_sb_n_nb} notebooks · {_sb_n_cal} calibrations</div>',
            unsafe_allow_html=True,
        )
    elif _sb_summary_err:
        st.markdown(
            f'<div class="sci-warn">read error: {_sb_summary_err[:80]}</div>',
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
        'Upload .txt / .ipynb files',
        type=['txt', 'ipynb'],
        accept_multiple_files=True,
        key='sb_oommf_uploader',
        help='OOMMF energy/hysteresis .txt files or simulation .ipynb notebooks',
        label_visibility='collapsed',
    )

    with st.expander('UPLOAD OPTIONS'):
        _sb_mat = st.selectbox(
            'Material',
            ['— inferir —'] + list(MATERIALS_DB.keys()),
            format_func=lambda x: (
                'infer from file' if x == '— inferir —'
                else f"{x.upper()}  {MATERIALS_DB[x]['name'].split('(')[0].strip()}"
            ),
            key='sb_up_mat',
        )
        _sb_d = st.number_input(
            'Diameter (nm)', min_value=1.0, max_value=500.0,
            value=42.0, step=1.0, key='sb_up_d',
        )
        _sb_geom_up = st.selectbox(
            'Geometry',
            list(GEOMETRY_MODES.keys()),
            format_func=lambda g: GEOMETRY_MODES[g]['name'],
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
                                f'Calibration saved: '
                                f'{_sb_d:.0f} nm · {_sb_mat_id or "auto"}'
                            )
                        _any_hyst = True
                    else:
                        st.success(
                            f'{_sbf.name} — Type: `{_dtype_sb}`'
                        )
                else:
                    st.error(f'Error — {_sbf.name}: {_r_sb.get("message","Error")}')
            except Exception as _e_sb:
                st.error(f'Error — {_sbf.name}: {_e_sb}')

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
#  VARIABLES GLOBALES POST-SIDEBAR
#  T_K y use_temp_correction se leen una vez aquí y se usan en toda la UI.
# ═══════════════════════════════════════════════════════════════════════════════
T_K: float               = float(st.session_state.get('T_sim', 300))
use_temp_correction: bool = bool(st.session_state.get('use_temp_correction', True))

# ═══════════════════════════════════════════════════════════════════════════════
#  PANTALLA PRINCIPAL — HOME CARD  (Dashboard único)
# ═══════════════════════════════════════════════════════════════════════════════

# ── Header compacto ───────────────────────────────────────────────────────────
_st_class = 'sci-status-ready' if st.session_state.sim_done else 'sci-status-idle'
_st_text  = 'ready'            if st.session_state.sim_done else 'idle'
_T_C_disp = T_K - 273.15
st.markdown(f"""
<div class="sci-header">
  <div>
    <span class="sci-header-name">SIMUGOD</span>
    <span class="sci-header-version">Micromagnetic ML Simulator · v4.0 · Ensemble RF+GBR+MLP</span>
  </div>
  <div style="display:flex;align-items:center;gap:16px;">
    <span class="sci-header-ctx">
      {mat_id.upper()} &nbsp;/&nbsp; {geom_id} &nbsp;/&nbsp; {d_nm}&nbsp;nm
      &nbsp;/&nbsp; {int(T_K)}&nbsp;K&nbsp;({_T_C_disp:.0f}&nbsp;°C)
    </span>
    <span class="sci-status {_st_class}">
      <span class="sci-status-dot"></span>{_st_text}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════════
#  FIGURA PRINCIPAL (calcula una vez)
# ═══════════════════════════════════════════════════════════════════════════════
if not st.session_state.sim_done:
    # ── CONFIG PREVIEW antes de la primera simulación ─────────────────────────
    st.markdown(f"""
<div class="sci-metric-row">
  <div class="sci-metric-cell">
    <div class="sci-metric-label">MATERIAL</div>
    <div class="sci-metric-value" style="color:{mat['color']}">{mat_id.upper()}</div>
    <div class="sci-metric-sub">{mat['category']}</div>
  </div>
  <div class="sci-metric-cell">
    <div class="sci-metric-label">GEOMETRY</div>
    <div class="sci-metric-value">{gm['name']}</div>
    <div class="sci-metric-sub">f_Hc &times; {gm['factor_hc']} &nbsp;&middot;&nbsp; Nd={gm.get('Nd_z','—')}</div>
  </div>
  <div class="sci-metric-cell">
    <div class="sci-metric-label">DIAMETER</div>
    <div class="sci-metric-value" style="color:var(--accent)">{d_nm} <span style="font-size:12px">nm</span></div>
    <div class="sci-metric-sub">range {lo}–{hi} nm</div>
  </div>
  <div class="sci-metric-cell">
    <div class="sci-metric-label">Hc (mT)</div>
    <div class="sci-metric-value" style="color:var(--text3)">—</div>
    <div class="sci-metric-sub">press RUN SIMULATION</div>
  </div>
  <div class="sci-metric-cell">
    <div class="sci-metric-label">Mr / Ms</div>
    <div class="sci-metric-value" style="color:var(--text3)">—</div>
    <div class="sci-metric-sub">press RUN SIMULATION</div>
  </div>
</div>
""", unsafe_allow_html=True)
    st.info('Configure material, geometry and size in the sidebar, then press RUN SIMULATION.')
    st.stop()

t0 = time.perf_counter()
fig_main, Hc_val, Mr_val = build_main_figure(
    mat_id, d_nm, geom_id, MODELS,
    noise_level=noise_level, dpi=export_dpi,
    compare_mat=compare_mat if compare_enabled else None,
    compare_geom=compare_geom if compare_enabled else None,
    T=T_K,
    use_temp_correction=use_temp_correction,
)
elapsed_ms = (time.perf_counter() - t0) * 1000

# ── Registro en historial + SQLite ────────────────────────────────────────────
entry = {
    'Hora': datetime.now().strftime('%H:%M:%S'),
    'Material': mat['name'], 'Geometry': gm['name'],
    'Size (nm)': d_nm,
    'Hc (mT)': round(Hc_val, 1), 'Mr/Ms': round(Mr_val, 3),
    'Extrapol.': '!' if is_extrapolation(d_nm, mat_id) else 'ok',
}
_last_key = f'{mat_id}_{geom_id}_{d_nm}'
if (not st.session_state.history or
        st.session_state.history[-1].get('Size (nm)') != d_nm or
        st.session_state.history[-1].get('Material') != mat['name'] or
        st.session_state.history[-1].get('Geometry') != gm['name']):
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

# ── HOME CARD — resultados de la simulación actual ───────────────────────────
# Se renderiza aquí (post-build_main_figure) para usar Hc_val/Mr_val reales.
# Una sola llamada a predict_geom_with_uncertainty_temp solo para las sigmas.
_, _, _sHc, _sMr, _ = predict_geom_with_uncertainty_temp(
    d_nm, mat_id, geom_id, MODELS,
    T_K=T_K, use_temp_correction=use_temp_correction)

st.markdown(f"""
<div class="sci-metric-row">
  <div class="sci-metric-cell">
    <div class="sci-metric-label">MATERIAL</div>
    <div class="sci-metric-value" style="color:{mat['color']}">{mat_id.upper()}</div>
    <div class="sci-metric-sub">{mat['category']}</div>
  </div>
  <div class="sci-metric-cell">
    <div class="sci-metric-label">GEOMETRY</div>
    <div class="sci-metric-value">{gm['name']}</div>
    <div class="sci-metric-sub">f_Hc &times; {gm['factor_hc']} &nbsp;&middot;&nbsp; Nd={gm.get('Nd_z','—')}</div>
  </div>
  <div class="sci-metric-cell">
    <div class="sci-metric-label">DIAMETER</div>
    <div class="sci-metric-value" style="color:var(--accent)">{d_nm} <span style="font-size:12px">nm</span></div>
    <div class="sci-metric-sub">range {lo}–{hi} nm</div>
  </div>
  <div class="sci-metric-cell">
    <div class="sci-metric-label">Hc (mT)</div>
    <div class="sci-metric-value" style="color:var(--accent)">{Hc_val:.1f} <span style="font-size:11px;color:var(--text3)">&plusmn;&thinsp;{_sHc:.1f}</span></div>
    <div class="sci-metric-sub">simulation &plusmn; &sigma;</div>
  </div>
  <div class="sci-metric-cell">
    <div class="sci-metric-label">Mr / Ms</div>
    <div class="sci-metric-value" style="color:var(--ok)">{Mr_val:.3f} <span style="font-size:11px;color:var(--text3)">&plusmn;&thinsp;{_sMr:.3f}</span></div>
    <div class="sci-metric-sub">simulation &plusmn; &sigma;</div>
  </div>
</div>
""", unsafe_allow_html=True)

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
        # ── Dataset cargado: banner compacto con sci-metric-row ───────────────
        _bn_hc_r = _bn_sum.get('hc_range', (0, 0))
        _bn_mr_r = _bn_sum.get('mr_range', (0, 0))
        _bn_mats = ', '.join(_bn_sum.get('materials', ['?'])) or '—'

        # Dataset detail line (hysteresis filenames)
        _bn_detail = ''
        if _bn_n_h > 0:
            try:
                _bn_hds = _odm_banner.scan_datasets(_bn_dir)['hysteresis']
                _bn_detail = '  ·  '.join(
                    f"{h['filename'].split('.')[0]}: "
                    f"Hc={h.get('Hc_mT','?'):.1f} mT  Mr={h.get('Mr_Ms','?'):.4f}"
                    for h in _bn_hds
                )
            except Exception:
                pass

        st.markdown(f"""
<div class="sci-metric-row" style="border-left:3px solid var(--ok);">
  <div class="sci-metric-cell" style="flex:2;min-width:0;">
    <div class="sci-metric-label" style="color:var(--ok)">OOMMF DATA ACTIVE</div>
    <div class="sci-metric-value" style="font-size:13px;color:var(--text1)">oommf_data/</div>
    <div class="sci-metric-sub">{_bn_mats}</div>
  </div>
  <div class="sci-metric-cell">
    <div class="sci-metric-label">HYSTERESIS</div>
    <div class="sci-metric-value" style="color:var(--ok)">{_bn_n_h}</div>
    <div class="sci-metric-sub">cycles</div>
  </div>
  <div class="sci-metric-cell">
    <div class="sci-metric-label">ENERGIES</div>
    <div class="sci-metric-value">{_bn_n_e}</div>
    <div class="sci-metric-sub">series</div>
  </div>
  <div class="sci-metric-cell">
    <div class="sci-metric-label">NOTEBOOKS</div>
    <div class="sci-metric-value">{_bn_n_nb}</div>
    <div class="sci-metric-sub">.ipynb</div>
  </div>
  <div class="sci-metric-cell">
    <div class="sci-metric-label">ML CALIBRATION</div>
    <div class="sci-metric-value" style="color:var(--accent)">{_bn_n_cal}</div>
    <div class="sci-metric-sub">points</div>
  </div>
</div>
{'<div style="font-family:var(--mono);font-size:9px;color:var(--text3);padding:3px 0 8px;">' + _bn_detail + '</div>' if _bn_detail else ''}
""", unsafe_allow_html=True)
    else:
        # ── Sin datos ─────────────────────────────────────────────────────────
        st.markdown(
            '<div class="sci-info" style="border-left-color:var(--text3);'
            'color:var(--text3);">'
            'No OOMMF data loaded — upload <code>.txt</code> (fd/mg) or '
            '<code>.ipynb</code> files from the sidebar to enable real energy '
            'visualization and ML calibration.'
            '</div>',
            unsafe_allow_html=True,
        )
except Exception:
    pass

# ═══════════════════════════════════════════════════════════════════════════════
#  TABS DE RESULTADOS
# ═══════════════════════════════════════════════════════════════════════════════
(tab_sim, tab_compare, tab_params,
 tab_3d, tab_dashboard, tab_export, tab_uval) = st.tabs([
    'Simulation',
    'Compare',
    'ML Parameters',
    '3D Visualization',
    'Dashboard',
    'Export',
    'Validation',
])

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 1 — SIMULACIÓN
# ─────────────────────────────────────────────────────────────────────────────
with tab_sim:
    st.markdown(
        f'<div class="sci-section-title">'
        f'SIMULATION RESULTS &nbsp;&mdash;&nbsp; {mat_id.upper()} / {geom_id} / {d_nm} nm / {int(T_K)} K'
        f'</div>',
        unsafe_allow_html=True)

    if btn_animate:
        # ── Modo animación: figura combinada clásica ──────────────────────────
        steps = list(range(lo, hi + 1, max(1, (hi - lo) // 25)))
        ph    = st.empty(); progress = st.progress(0)
        for i, s in enumerate(steps):
            f_a, _, _ = build_main_figure(mat_id, s, geom_id, MODELS,
                                           noise_level=noise_level, dpi=100,
                                           T=T_K, use_temp_correction=use_temp_correction)
            buf = io.BytesIO()
            f_a.savefig(buf, format='png', dpi=100,
                        bbox_inches='tight', facecolor='#0d1117')
            plt.close(f_a); buf.seek(0)
            ph.image(buf, use_column_width=True,
                     caption=f'{s} nm — {gm["name"]}')
            progress.progress((i + 1) / len(steps))
            time.sleep(0.10)
        progress.empty()

    else:
        # ── Dashboard 2×2 ─────────────────────────────────────────────────────
        _row1_l, _row1_r = st.columns(2, gap='small')
        _row2_l, _row2_r = st.columns(2, gap='small')

        _card_cfg = {'displayModeBar': True,
                     'modeBarButtonsToRemove': ['select2d', 'lasso2d',
                                                'autoScale2d', 'resetScale2d'],
                     'displaylogo': False}

        # ── Card 1: Hysteresis ───────────────────────────────────────────────
        with _row1_l:
            st.markdown(f"""
<div class="sim-card-header">
  <span class="sim-card-title">Hysteresis Loop</span>
  <span class="sim-card-tag">Hc = {Hc_val:.1f} mT &nbsp;&middot;&nbsp; Mr = {Mr_val:.3f}</span>
</div>""", unsafe_allow_html=True)
            _fig_hyst = build_hysteresis_card(
                mat_id, d_nm, geom_id,
                Hc_val, Mr_val, _sHc, _sMr,
                T=T_K, noise_level=noise_level)
            st.plotly_chart(_fig_hyst, use_container_width=True,
                            config=_card_cfg)

        # ── Card 2: Energy vs Field ──────────────────────────────────────────
        with _row1_r:
            st.markdown(f"""
<div class="sim-card-header">
  <span class="sim-card-title">Energy vs Field</span>
  <span class="sim-card-tag">Zeeman · Exchange · Demag · Anisotropy</span>
</div>""", unsafe_allow_html=True)
            _fig_enrg = build_energy_card(Hc_val, mat_id)
            st.plotly_chart(_fig_enrg, use_container_width=True,
                            config=_card_cfg)

        # ── Card 3: Hc vs Diameter sweep ────────────────────────────────────
        with _row2_l:
            st.markdown(f"""
<div class="sim-card-header">
  <span class="sim-card-title">Hc vs Diameter</span>
  <span class="sim-card-tag">{mat['name'].split('(')[0].strip()} &nbsp;&middot;&nbsp; {gm['name']}</span>
</div>""", unsafe_allow_html=True)
            _sweep_key = f'_fig_sweep_{mat_id}_{geom_id}_{int(T_K)}_{use_temp_correction}'
            if _sweep_key not in st.session_state:
                st.session_state[_sweep_key] = build_magnetization_sweep_card(
                    mat_id, geom_id, d_nm, MODELS, T_K,
                    use_temp_correction=use_temp_correction)
            else:
                # Reconstruir si cambió el punto actual (d_nm)
                _prev = st.session_state.get(f'{_sweep_key}_d')
                if _prev != d_nm:
                    st.session_state[_sweep_key] = build_magnetization_sweep_card(
                        mat_id, geom_id, d_nm, MODELS, T_K,
                        use_temp_correction=use_temp_correction)
            st.session_state[f'{_sweep_key}_d'] = d_nm
            st.plotly_chart(st.session_state[_sweep_key],
                            use_container_width=True, config=_card_cfg)

        # ── Card 4: 3D Geometry (Plotly) ─────────────────────────────────────
        with _row2_r:
            st.markdown(f"""
<div class="sim-card-header">
  <span class="sim-card-title">3D Geometry</span>
  <span class="sim-card-tag">{gm['name']} &nbsp;&middot;&nbsp; {d_nm} nm</span>
</div>""", unsafe_allow_html=True)
            _vox_key = f'_fig_vox_dash_{geom_id}_{d_nm}'
            if _vox_key not in st.session_state:
                with st.spinner('Voxelizing…'):
                    try:
                        _vf = _viz3d.voxel_geometry_3d(
                            geom_id, d_nm, GEOMETRY_MODES, n=16)
                        if _vf is not None:
                            _vf.update_layout(height=340,
                                              margin=dict(l=0, r=0, t=30, b=0))
                        st.session_state[_vox_key] = _vf
                    except Exception:
                        st.session_state[_vox_key] = None
            _fig_vox_d = st.session_state[_vox_key]
            if _fig_vox_d is not None:
                st.plotly_chart(_fig_vox_d, use_container_width=True,
                                config={'displayModeBar': False})
            else:
                st.markdown(
                    '<div class="sci-warn" style="margin:24px 0">3D engine unavailable</div>',
                    unsafe_allow_html=True)

    # ── Stats strip ───────────────────────────────────────────────────────────
    st.markdown(f"""
<div class="sci-metric-row" style="margin-top:10px">
  <div class="sci-metric-cell">
    <div class="sci-metric-label">Hc (mT)</div>
    <div class="sci-metric-value" style="color:var(--accent)">{Hc_val:.1f}</div>
    <div class="sci-metric-sub">&plusmn; {_sHc:.1f} &sigma;</div>
  </div>
  <div class="sci-metric-cell">
    <div class="sci-metric-label">Mr / Ms</div>
    <div class="sci-metric-value" style="color:var(--ok)">{Mr_val:.3f}</div>
    <div class="sci-metric-sub">&plusmn; {_sMr:.3f} &sigma;</div>
  </div>
  <div class="sci-metric-cell">
    <div class="sci-metric-label">Field max</div>
    <div class="sci-metric-value">{mat['field_max']}</div>
    <div class="sci-metric-sub">mT</div>
  </div>
  <div class="sci-metric-cell">
    <div class="sci-metric-label">Shape factor</div>
    <div class="sci-metric-value">&times;{gm['factor_hc']}</div>
    <div class="sci-metric-sub">f_Hc</div>
  </div>
  <div class="sci-metric-cell">
    <div class="sci-metric-label">Compute</div>
    <div class="sci-metric-value">{elapsed_ms:.0f}</div>
    <div class="sci-metric-sub">ms</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # ── Full analysis table ───────────────────────────────────────────────────
    with st.expander('📊  FULL SIMULATION ANALYSIS', expanded=False):
        import pandas as _pd_tab

        # ── Derived physics quantities ────────────────────────────────────────
        _p      = mat['params']
        _Ms_Am  = _p['Ms_MA_m'] * 1e6          # A/m
        _K1_Jm3 = abs(_p['K1_kJ_m3']) * 1e3    # J/m³
        _A_Jm   = _p['A_pJ_m'] * 1e-12         # J/m
        _Tc     = _p['Tc_K']
        _alpha  = _p['alpha']
        _lam_nm = _p['lambda_ex_nm']            # nm

        _r_m    = (d_nm * 1e-9) / 2.0
        _V_m3   = (4.0 / 3.0) * np.pi * _r_m ** 3
        _V_nm3  = _V_m3 * 1e27

        # Exchange length λ_ex = sqrt(2A / μ₀Ms²)
        _mu0    = 4 * np.pi * 1e-7
        _lam_ex = np.sqrt(2 * _A_Jm / (_mu0 * _Ms_Am ** 2)) * 1e9  # nm

        # Magnetocrystalline anisotropy field H_K (mT)
        _HK_mT  = (2 * _K1_Jm3 / (_mu0 * _Ms_Am)) * 1e3 if _Ms_Am > 0 else 0

        # Demagnetizing field correction  ΔH_dem (mT)  via shape factor Nd_z
        _Nd_z   = float(gm.get('Nd_z', 1.0 / 3.0))
        _Hdem_mT = _Nd_z * _Ms_Am * _mu0 * 1e3       # mT

        # Anisotropy energy barrier  E_b = K1 · V
        _Eb_J   = _K1_Jm3 * _V_m3
        _kB     = 1.380649e-23
        _T_safe = max(float(T_K), 1.0)
        _barrier = _Eb_J / (_kB * _T_safe)

        # Reduced magnetization at T (Callen-Callen)
        _Tc_safe = max(float(_Tc), 1.0)
        _ms_T = max(1.0 - (min(_T_safe / _Tc_safe, 0.9999)) ** 1.5, 0.0) ** (1.0 / 3.0)

        # Stoner-Wohlfarth critical field  H_SW (mT)
        _delta_N = max(0.5 - _Nd_z, 0.0)   # ΔN = N_⊥ - N_∥ for easy axis
        _H_sw_mT = _HK_mT + _delta_N * _Ms_Am * _mu0 * 1e3

        # SPM status
        _is_spm = _barrier < 25.0

        # Per-model predictions (raw, no geom factor for display)
        _all_preds = MODELS.predict_all_models(d_nm, mat_id, T=float(MODELS.T_sim))
        _geom_factor_hc = float(gm['factor_hc'])
        _geom_factor_mr = float(gm['factor_mr'])

        # ── Build table sections ──────────────────────────────────────────────
        def _section(title):
            return {'Parameter': f'── {title} ──', 'Value': '', 'Unit': '', 'Notes': ''}

        _rows = [
            # ── Identity ─────────────────────────────────────────────────────
            _section('MATERIAL & GEOMETRY'),
            {'Parameter': 'Material',         'Value': mat['name'],         'Unit': '—',    'Notes': mat.get('category','')},
            {'Parameter': 'Formula',           'Value': mat.get('formula',''), 'Unit': '—', 'Notes': ''},
            {'Parameter': 'Geometry',          'Value': gm['name'],          'Unit': '—',   'Notes': gm.get('description','')},
            {'Parameter': 'Diameter',          'Value': f'{d_nm:.1f}',       'Unit': 'nm',  'Notes': f'Range {lo}–{hi} nm'},
            {'Parameter': 'Volume',            'Value': f'{_V_nm3:.2f}',     'Unit': 'nm³', 'Notes': f'Sphere: V = (4/3)π(d/2)³'},

            # ── Physical parameters ───────────────────────────────────────────
            _section('PHYSICAL PARAMETERS'),
            {'Parameter': 'Ms',                'Value': f'{_p["Ms_MA_m"]:.3f}','Unit': 'MA/m','Notes': 'Saturation magnetization'},
            {'Parameter': 'K₁',                'Value': f'{_p["K1_kJ_m3"]:.1f}','Unit': 'kJ/m³','Notes': 'Anisotropy constant'},
            {'Parameter': 'A',                 'Value': f'{_p["A_pJ_m"]:.1f}',  'Unit': 'pJ/m', 'Notes': 'Exchange stiffness'},
            {'Parameter': 'α (LLG)',           'Value': f'{_alpha:.4f}',        'Unit': '—',    'Notes': 'Gilbert damping'},
            {'Parameter': 'λ_ex (material)',   'Value': f'{_lam_nm:.2f}',       'Unit': 'nm',   'Notes': 'Exchange length (stored)'},
            {'Parameter': 'λ_ex (computed)',   'Value': f'{_lam_ex:.2f}',       'Unit': 'nm',   'Notes': '√(2A / μ₀Ms²)'},
            {'Parameter': 'Tc',                'Value': f'{_Tc:.0f}',           'Unit': 'K',    'Notes': 'Curie temperature'},

            # ── Simulation conditions ─────────────────────────────────────────
            _section('SIMULATION CONDITIONS'),
            {'Parameter': 'Temperature',       'Value': f'{T_K:.1f}',          'Unit': 'K',    'Notes': ''},
            {'Parameter': 'T / Tc',            'Value': f'{_T_safe/_Tc_safe:.4f}','Unit':'—',  'Notes': 'Reduced temperature'},
            {'Parameter': 'ms(T) / ms(0)',     'Value': f'{_ms_T:.4f}',         'Unit': '—',   'Notes': 'Callen-Callen: (1–(T/Tc)^1.5)^(1/3)'},
            {'Parameter': 'Temp. correction',  'Value': 'ON' if use_temp_correction else 'OFF','Unit':'—','Notes': ''},
            {'Parameter': 'Noise level',       'Value': f'{noise_level:.0%}',   'Unit': '—',   'Notes': 'Synthesis noise added to training'},
            {'Parameter': 'Field max',         'Value': f'{mat["field_max"]}',  'Unit': 'mT',  'Notes': 'Sweep range'},
            {'Parameter': 'Extrapolation',     'Value': '⚠ Yes' if is_extrapolation(d_nm, mat_id) else 'No','Unit':'—','Notes': 'Outside training range'},
            {'Parameter': 'Compute time',      'Value': f'{elapsed_ms:.1f}',    'Unit': 'ms',  'Notes': 'build_main_figure()'},

            # ── Geometry (Nd) ─────────────────────────────────────────────────
            _section('GEOMETRY — DEMAGNETIZATION'),
            {'Parameter': 'Nd_z (easy axis)',  'Value': f'{_Nd_z:.4f}',         'Unit': '—',   'Notes': 'Osborn/Chen/Aharoni analytical'},
            {'Parameter': 'f_Hc',             'Value': f'{gm["factor_hc"]:.3f}','Unit': '—',   'Notes': 'Shape factor for Hc'},
            {'Parameter': 'f_Mr',             'Value': f'{gm["factor_mr"]:.3f}','Unit': '—',   'Notes': 'Shape factor for Mr'},
            {'Parameter': 'H_dem (Nd·μ₀Ms)',  'Value': f'{_Hdem_mT:.1f}',       'Unit': 'mT',  'Notes': 'Demagnetizing field estimate'},

            # ── Physics-derived ───────────────────────────────────────────────
            _section('DERIVED PHYSICS'),
            {'Parameter': 'H_K (anisotropy)', 'Value': f'{_HK_mT:.1f}',         'Unit': 'mT',  'Notes': '2K₁/(μ₀Ms)'},
            {'Parameter': 'H_SW (Stoner-W.)', 'Value': f'{_H_sw_mT:.1f}',       'Unit': 'mT',  'Notes': 'H_K + ΔN·μ₀Ms (single-domain)'},
            {'Parameter': 'E_b (barrier)',    'Value': f'{_Eb_J:.3e}',           'Unit': 'J',   'Notes': 'K₁·V'},
            {'Parameter': 'E_b / k_BT',       'Value': f'{_barrier:.2f}',        'Unit': '—',   'Notes': 'Thermal stability ratio'},
            {'Parameter': 'SPM regime',       'Value': '⚠ Yes (E_b/k_BT < 25)' if _is_spm else 'No','Unit':'—','Notes': 'Néel criterion'},

            # ── Per-model ML predictions ──────────────────────────────────────
            _section('ML PREDICTIONS — RAW (sphere reference)'),
            {'Parameter': 'GBR  Hc',          'Value': f'{_all_preds["GBR"]["Hc"]:.2f}', 'Unit':'mT','Notes': 'Gradient Boosting'},
            {'Parameter': 'GBR  Mr',          'Value': f'{_all_preds["GBR"]["Mr"]:.4f}', 'Unit':'—', 'Notes': ''},
            {'Parameter': 'RF   Hc',          'Value': f'{_all_preds["RF"]["Hc"]:.2f}',  'Unit':'mT','Notes': 'Random Forest'},
            {'Parameter': 'RF   Mr',          'Value': f'{_all_preds["RF"]["Mr"]:.4f}',  'Unit':'—', 'Notes': ''},
            {'Parameter': 'MLP  Hc',          'Value': f'{_all_preds["MLP"]["Hc"]:.2f}', 'Unit':'mT','Notes': 'Neural Network'},
            {'Parameter': 'MLP  Mr',          'Value': f'{_all_preds["MLP"]["Mr"]:.4f}', 'Unit':'—', 'Notes': ''},
            {'Parameter': 'Ensemble Hc (raw)','Value': f'{_all_preds["Ensemble"]["Hc"]:.2f}','Unit':'mT','Notes': 'Weighted avg by R² CV'},
            {'Parameter': 'Ensemble Mr (raw)','Value': f'{_all_preds["Ensemble"]["Mr"]:.4f}','Unit':'—', 'Notes': ''},

            # ── Final result ──────────────────────────────────────────────────
            _section('FINAL RESULT (geometry + temperature corrected)'),
            {'Parameter': 'Hc',               'Value': f'{Hc_val:.2f}',         'Unit': 'mT',  'Notes': f'±{_sHc:.2f} mT  (1σ RF variance)'},
            {'Parameter': 'Mr / Ms',          'Value': f'{Mr_val:.4f}',          'Unit': '—',   'Notes': f'±{_sMr:.4f}  (1σ)'},
            {'Parameter': 'Hc · f_Hc',        'Value': f'{Hc_val / max(_geom_factor_hc,1e-6):.2f}','Unit':'mT','Notes': 'Back-projected sphere value'},
            {'Parameter': 'μ₀Ms (saturation)',  'Value': f'{_Ms_Am*_mu0*1e3:.1f}','Unit': 'mT',  'Notes': 'μ₀Ms'},
            {'Parameter': 'Hc / μ₀Ms',        'Value': f'{Hc_val / max(_Ms_Am*_mu0*1e3,1e-6):.4f}','Unit':'—','Notes': 'Reduced coercivity'},
        ]

        _df_analysis = _pd_tab.DataFrame(_rows)
        # Style: section separators in a muted colour
        def _style_sections(row):
            if str(row['Parameter']).startswith('──'):
                return ['background-color:#1e293b; color:#94a3b8; font-weight:600'] * len(row)
            return [''] * len(row)

        st.dataframe(
            _df_analysis.style.apply(_style_sections, axis=1),
            use_container_width=True,
            hide_index=True,
            height=min(40 + len(_rows) * 35, 780),
        )

        # Download button — CSV
        _csv_bytes = _df_analysis.to_csv(index=False).encode()
        _csv_fname = (f'analysis_{mat_id}_{geom_id}_{d_nm:.0f}nm_{T_K:.0f}K.csv')
        st.download_button(
            label='DOWNLOAD ANALYSIS (.csv)',
            data=_csv_bytes,
            file_name=_csv_fname,
            mime='text/csv',
            use_container_width=True,
        )

    # ── Exportar a OriginLab ───────────────────────────────────────────────────
    with st.expander('EXPORT — OriginLab (.txt)', expanded=False):
        st.markdown(
            'Tab-delimited format compatible with **OriginLab 8+**.\n\n'
            'Includes: hysteresis (both branches) · energy landscape · '
            'geometry comparison table with uncertainty ±σ.\n\n'
            '**Import in Origin:** `File → Import → ASCII` and enable '
            '*Read Long Name / Units / Comments from rows*.'
        )
        _T_cur = float(st.session_state.get('T_sim', 300))
        _origin_bytes = export_to_originlab(
            mat_id, d_nm, geom_id, Hc_val, Mr_val, _T_cur, MODELS)
        _origin_fname = (f'SimuMag_{mat["name"].replace(" ","_")}_'
                         f'{d_nm:.0f}nm_{geom_id}_{_T_cur:.0f}K.txt')
        st.download_button(
            label=f'DOWNLOAD {_origin_fname}',
            data=_origin_bytes,
            file_name=_origin_fname,
            mime='text/plain',
            use_container_width=True,
        )
        st.caption(
            'In OriginLab: Plot → Line / Scatter to generate publication-quality figures.'
        )

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 2 — COMPARAR  (material + geometría)
# ─────────────────────────────────────────────────────────────────────────────
with tab_compare:
    st.markdown('<div class="sci-section-title">COMPARE — Material &times; Geometry</div>',
                unsafe_allow_html=True)

    cc1, cc2 = st.columns(2)
    all_mats  = list(MATERIALS_DB.keys())
    all_geoms = list(GEOMETRY_MODES.keys())

    sel_a_mat  = cc1.selectbox('Material A', all_mats, index=0,
                                format_func=lambda x: f"{x.upper()} · {MATERIALS_DB[x]['name']}",
                                key='c_amat')
    sel_a_geom = cc1.selectbox('Geometry A', all_geoms, index=0,
                                format_func=lambda x: GEOMETRY_MODES[x]['name'],
                                key='c_ageom')
    sel_b_mat  = cc2.selectbox('Material B', all_mats, index=1,
                                format_func=lambda x: f"{x.upper()} · {MATERIALS_DB[x]['name']}",
                                key='c_bmat')
    sel_b_geom = cc2.selectbox('Geometry B', all_geoms, index=1,
                                format_func=lambda x: GEOMETRY_MODES[x]['name'],
                                key='c_bgeom')

    c_size = st.slider('Diameter for comparison (nm)', 5, 150, 30, key='c_sz')

    # ── Calcular Hc/Mr + incertidumbre para A y B ─────────────────────────────
    _cmp_pairs = []
    for _sm, _sg in [(sel_a_mat, sel_a_geom), (sel_b_mat, sel_b_geom)]:
        _m    = MATERIALS_DB[_sm]
        _gd   = GEOMETRY_MODES[_sg]
        _lo, _hi = _m['range']
        _dc   = float(min(max(c_size, _lo), _hi))
        _Hcc, _Mrc, _sHcc, _sMrc, _ = predict_geom_with_uncertainty_temp(
            _dc, _sm, _sg, MODELS,
            T_K=T_K, use_temp_correction=use_temp_correction)
        _cmp_pairs.append((_sm, _sg, _m, _gd, _dc, _Hcc, _Mrc, _sHcc, _sMrc))

    (sm_a, sg_a, m_a, gd_a, dc_a, Hc_a, Mr_a, sHc_a, sMr_a) = _cmp_pairs[0]
    (sm_b, sg_b, m_b, gd_b, dc_b, Hc_b, Mr_b, sHc_b, sMr_b) = _cmp_pairs[1]

    # ── Temperatura activa ────────────────────────────────────────────────────
    _tc_note = (f'Thermal correction ON — {int(T_K)} K'
                if use_temp_correction else f'Thermal correction OFF — raw ML')
    st.markdown(
        f'<div class="sci-info" style="margin-bottom:10px;font-size:10px;">'
        f'T = {int(T_K)} K &nbsp;&nbsp;|&nbsp;&nbsp; {_tc_note}'
        f'</div>',
        unsafe_allow_html=True)

    # ── Cards A / B (Plotly, igual altura) ───────────────────────────────────
    _cfg_cmp = {'displayModeBar': True,
                'modeBarButtonsToRemove': ['select2d','lasso2d','autoScale2d'],
                'displaylogo': False}

    _col_a, _col_b = st.columns(2, gap='small')

    def _make_hyst_cmp(mat_id, geom_id, d_c, Hc, Mr, sHc, sMr, color):
        """Plotly hysteresis card para el tab Compare."""
        m     = MATERIALS_DB[mat_id]
        H_max = m['field_max']
        H, M_up, M_dn = llg_hysteresis(Hc, Mr, H_max=H_max, seed=42)
        _, M_hi, _    = llg_hysteresis(Hc + sHc, min(Mr + sMr, 1.0),
                                        H_max=H_max, noise_level=0, seed=42)
        _, M_lo, _    = llg_hysteresis(max(Hc - sHc, 0.5), max(Mr - sMr, 0.05),
                                        H_max=H_max, noise_level=0, seed=42)
        fig = go.Figure()
        # Banda ±1σ
        fig.add_trace(go.Scatter(
            x=np.concatenate([H, H[::-1]]),
            y=np.concatenate([M_hi, M_lo[::-1]]),
            fill='toself',
            fillcolor=f'rgba({int(color[1:3],16)},{int(color[3:5],16)},{int(color[5:7],16)},0.13)',
            line=dict(width=0), name='±1σ', hoverinfo='skip'))
        # Ramas
        fig.add_trace(go.Scatter(
            x=H, y=M_up, mode='lines',
            line=dict(color=color, width=2.2),
            name=f'M↑  Hc={Hc:.1f} mT',
            hovertemplate='H = %{x:.1f} mT<br>M/Ms = %{y:.3f}<extra>↑</extra>'))
        fig.add_trace(go.Scatter(
            x=H, y=M_dn, mode='lines',
            line=dict(color=color, width=1.7, dash='dash'),
            name=f'M↓  Mr={Mr:.3f}',
            hovertemplate='H = %{x:.1f} mT<br>M/Ms = %{y:.3f}<extra>↓</extra>'))
        # Marcadores Hc / Mr
        fig.add_trace(go.Scatter(
            x=[Hc, -Hc], y=[0, 0], mode='markers',
            marker=dict(color='#ffb74d', size=9, symbol='line-ns-open', line_width=2),
            name=f'Hc=±{Hc:.1f}',
            hovertemplate='Hc = %{x:.1f} mT<extra>Coercivity</extra>'))
        fig.add_trace(go.Scatter(
            x=[0, 0], y=[Mr, -Mr], mode='markers',
            marker=dict(color='#81c784', size=9, symbol='line-ew-open', line_width=2),
            name=f'Mr={Mr:.3f}',
            hovertemplate='Mr/Ms = %{y:.3f}<extra>Remanence</extra>'))
        fig.add_hline(y=0, line_color='#334155', line_width=0.8)
        fig.add_vline(x=0, line_color='#334155', line_width=0.8)
        _apply_plotly_theme(fig, xaxis_title='H (mT)', yaxis_title='M / Ms')
        fig.update_layout(
            height=340,
            margin=dict(l=52, r=12, t=16, b=44),
            legend=dict(orientation='h', yanchor='bottom', y=1.01,
                        xanchor='left', x=0, font_size=9),
            hovermode='x unified')
        return fig

    with _col_a:
        st.markdown(f"""
<div class="sim-card-header">
  <span class="sim-card-title">{m_a['name'].split('(')[0].strip()} · {gd_a['name']}</span>
  <span class="sim-card-tag">{dc_a:.0f} nm &nbsp;&middot;&nbsp; A</span>
</div>""", unsafe_allow_html=True)
        st.plotly_chart(
            _make_hyst_cmp(sm_a, sg_a, dc_a, Hc_a, Mr_a, sHc_a, sMr_a, m_a['color']),
            use_container_width=True, config=_cfg_cmp)

    with _col_b:
        st.markdown(f"""
<div class="sim-card-header">
  <span class="sim-card-title">{m_b['name'].split('(')[0].strip()} · {gd_b['name']}</span>
  <span class="sim-card-tag">{dc_b:.0f} nm &nbsp;&middot;&nbsp; B</span>
</div>""", unsafe_allow_html=True)
        st.plotly_chart(
            _make_hyst_cmp(sm_b, sg_b, dc_b, Hc_b, Mr_b, sHc_b, sMr_b, m_b['color']),
            use_container_width=True, config=_cfg_cmp)

    # ── Strip de métricas A vs B ──────────────────────────────────────────────
    _dHc  = Hc_a - Hc_b
    _dMr  = Mr_a - Mr_b
    _dHc_col  = 'var(--ok)' if abs(_dHc)  < 5  else 'var(--warn)'
    _dMr_col  = 'var(--ok)' if abs(_dMr)  < 0.05 else 'var(--warn)'
    st.markdown(f"""
<div class="sci-metric-row" style="margin-top:8px">
  <div class="sci-metric-cell">
    <div class="sci-metric-label">Hc — A (mT)</div>
    <div class="sci-metric-value" style="color:{m_a['color']}">{Hc_a:.1f}</div>
    <div class="sci-metric-sub">&plusmn; {sHc_a:.1f} &sigma;</div>
  </div>
  <div class="sci-metric-cell">
    <div class="sci-metric-label">Hc — B (mT)</div>
    <div class="sci-metric-value" style="color:{m_b['color']}">{Hc_b:.1f}</div>
    <div class="sci-metric-sub">&plusmn; {sHc_b:.1f} &sigma;</div>
  </div>
  <div class="sci-metric-cell">
    <div class="sci-metric-label">&Delta;Hc  A&minus;B</div>
    <div class="sci-metric-value" style="color:{_dHc_col}">{_dHc:+.1f}</div>
    <div class="sci-metric-sub">mT</div>
  </div>
  <div class="sci-metric-cell">
    <div class="sci-metric-label">Mr — A</div>
    <div class="sci-metric-value" style="color:{m_a['color']}">{Mr_a:.3f}</div>
    <div class="sci-metric-sub">&plusmn; {sMr_a:.3f} &sigma;</div>
  </div>
  <div class="sci-metric-cell">
    <div class="sci-metric-label">Mr — B</div>
    <div class="sci-metric-value" style="color:{m_b['color']}">{Mr_b:.3f}</div>
    <div class="sci-metric-sub">&plusmn; {sMr_b:.3f} &sigma;</div>
  </div>
  <div class="sci-metric-cell">
    <div class="sci-metric-label">&Delta;Mr  A&minus;B</div>
    <div class="sci-metric-value" style="color:{_dMr_col}">{_dMr:+.3f}</div>
    <div class="sci-metric-sub">Mr/Ms</div>
  </div>
</div>""", unsafe_allow_html=True)

    # ── Overlay normalizado (Plotly) ──────────────────────────────────────────
    st.markdown(
        '<div class="sci-section-title" style="margin-top:18px">'
        'NORMALIZED OVERLAY — H / H<sub>max</sub></div>',
        unsafe_allow_html=True)

    fig_ov = go.Figure()
    for _sm, _sg, _m, _gd, _dc, _Hcc, _Mrc, _sHcc, _sMrc in _cmp_pairs:
        _H, _Mu, _Md = llg_hysteresis(_Hcc, _Mrc, H_max=_m['field_max'], seed=42)
        _Hn = _H / _m['field_max']
        _lbl = f"{_m['name'].split('(')[0].strip()} · {_gd['name']}  {_dc:.0f} nm"
        fig_ov.add_trace(go.Scatter(
            x=_Hn, y=_Mu, mode='lines',
            line=dict(color=_m['color'], width=2.2),
            name=f'{_lbl}  ↑',
            hovertemplate='H/Hmax = %{x:.3f}<br>M/Ms = %{y:.3f}<extra>' + _lbl + ' ↑</extra>'))
        fig_ov.add_trace(go.Scatter(
            x=_Hn, y=_Md, mode='lines',
            line=dict(color=_m['color'], width=1.6, dash='dash'),
            name=f'{_lbl}  ↓',
            showlegend=False,
            hovertemplate='H/Hmax = %{x:.3f}<br>M/Ms = %{y:.3f}<extra>' + _lbl + ' ↓</extra>'))
    fig_ov.add_hline(y=0, line_color='#334155', line_width=0.8)
    _apply_plotly_theme(fig_ov,
                        xaxis_title='H / H<sub>max</sub>  (normalized)',
                        yaxis_title='M / Ms')
    fig_ov.update_layout(
        height=320,
        margin=dict(l=52, r=16, t=16, b=44),
        legend=dict(orientation='h', yanchor='bottom', y=1.01,
                    xanchor='left', x=0, font_size=9),
        hovermode='x unified')
    st.plotly_chart(fig_ov, use_container_width=True, config=_cfg_cmp)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 3 — PARÁMETROS ML FASE 4  (ensemble · features · online learning)
# ─────────────────────────────────────────────────────────────────────────────
with tab_params:
    st.markdown(f'<div class="sci-section-title">ML Parameters — {mat["name"]}</div>',
                unsafe_allow_html=True)

    # ── Subtabs del panel ML ──────────────────────────────────────────────────
    ml_tab1, ml_tab2, ml_tab3, ml_tab4 = st.tabs([
        'Ensemble Metrics',
        'Feature Importance',
        'Uncertainty Curves',
        'Online Learning',
    ])

    eng_metrics = MODELS.get_metrics(mat_id)

    # ── SUB-TAB 1: Métricas ───────────────────────────────────────────────────
    with ml_tab1:
        st.markdown('#### Model Comparison — R² (CV 5-fold) and RMSE')

        p = mat['params']
        col_p1, col_p2, col_p3, col_p4 = st.columns(4)
        col_p1.metric('K₁ (kJ/m³)', p['K1_kJ_m3'])
        col_p1.metric('A (pJ/m)',    p['A_pJ_m'])
        col_p2.metric('Ms (MA/m)',   p['Ms_MA_m'])
        col_p2.metric('α (LLG)',     p['alpha'])
        col_p3.metric('λₑₓ (nm)',    p['lambda_ex_nm'])
        col_p3.metric('Tc (K)',      p['Tc_K'])
        col_p4.metric('Training pts.', eng_metrics.get('n_train', '—'))
        col_p4.metric('Feedback',      eng_metrics.get('n_feedback', 0))
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
                'Model':        name,
                'R² CV  Hc':    round(r2_hc.get(name, 0), 4),
                'R² CV  Mr':    round(r2_mr.get(name, 0), 4),
                'RMSE Hc (mT)': round(rmse_hc.get(name, 0), 2),
                'RMSE Mr':      round(rmse_mr.get(name, 0), 5),
                'Weight Hc':    f'{w_hc[i]:.3f}',
                'Weight Mr':    f'{w_mr[i]:.3f}',
            })
        st.dataframe(pd.DataFrame(rows_metrics),
                     use_container_width=True, hide_index=True)

        # R² bar chart — Plotly (interactive)
        _r2_model_colors = ['#38bdf8', '#34d399', '#f472b6']
        _r2_col1, _r2_col2 = st.columns(2)
        for _r2_col, _r2_dict, _r2_title in zip(
            [_r2_col1, _r2_col2],
            [r2_hc, r2_mr],
            ['R² CV — Hc (coercive field)', 'R² CV — Mr/Ms (remanence)'],
        ):
            _r2_names = list(_r2_dict.keys())
            _r2_vals  = [round(v, 4) for v in _r2_dict.values()]
            _fig_r2 = go.Figure(go.Bar(
                x=_r2_names, y=_r2_vals,
                marker_color=_r2_model_colors[:len(_r2_names)],
                text=[f'{v:.3f}' for v in _r2_vals],
                textposition='outside',
                hovertemplate='<b>%{x}</b><br>R² = %{y:.4f}<extra></extra>',
            ))
            _apply_plotly_theme(_fig_r2, title=_r2_title, height=300)
            _fig_r2.update_layout(
                yaxis=dict(range=[0, 1.1], title='R²'),
                showlegend=False,
                margin=dict(l=40, r=20, t=40, b=30),
            )
            _r2_col.plotly_chart(_fig_r2, use_container_width=True,
                                 config={'displayModeBar': False})

        st.divider()
        st.markdown('**Size sweep — Ensemble predictions by geometry**')
        sizes_sweep = sorted(set(
            list(range(lo, hi + 1, max(1, (hi - lo) // 10))) + [hi]))
        sizes_sweep_arr = np.array(sizes_sweep, dtype=float)
        # 1 batch call → luego escalar por factor de geometría (antes 8×n calls)
        Hc_sw_base, _ = MODELS.predict_batch(sizes_sweep_arr, mat_id)
        rows_sw = []
        for i, s in enumerate(sizes_sweep):
            row_s = {'Size (nm)': s,
                     'Extrapol.': '!' if is_extrapolation(s, mat_id) else 'ok'}
            for gid, gdata in GEOMETRY_MODES.items():
                Hc_g = float(Hc_sw_base[i]) * GEOMETRY_MODES[gid]['factor_hc']
                row_s[f'Hc {gdata["name"]}'] = round(Hc_g, 1)
            rows_sw.append(row_s)
        st.dataframe(pd.DataFrame(rows_sw),
                     use_container_width=True, hide_index=True)
        st.info(
            f'Selection: {d_nm} nm · {gm["name"]} · '
            f'Hc = {Hc_val:.1f} mT · Mr/Ms = {Mr_val:.3f}'
        )

    # ── SUB-TAB 2: Importancia de features ───────────────────────────────────
    with ml_tab2:
        st.markdown('#### Feature Importance — GradientBoostingRegressor')
        st.caption(
            'Fraction of total "information gain" attributed to each feature. '
            'Physically motivated features: the model learns which parameters '
            'dominate coercivity and remanence.'
        )
        fi = MODELS.feature_importance(mat_id)
        if fi:
            _fi_colors = ['#38bdf8','#fb923c','#34d399','#f472b6',
                          '#fbbf24','#a78bfa','#f87171']
            _fi_col1, _fi_col2 = st.columns(2)
            for _fi_col, _fi_key, _fi_title in zip(
                [_fi_col1, _fi_col2],
                ['hc', 'mr'],
                ['Importance → Hc (coercive field)',
                 'Importance → Mr/Ms (remanence)'],
            ):
                _fi_imps  = fi[_fi_key]
                _fi_names = fi['names']
                _fi_idx   = np.argsort(_fi_imps)          # ascending for horizontal bar
                _fi_sorted_names = [_fi_names[i] for i in _fi_idx]
                _fi_sorted_imps  = _fi_imps[_fi_idx]
                _fi_bar_colors   = [_fi_colors[i % len(_fi_colors)] for i in _fi_idx]
                _fig_fi = go.Figure(go.Bar(
                    x=_fi_sorted_imps,
                    y=_fi_sorted_names,
                    orientation='h',
                    marker_color=_fi_bar_colors,
                    text=[f'{v:.3f}' for v in _fi_sorted_imps],
                    textposition='outside',
                    hovertemplate='<b>%{y}</b><br>Importance = %{x:.4f}<extra></extra>',
                ))
                _apply_plotly_theme(_fig_fi, title=_fi_title, height=320)
                _fig_fi.update_layout(
                    xaxis=dict(title='Relative importance', range=[0, max(_fi_sorted_imps) * 1.25]),
                    showlegend=False,
                    margin=dict(l=100, r=40, t=40, b=30),
                )
                _fi_col.plotly_chart(_fig_fi, use_container_width=True,
                                     config={'displayModeBar': False})

            st.markdown('**Feature descriptions**')
            feat_desc = {
                'd (nm)':        'Particle diameter — base feature',
                'd / λₑₓ':       'Size normalized to exchange length (critical for SPM transition)',
                'log₁₀(d)':      'Log scale — captures power-law behavior',
                'K₁V / k_BT':    'Anisotropy energy barrier vs. thermal energy',
                'Ms (MA/m)':     'Saturation magnetization of the material',
                'α (LLG)':       'Gilbert damping parameter',
                'T / Tc':        'Reduced temperature (0 = 0 K, 1 = Curie point)',
            }
            for fname, fdesc in feat_desc.items():
                st.markdown(f'  - **{fname}**: {fdesc}')

    # ── SUB-TAB 3: Curvas con bandas de incertidumbre ─────────────────────────
    with ml_tab3:
        st.markdown('#### Hc vs Size — model comparison and ±1σ band')
        st.caption(
            'Lines = individual predictions (GBR, RF, MLP). '
            'Thick line = weighted Ensemble. '
            'Shaded band = ±1σ estimated from RandomForest variance.'
        )

        sweep_data = MODELS.predict_all_models_sweep(mat_id, n_pts=50)
        sizes_full = sweep_data['sizes']
        _unc_m_colors = {'GBR': '#38bdf8', 'RF': '#34d399', 'MLP': '#f472b6'}
        _unc_col1, _unc_col2 = st.columns(2)

        for _unc_col, _unc_target, _unc_ylabel, _unc_std_key in zip(
            [_unc_col1, _unc_col2],
            ['Hc', 'Mr'],
            ['Hc (mT)', 'Mr / Ms'],
            ['Hc_std', 'Mr_std'],
        ):
            _unc_ens = sweep_data['Ensemble'][_unc_target]
            _unc_std = sweep_data[_unc_std_key]
            _fig_unc = go.Figure()

            # ±1σ band
            _fig_unc.add_trace(go.Scatter(
                x=np.concatenate([sizes_full, sizes_full[::-1]]),
                y=np.concatenate([_unc_ens + _unc_std, (_unc_ens - _unc_std)[::-1]]),
                fill='toself', fillcolor='rgba(241,245,249,0.12)',
                line=dict(color='rgba(0,0,0,0)'), name='±1σ (RF)',
                hoverinfo='skip',
            ))
            # Individual models
            for _unc_nm, _unc_col_c in _unc_m_colors.items():
                _fig_unc.add_trace(go.Scatter(
                    x=sizes_full, y=sweep_data[_unc_nm][_unc_target],
                    name=_unc_nm, line=dict(color=_unc_col_c, width=1.5, dash='dash'),
                    hovertemplate=f'<b>{_unc_nm}</b><br>d = %{{x:.1f}} nm<br>{_unc_ylabel} = %{{y:.3f}}<extra></extra>',
                ))
            # Ensemble line
            _fig_unc.add_trace(go.Scatter(
                x=sizes_full, y=_unc_ens, name='Ensemble',
                line=dict(color='#f1f5f9', width=2.5),
                hovertemplate=f'<b>Ensemble</b><br>d = %{{x:.1f}} nm<br>{_unc_ylabel} = %{{y:.3f}}<extra></extra>',
            ))
            # Current point marker
            _unc_cur = float(_unc_ens[np.argmin(np.abs(sizes_full - d_nm))])
            _fig_unc.add_trace(go.Scatter(
                x=[d_nm], y=[_unc_cur], name=f'{d_nm:.0f} nm',
                mode='markers', marker=dict(color='#fbbf24', size=10, symbol='diamond'),
                hovertemplate=f'd = {d_nm:.1f} nm<br>{_unc_ylabel} = %{{y:.3f}}<extra></extra>',
            ))
            # Valid range shading
            _fig_unc.add_vrect(x0=lo, x1=hi, fillcolor='#38bdf8',
                               opacity=0.07, layer='below', line_width=0)
            _apply_plotly_theme(_fig_unc,
                                title=f'{_unc_ylabel} vs Size — {mat["name"]}',
                                height=360)
            _fig_unc.update_layout(
                xaxis_title='Size (nm)', yaxis_title=_unc_ylabel,
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02,
                            xanchor='right', x=1, font_size=9),
            )
            _unc_col.plotly_chart(_fig_unc, use_container_width=True,
                                  config={'displayModeBar': False})

        # Hc vs Size — all geometries
        st.divider()
        st.markdown('**Hc vs Size — all geometries (Ensemble)**')
        sizes_geom = np.linspace(max(2, lo - 5), hi + 5,
                                 min(25, max(10, (hi - lo) // 4)))
        Hc_base_geom, _ = MODELS.predict_batch(sizes_geom, mat_id)
        _geom_pal = ['#38bdf8','#fb923c','#34d399','#f472b6',
                     '#fbbf24','#a78bfa','#6ee7b7','#f87171']
        _fig_hc = go.Figure()
        for (gid, gdata), gcol in zip(GEOMETRY_MODES.items(), _geom_pal):
            _hc_g = Hc_base_geom * GEOMETRY_MODES[gid]['factor_hc']
            _fig_hc.add_trace(go.Scatter(
                x=sizes_geom, y=_hc_g, name=gdata['name'],
                line=dict(color=gcol, width=1.8),
                hovertemplate=f'<b>{gdata["name"]}</b><br>d = %{{x:.1f}} nm<br>Hc = %{{y:.1f}} mT<extra></extra>',
            ))
        _fig_hc.add_vline(x=d_nm, line_color='#f1f5f9', line_dash='dot',
                          line_width=1, opacity=0.7)
        _fig_hc.add_vrect(x0=lo, x1=hi, fillcolor='#38bdf8',
                          opacity=0.07, layer='below', line_width=0)
        _apply_plotly_theme(_fig_hc,
                            title=f'Hc vs Size — {mat["name"]} — all geometries',
                            height=380)
        _fig_hc.update_layout(
            xaxis_title='Size (nm)', yaxis_title='Hc (mT)',
            hovermode='x unified',
            legend=dict(orientation='h', yanchor='bottom', y=1.02,
                        xanchor='right', x=1, font_size=9),
        )
        st.plotly_chart(_fig_hc, use_container_width=True,
                        config={'displayModeBar': False})

    # ── SUB-TAB 4: Online Learning ────────────────────────────────────────────
    with ml_tab4:
        st.markdown('#### Online Learning — Simulation Feedback')
        st.markdown(
            'Each simulation adds a feedback point to the ML engine. '
            'When retraining, these points receive **×20 priority** over '
            'synthetic data, allowing the model to specialize in '
            'the ranges you use most.'
        )

        fb_counts = MODELS.feedback_counts
        total_fb  = MODELS.total_feedback

        fcol1, fcol2, fcol3 = st.columns(3)
        fcol1.metric('Total feedback pts.',   total_fb)
        fcol2.metric(f'Feedback {mat["name"]}', fb_counts.get(mat_id, 0))
        fcol3.metric('Materials with feedback',
                     sum(1 for v in fb_counts.values() if v > 0))

        if total_fb > 0:
            st.divider()
            # Tabla de feedback por material
            fb_rows = [{'Material': MATERIALS_DB[mid]['name'],
                        'ID': mid.upper(),
                        'Feedback pts.': cnt}
                       for mid, cnt in fb_counts.items() if cnt > 0]
            st.dataframe(pd.DataFrame(fb_rows),
                         use_container_width=True, hide_index=True)

        st.divider()
        btn_retrain = st.button(
            'RETRAIN WITH FEEDBACK',
            use_container_width=True, type='primary',
            disabled=(total_fb == 0),
            help='Incorporates all feedback points into training. '
                 'Previous simulation data improves future predictions.',
        )
        if btn_retrain:
            with st.spinner('Retraining ensemble GBR + RF + MLP…'):
                MODELS.retrain_with_feedback()
            st.success(
                f'Models retrained with {total_fb} feedback points. '
                f'Predictions updated.'
            )
            st.rerun()

        st.divider()
        st.markdown('**How does online learning work?**')
        st.markdown('''
1. Each time you press **▶ SIMULATE**, the result (Hc, Mr) is recorded as a feedback point.
2. Feedback includes the **7 physical features** of the simulated point.
3. When you press **Retrain**, the engine rebuilds the dataset with:
   - Base synthetic data (interpolated from literature)
   - Feedback points with **×20 weight** (high priority)
4. All three models (GBR, RF, MLP) are retrained and ensemble weights are recalculated.
5. Predictions improve in the size ranges and materials you use most.
        ''')
        st.info(
            'Tip: Simulate the same material at several sizes to help the model '
            'learn the real shape of the Hc(d) curve for that material.'
        )

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 4 — 3D INTERACTIVO
# ─────────────────────────────────────────────────────────────────────────────
with tab_3d:
    st.markdown('<div class="sci-section-title">3D Interactive Visualizations</div>',
                unsafe_allow_html=True)

    # ── Contexto de simulación activa ─────────────────────────────────────────
    st.markdown(
        f'<div class="sci-info" style="margin-bottom:10px;font-size:10px;">'
        f'{mat["name"].split("(")[0].strip()} &nbsp;|&nbsp; {gm["name"]}'
        f' &nbsp;|&nbsp; {d_nm} nm &nbsp;|&nbsp; {int(T_K)} K'
        f' &nbsp;&nbsp;&mdash;&nbsp;&nbsp; Plotly: zoom, rotate &amp; hover with mouse'
        f'</div>',
        unsafe_allow_html=True)

    # ── Metadata por visualización: descripción + tiempo estimado ─────────────
    _VIZ_META = {
        'Voxel 3D Geometry':         ('Nanoparticle voxelized in 3D, colored by mz remanent state.',          'fast',   '#238636'),
        'Magnetization 2D Map':      ('Four magnetic states on XY cross-section with in-plane arrows.',        'fast',   '#238636'),
        'Energy Components':         ('Zeeman · Exchange · Demag · Anisotropy vs H for multiple sizes.',       'medium', '#9e6a03'),
        'Energy Surface  E(H, d)':   ('3D surface E(H, d) — field on X, diameter on Y, energy on Z.',         'medium', '#9e6a03'),
        'Magnetization Vectors':     ('Cone glyphs on sphere surface showing M direction and magnitude.',      'fast',   '#238636'),
        'Heatmap  Hc(material, nm)': ('Predicted Hc (mT) across all materials × all diameters.',              'slow',   '#b62324'),
        'Polar Anisotropy  E_K(θ)':  ('E_K = K₁·sin²(θ) polar chart — larger area = stronger anisotropy.',   'instant','#1f6feb'),
        'Hysteresis 3D Stack':       ('Stacked hysteresis loops at different diameters, color-coded by size.','medium', '#9e6a03'),
    }
    _TIME_LABEL = {'instant': 'instant', 'fast': '~1 s', 'medium': '~3 s', 'slow': '~8 s'}

    # ── Selector compacto ─────────────────────────────────────────────────────
    _viz_options = list(_VIZ_META.keys())
    _sel_col, _badge_col = st.columns([3, 1], gap='small')
    with _sel_col:
        viz_sel = st.selectbox('Visualization type', _viz_options, key='viz3d_sel')
    with _badge_col:
        _vm = _VIZ_META[viz_sel if 'viz3d_sel' not in st.session_state
                        else st.session_state.get('viz3d_sel', _viz_options[0])]
        _speed_key = _VIZ_META.get(viz_sel, ('', 'fast', '#238636'))
        st.markdown(
            f'<div style="margin-top:28px;padding:4px 10px;border-radius:4px;'
            f'background:{_speed_key[2]}22;border:1px solid {_speed_key[2]}55;'
            f'font-family:var(--mono);font-size:10px;color:{_speed_key[2]};'
            f'text-align:center;">'
            f'{_TIME_LABEL.get(_speed_key[1], _speed_key[1])}'
            f'</div>',
            unsafe_allow_html=True)

    # Descripción de la visualización seleccionada
    st.markdown(
        f'<div class="sci-info" style="margin:6px 0 10px;font-size:10px;">'
        f'{_VIZ_META.get(viz_sel, ("","",""))[0]}'
        f'</div>',
        unsafe_allow_html=True)

    # adaptar predict para viz3d — usa predict_fast (sin varianza RF → rápido)
    def _predict_for_viz(d, mid, geom_key, models):
        return models.predict_fast(d, mid)

    # ── Helper: botones de exportación en expander ───────────────────────────
    def _export_plotly(fig, fname_base):
        """Descarga HTML interactivo + intento PNG dentro de expander."""
        with st.expander('EXPORT — HTML / PNG', expanded=False):
            c_html, c_png = st.columns(2)
            html_bytes = fig.to_html(full_html=True, include_plotlyjs='cdn').encode()
            c_html.download_button(
                'DOWNLOAD HTML (interactive)',
                data=html_bytes,
                file_name=f'{fname_base}.html',
                mime='text/html',
                use_container_width=True,
            )
            try:
                png_bytes = fig.to_image(format='png', width=1400, height=800, scale=2)
                c_png.download_button(
                    'DOWNLOAD PNG (hi-res)',
                    data=png_bytes,
                    file_name=f'{fname_base}.png',
                    mime='image/png',
                    use_container_width=True,
                )
            except Exception:
                c_png.caption('PNG export requires `kaleido`  ·  `pip install kaleido`')

    # ── Geometría 3D Vóxel ────────────────────────────────────────────────────
    if viz_sel == 'Voxel 3D Geometry':
        with st.expander('Chart settings', expanded=False):
            n_vox = st.slider('Voxel resolution', 14, 30, 20, step=2, key='vox_n')
        _vox_key = f'fig_vox_{geom_id}_{d_nm}_{n_vox}'
        if _vox_key not in st.session_state:
            with st.spinner('Computing voxel geometry…'):
                st.session_state[_vox_key] = _viz3d.voxel_geometry_3d(
                    geom_id, d_nm, GEOMETRY_MODES, n=n_vox)
        fig_vox = st.session_state[_vox_key]
        st.plotly_chart(fig_vox, use_container_width=True)
        _export_plotly(fig_vox, f'voxel_{geom_id}_{d_nm:.0f}nm')

    # ── Mapa 2D de Magnetización ──────────────────────────────────────────────
    elif viz_sel == 'Magnetization 2D Map':
        with st.expander('Chart settings', expanded=False):
            n_grid_map = st.slider('Grid resolution', 18, 48, 28, step=2, key='map_grid')
        _map_key = f'fig_map_{mat_id}_{d_nm}_{n_grid_map}'
        if _map_key not in st.session_state:
            with st.spinner('Computing 2D magnetization maps…'):
                st.session_state[_map_key] = _viz3d.magnetization_map_2d(
                    mat_id, d_nm, MODELS, MATERIALS_DB, _predict_for_viz,
                    n_grid=n_grid_map)
        fig_map = st.session_state[_map_key]
        st.plotly_chart(fig_map, use_container_width=True)
        _export_plotly(fig_map, f'map2d_{mat_id}_{d_nm:.0f}nm')

    # ── Componentes de Energía (4 paneles) ────────────────────────────────────
    elif viz_sel == 'Energy Components':
        st.markdown(
            f'<div class="sci-info">'
            f'Four energy contributions for multiple particle sizes. '
            f'Magnitudes computed from physical parameters of '
            f'<strong>{mat["name"]}</strong>. '
            f'Inspired by Fig. 3 of Galvis, Mesa et al. (Results in Physics, 2025).'
            f'</div>',
            unsafe_allow_html=True)

        # ── Toggle: datos reales OOMMF ────────────────────────────────────────
        _show_real = False
        if _REAL_DATA_OK:
            _show_real = st.toggle(
                'Show real OOMMF data (2 Fe spheres, r=21 nm, sep=6 nm)',
                value=False, key='energy_show_real',
            )

        if _show_real and _REAL_DATA_OK:
            st.markdown(
                '<div class="sci-info">'
                '<strong>Real OOMMF data</strong> — System: 2 Fe spheres '
                '(r=21 nm, sep=6 nm) in 114×42×42 nm box · ±400 mT sweep · '
                'Runner: ExeOOMMFRunner · Source: <code>12nm.ipynb</code> '
                '(Galvis, Mesa et al.)'
                '</div>',
                unsafe_allow_html=True)
            _rd_key = 'fig_energy_real_oommf'
            if _rd_key not in st.session_state:
                with st.spinner('Loading real OOMMF data…'):
                    _hist   = _ref_data.load_hysteresis()
                    _energ  = _ref_data.load_energies()

                    _fig_real = go.Figure()

                    from plotly.subplots import make_subplots
                    _fig_real = make_subplots(
                        rows=3, cols=2,
                        subplot_titles=[
                            'Hysteresis Loop  (M/Ms)',
                            'Anisotropy Energy  (J)',
                            'Zeeman Energy  (J)',
                            'Dipolar Energy  (J)',
                            'Exchange Energy  (J)',
                            '',
                        ],
                        vertical_spacing=0.12,
                        horizontal_spacing=0.10,
                    )

                    _clr_desc = '#38bdf8'
                    _clr_asc  = '#fb923c'

                    # Panel 1: Hysteresis
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
                    _Hc_r, _Mr_r = _ref_data.extract_hc_mr()
                    _fig_real.add_vline(x= _Hc_r, line_dash='dot',
                        line_color='#f43f5e', line_width=1.5, row=1, col=1)
                    _fig_real.add_vline(x=-_Hc_r, line_dash='dot',
                        line_color='#f43f5e', line_width=1.5, row=1, col=1)

                    def _add_energy_panel(key, row, col, label):
                        _e = _energ.get(key)
                        if _e is None:
                            return
                        _fig_real.add_trace(go.Scatter(
                            x=_e['fd_desc'], y=_e['mg_desc'],
                            mode='lines+markers', name=f'{label} ↓', marker_size=4,
                            line=dict(color=_clr_desc, width=2), showlegend=False,
                        ), row=row, col=col)
                        _fig_real.add_trace(go.Scatter(
                            x=_e['fd_asc'], y=_e['mg_asc'],
                            mode='lines+markers', name=f'{label} ↑', marker_size=4,
                            line=dict(color=_clr_asc, width=2), showlegend=False,
                        ), row=row, col=col)

                    _add_energy_panel('anisotropy', row=1, col=2, label='E_anis')
                    _add_energy_panel('zeeman',     row=2, col=1, label='E_Z')
                    _add_energy_panel('dipolar',    row=2, col=2, label='E_dip')
                    _add_energy_panel('exchange',   row=3, col=1, label='E_ex')

                    _fig_real.update_layout(
                        height=850,
                        paper_bgcolor='#0d1117',
                        plot_bgcolor='#161b22',
                        font=dict(color='#e6edf3', size=11),
                        title=dict(
                            text='Real OOMMF — 2 Fe spheres · r=21 nm · sep=6 nm · ±400 mT',
                            font=dict(color='#e6edf3', size=14),
                        ),
                        legend=dict(
                            bgcolor='#161b22', bordercolor='#30363d',
                            title=dict(text='Branch'),
                        ),
                        margin=dict(t=80, b=40, l=60, r=30),
                    )
                    for _ax in _fig_real.layout:
                        if _ax.startswith('xaxis') or _ax.startswith('yaxis'):
                            _fig_real.layout[_ax].update(
                                gridcolor='#21262d', zerolinecolor='#30363d',
                                color='#8b949e',
                            )
                    for _r, _c in [(1,1),(1,2),(2,1),(2,2),(3,1)]:
                        _fig_real.update_xaxes(title_text='H (mT)', row=_r, col=_c)
                    _ylabels = {(1,1):'M/Ms',(1,2):'E (J)',(2,1):'E (J)',
                                (2,2):'E (J)',(3,1):'E (J)'}
                    for (_r,_c), _lbl in _ylabels.items():
                        _fig_real.update_yaxes(title_text=_lbl, row=_r, col=_c)

                    st.session_state[_rd_key] = _fig_real

            fig_real_oommf = st.session_state[_rd_key]
            st.plotly_chart(fig_real_oommf, use_container_width=True)
            _export_plotly(fig_real_oommf, 'energy_real_oommf_2spheres_fe')

            # Métricas extraídas de los datos reales
            st.markdown(
                '<div class="sci-section-title" style="margin-top:12px">'
                'EXTRACTED VALUES — Real OOMMF data</div>',
                unsafe_allow_html=True)
            _rp = _ref_data.REFERENCE_PARAMS
            _rm1, _rm2, _rm3, _rm4, _rm5 = st.columns(5)
            _rm1.metric('Hc (mT)',       f"{_rp['Hc_mT']:.1f}")
            _rm2.metric('Mr / Ms',        f"{_rp['Mr_Ms']:.4f}")
            _rm3.metric('H_max (mT)',     f"{_rp['H_max_mT']:.0f}")
            _rm4.metric('Radius (nm)',    f"{_rp['radius_nm']:.0f}")
            _rm5.metric('Separation (nm)',f"{_rp['separation_nm']:.0f}")
            st.caption(
                f"Material: **{_rp['material']}**  ·  "
                f"Ms = {_rp['Ms_Am']/1e6:.2f} MA/m  ·  "
                f"K₁ = {_rp['K1_Jm3']/1e3:.0f} kJ/m³  ·  "
                f"A = {_rp['A_Jm']*1e12:.1f} pJ/m  ·  "
                f"Cell = {_rp['cell_nm']:.0f} nm  ·  "
                f"Source: `{_rp['source_nb']}`"
            )

        else:
            with st.expander('Chart settings', expanded=False):
                n_curvas = st.slider('Number of sizes', 1, 10, 5, key='energy_n')
            _eng_key = f'fig_energy_{mat_id}_{n_curvas}_{int(T_K)}_{use_temp_correction}'
            if _eng_key not in st.session_state:
                with st.spinner('Computing energy components…'):
                    st.session_state[_eng_key] = _viz3d.energy_components_4panel(
                        mat_id, MODELS, MATERIALS_DB, _predict_for_viz, n_sizes=n_curvas)
            fig_energy4 = st.session_state[_eng_key]
            st.plotly_chart(fig_energy4, use_container_width=True)
            _export_plotly(fig_energy4, f'energy_{mat_id}_{d_nm:.0f}nm')

    elif viz_sel == 'Energy Surface  E(H, d)':
        _surf_key = f'fig_surf_{mat_id}_{int(T_K)}_{use_temp_correction}'
        if _surf_key not in st.session_state:
            with st.spinner('Computing 3D energy surface…'):
                st.session_state[_surf_key] = _viz3d.surface_energy_3d(
                    mat_id, MODELS, MATERIALS_DB, _predict_for_viz)
        fig_surf = st.session_state[_surf_key]
        st.plotly_chart(fig_surf, use_container_width=True)
        _export_plotly(fig_surf, f'surface_e_{mat_id}_{d_nm:.0f}nm')

    elif viz_sel == 'Magnetization Vectors':
        _vec_key = f'fig_vecs_{mat_id}_{d_nm}_{int(T_K)}_{use_temp_correction}'
        if _vec_key not in st.session_state:
            with st.spinner('Generating magnetization vectors…'):
                st.session_state[_vec_key] = _viz3d.magnetization_vectors(
                    mat_id, d_nm, MODELS, MATERIALS_DB, _predict_for_viz)
        fig_vecs = st.session_state[_vec_key]
        st.plotly_chart(fig_vecs, use_container_width=True)
        _export_plotly(fig_vecs, f'vectors_m_{mat_id}_{d_nm:.0f}nm')

    elif viz_sel == 'Heatmap  Hc(material, nm)':
        st.markdown(
            '<div class="sci-info">'
            'Hc heatmap across all materials and diameters. '
            'Each cell shows the predicted coercive field (mT) for that '
            'material–size combination.'
            '</div>',
            unsafe_allow_html=True)
        with st.expander('Chart settings', expanded=False):
            geom_hm = st.radio('Base geometry', ['sphere', 'cuboid'],
                                format_func=lambda x: 'Sphere' if x == 'sphere' else 'Cuboid',
                                horizontal=True, key='hm_geom')
        _hm_key = f'fig_hm_{geom_hm}'
        if _hm_key not in st.session_state:
            with st.spinner('Computing Hc heatmap…'):
                st.session_state[_hm_key] = _viz3d.hc_heatmap(
                    MODELS, MATERIALS_DB, _predict_for_viz, geom=geom_hm)
        fig_hm = st.session_state[_hm_key]
        st.plotly_chart(fig_hm, use_container_width=True)
        _export_plotly(fig_hm, f'heatmap_hc_{geom_hm}')

    elif viz_sel == 'Polar Anisotropy  E_K(θ)':
        st.markdown(
            '<div class="sci-info">'
            'E_K(θ) = K₁·sin²(θ), normalized to K₁_max. '
            'Larger area = higher anisotropy barrier. '
            'Compare materials to assess magnetic hardness.'
            '</div>',
            unsafe_allow_html=True)
        mats_polar = st.multiselect(
            'Materials',
            list(MATERIALS_DB.keys()), default=list(MATERIALS_DB.keys()),
            format_func=lambda x: f"{x.upper()} · {MATERIALS_DB[x]['name']}",
            key='polar_mats',
        )
        if mats_polar:
            fig_polar = _viz3d.polar_anisotropy(MATERIALS_DB, mat_ids=mats_polar)
            st.plotly_chart(fig_polar, use_container_width=True)
            _export_plotly(fig_polar, 'polar_anisotropy')
        else:
            st.markdown(
                '<div class="sci-warn">Select at least one material.</div>',
                unsafe_allow_html=True)

    elif viz_sel == 'Hysteresis 3D Stack':
        with st.expander('Chart settings', expanded=False):
            n_lazos = st.slider('Number of loops', 1, 14, 6, key='stack_n')
        _stack_key = f'fig_stack_{mat_id}_{n_lazos}_{int(T_K)}_{use_temp_correction}'
        if _stack_key not in st.session_state:
            with st.spinner('Generating 3D hysteresis stack…'):
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
    st.markdown('<div class="sci-section-title">Dashboard — SQLite + Session</div>',
                unsafe_allow_html=True)

    db_rows  = _db.get_all_simulations()
    db_stats = _db.get_stats()

    source_tab, hist_tab = st.tabs(['SQLite (persistent)', 'Session history'])

    with source_tab:
        if not db_rows:
            st.info('No records yet. Run a simulation to populate the database.')
        else:
            df_db = pd.DataFrame(db_rows)
            kp1,kp2,kp3,kp4,kp5 = st.columns(5)
            kp1.metric('Total simulations',  db_stats.get('total', 0))
            kp2.metric('Unique materials',   db_stats.get('unique_materials', 0))
            kp3.metric('Min size (nm)',       f"{db_stats.get('min_size', 0):.1f}")
            kp4.metric('Max size (nm)',       f"{db_stats.get('max_size', 0):.1f}")
            kp5.metric('Extrapolations',      db_stats.get('total_extrapolations', 0))
            st.divider()
            if len(df_db) > 1:
                st.markdown('**Historical Hc evolution**')
                _fig_db2 = go.Figure()
                for _db_mat in df_db['material'].unique():
                    _db_sub   = df_db[df_db['material'] == _db_mat].dropna(subset=['hc_val'])
                    _db_color = next((v['color'] for v in MATERIALS_DB.values()
                                      if v['name'] == _db_mat), '#38bdf8')
                    if not _db_sub.empty:
                        _fig_db2.add_trace(go.Scatter(
                            x=list(range(len(_db_sub))),
                            y=_db_sub['hc_val'].values,
                            name=_db_mat,
                            mode='lines+markers',
                            line=dict(color=_db_color, width=1.8),
                            marker=dict(size=5),
                            hovertemplate=f'<b>{_db_mat}</b><br>Record #%{{x}}<br>Hc = %{{y:.1f}} mT<extra></extra>',
                        ))
                _apply_plotly_theme(_fig_db2, title='SQLite History — Hc', height=260)
                _fig_db2.update_layout(
                    xaxis_title='Record #', yaxis_title='Hc (mT)',
                    hovermode='x unified',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02,
                                xanchor='right', x=1, font_size=9),
                )
                st.plotly_chart(_fig_db2, use_container_width=True,
                                config={'displayModeBar': False})
            st.divider()
            cols_show = ['id','timestamp','material','size_nm','geometry',
                         'hc_val','mr_val','extrapolation']
            st.dataframe(df_db[cols_show], use_container_width=True, hide_index=True)
            if st.button('CLEAR SQLite database', key='clr_db'):
                _db.clear_simulations()
                st.rerun()

    with hist_tab:
        hist = st.session_state.history
        if not hist:
            st.info('No simulations in this session yet. Run a simulation to start tracking.')
        else:
            df_hist = pd.DataFrame(hist)
            kp1,kp2,kp3,kp4 = st.columns(4)
            kp1.metric('Simulations',     len(df_hist))
            kp2.metric('Unique materials', df_hist['Material'].nunique())
            kp3.metric('Min size (nm)',    df_hist['Size (nm)'].min())
            kp4.metric('Max size (nm)',    df_hist['Size (nm)'].max())
            st.divider()
            st.dataframe(df_hist, use_container_width=True, hide_index=True)
            st.divider()
            st.markdown('**Distribution by material**')
            counts = df_hist['Material'].value_counts()
            _pie_colors = [next((v['color'] for v in MATERIALS_DB.values()
                                 if v['name'] == nm), '#38bdf8')
                           for nm in counts.index]
            _fig_pie = go.Figure(go.Pie(
                labels=counts.index,
                values=counts.values,
                marker=dict(colors=_pie_colors,
                            line=dict(color='#0f172a', width=2)),
                textinfo='label+percent',
                textfont=dict(size=11, color='#f1f5f9'),
                hovertemplate='<b>%{label}</b><br>%{value} simulations (%{percent})<extra></extra>',
            ))
            _apply_plotly_theme(_fig_pie, title='Simulations by material', height=350)
            _fig_pie.update_layout(showlegend=False)
            st.plotly_chart(_fig_pie, use_container_width=True,
                            config={'displayModeBar': False})

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 6 — EXPORTAR
# ─────────────────────────────────────────────────────────────────────────────
with tab_export:
    st.markdown('<div class="sci-section-title">Export Results</div>',
                unsafe_allow_html=True)

    # ── PDF ────────────────────────────────────────────────────────────────
    st.markdown('#### Scientific PDF Report')
    st.markdown('Cover · physical parameters · ML predictions · equations · references.')
    if st.button('GENERATE PDF REPORT', use_container_width=True, type='primary'):
        with st.spinner('Generating PDF report…'):
            pdf_bytes = _report.generate_report(
                mat_id=mat_id, mat_name=mat['name'], d_nm=d_nm,
                preds={gm['name']: (Hc_val, Mr_val)},
                mat_params=mat['params'], mat_range=mat['range'],
                field_max=mat['field_max'], fig_main=None, fig_energy=None,
                noise_level=noise_level,
                extrapolation=is_extrapolation(d_nm, mat_id),
                history_rows=_db.get_all_simulations()[:30],
                T_K=T_K, geom_name=gm['name'],
            )
        fname_pdf = f'report_{mat_id}_{geom_id}_{d_nm:.0f}nm.pdf'
        st.download_button(
            label=f'DOWNLOAD {fname_pdf}', data=pdf_bytes,
            file_name=fname_pdf, mime='application/pdf',
            use_container_width=True)
        _db.log_report(mat_id, d_nm, fname_pdf)
        st.success(f'{fname_pdf} generated.')

    st.divider()
    ex1, ex2 = st.columns(2)

    with ex1:
        st.markdown('#### Simulation Figure (PNG)')
        st.caption(f'Main simulation canvas exported as PNG · DPI: {export_dpi}')
        buf_png = io.BytesIO()
        fig_main.savefig(buf_png, format='png', dpi=export_dpi,
                         bbox_inches='tight', facecolor='#0d1117')
        buf_png.seek(0)
        fname_png = f'fig_{mat_id}_{geom_id}_{d_nm}nm.png'
        st.download_button(f'DOWNLOAD {fname_png}', buf_png, fname_png, 'image/png',
                           use_container_width=True)

    with ex2:
        st.markdown('#### Hysteresis Loop (CSV)')
        st.caption(f'500 pts · H ∈ [−{mat["field_max"]}, +{mat["field_max"]}] mT · columns: H, M_up/Ms, M_dn/Ms')
        H_csv, M_up_csv, M_dn_csv = llg_hysteresis(
            Hc_val, Mr_val, H_max=mat['field_max'],
            noise_level=noise_level, seed=42)
        df_csv = pd.DataFrame({'H (mT)': H_csv,
                                'M_up/Ms': M_up_csv, 'M_dn/Ms': M_dn_csv})
        fname_csv = f'hysteresis_{mat_id}_{geom_id}_{d_nm}nm.csv'
        st.download_button(f'DOWNLOAD {fname_csv}',
                           df_csv.to_csv(index=False).encode(),
                           fname_csv, 'text/csv', use_container_width=True)

    st.divider()
    st.markdown('#### SQLite History (CSV)')
    st.caption('Full simulation log from the persistent database — all sessions.')
    if db_rows:
        st.download_button(
            'DOWNLOAD simulation_history.csv',
            pd.DataFrame(db_rows).to_csv(index=False).encode(),
            'simulation_history.csv', 'text/csv', use_container_width=True)
    else:
        st.info('No records yet. Run a simulation to populate the database.')

    st.divider()
    st.markdown('#### Material Parameters (JSON)')
    st.caption(f'Physical constants and ML training range for {mat["name"]}.')
    mat_json = {mat_id: {k: v for k, v in mat.items()
                          if k not in ('sphere', 'cuboid')}}
    def _np_serial(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        return o
    st.download_button(
        f'DOWNLOAD params_{mat_id}.json',
        json.dumps(mat_json, default=_np_serial, indent=2, ensure_ascii=False).encode(),
        f'params_{mat_id}.json', 'application/json', use_container_width=True)

plt.close(fig_main)

# ─────────────────────────────────────────────────────────────────────────────
#  TAB 7 — VALIDACIÓN UBERMAG  (discretisedfield + micromagneticmodel + oommfc)
# ─────────────────────────────────────────────────────────────────────────────
with tab_uval:
    st.markdown('<div class="sci-section-title">Geometry Validation — Ubermag / OOMMF</div>',
                unsafe_allow_html=True)
    st.markdown(
        '<div class="sci-info">'
        'Physical validation of the 8 geometries using the '
        '<strong>Ubermag</strong> scientific framework '
        '(discretisedfield + micromagneticmodel + oommfc). '
        'Shape factors are derived from analytical demagnetization factors '
        '(Osborn 1945, Chen 1991, Aharoni 1998) and compared against the '
        'values used in the simulator.'
        '</div>',
        unsafe_allow_html=True)

    # ── Subtabs de validación ─────────────────────────────────────────────────
    uv_tab1, uv_tab2, uv_tab3, uv_tab4 = st.tabs([
        'Geometry (discretisedfield)',
        'Shape Factors (Stoner-Wohlfarth)',
        'Anisotropy Radar',
        'OOMMF Simulation',
    ])

    # Crear validador (cacheado en session_state)
    if 'ubermag_validator' not in st.session_state:
        d_val = st.session_state.get('d_nm', 20.0)
        st.session_state['ubermag_validator'] = _uval.UbermagValidator(
            MATERIALS_DB, GEOMETRY_MODES, d_test_nm=d_val)

    validator = st.session_state['ubermag_validator']

    # ── Sub-tab 1: Geometría ──────────────────────────────────────────────────
    with uv_tab1:
        st.markdown(
            '<div class="sci-info">'
            'Each geometry is built as a 3D discretized mesh. '
            'Discretized volume is measured and compared to the analytical sphere volume. '
            'Nd factors are computed with exact analytical formulas. '
            'Tolerance band: V/V_sphere within <strong>±15 %</strong> = pass.'
            '</div>',
            unsafe_allow_html=True)

        # ── Availability notice ───────────────────────────────────────────────
        if not _uval._HAS_DISCRETISEDFIELD:
            st.info(
                '**discretisedfield / micromagneticmodel not installed** — '
                'showing analytical geometry approximations instead of discrete mesh data.  \n'
                'To enable full mesh validation:  \n'
                '```\npip install -r requirements-oommf.txt\n```',
                icon='ℹ️',
            )

        _uv1_col1, _uv1_col2 = st.columns(2, gap='small')
        with _uv1_col1:
            d_val_input = st.slider(
                'Test diameter (nm)', 10, 60, 20, step=5, key='uval_d')
        with _uv1_col2:
            cell_val = st.select_slider(
                'Cell size (nm)', [1.5, 2.0, 2.5, 3.0], value=2.5, key='uval_cell')

        _geom_key = f'uval_geom_{d_val_input}_{cell_val}'
        if _geom_key not in st.session_state:
            _spinner_msg = (
                'Computing analytical geometry approximations…'
                if not _uval._HAS_DISCRETISEDFIELD else
                'Building meshes with discretisedfield…'
            )
            with st.spinner(_spinner_msg):
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

        # Tabla de métricas geométricas con pass/fail
        import pandas as pd
        _is_fallback = _uval.UBERMAG_AVAILABLE is False
        rows_g = []
        for gid, m in gd['metrics'].items():
            _vr  = m['V_rel']
            _tol = '✓ pass' if 0.85 <= _vr <= 1.15 else ('~ warn' if 0.80 <= _vr <= 1.20 else '✗ fail')
            _cells = (
                '—' if m.get('analytical_fallback') else str(m['n_cells'])
            )
            rows_g.append({
                'Geometry':       _uval.GEOM_LABELS[gid].split('(')[0].strip(),
                'Phys. ref.':     _uval.GEOM_LABELS[gid].split('(')[-1].replace(')','')
                                  if '(' in _uval.GEOM_LABELS[gid] else '—',
                'Valid cells':    _cells,
                'V (nm³)':        f"{m['V_nm3']:.1f}",
                'V / V_sphere':   f"{_vr:.3f}",
                'Status':         _tol,
                'N_z (analyt.)':  f"{m.get('Nd_z', 0.333):.3f}",
                'N_x (analyt.)':  f"{m.get('Nd_x', 0.333):.3f}",
                'ΔN':             f"{m.get('Nd_aniso', 0):.4f}",
            })
        st.dataframe(pd.DataFrame(rows_g), use_container_width=True, hide_index=True)

        st.markdown(
            '<div class="sci-info">'
            '<strong>Analytical Nd</strong> — Osborn (1945) for ellipsoids, '
            'Chen (1991) for cylinders, Aharoni (1998) for cuboids, '
            'Field et al. (2011) for toroid, Nogués et al. (1999) for core-shell.'
            '</div>',
            unsafe_allow_html=True)

    # ── Sub-tab 2: Factores Stoner-Wohlfarth ─────────────────────────────────
    with uv_tab2:
        st.markdown(
            '<div class="sci-info">'
            'Simulator shape factors are compared against values computed by the '
            'Stoner-Wohlfarth model using analytical Nd factors. '
            'Shape anisotropy ΔN·Ms modifies the switching field relative to the '
            'sphere reference.'
            '</div>',
            unsafe_allow_html=True)

        mat_sw = st.selectbox(
            'Reference material for SW factors',
            list(MATERIALS_DB.keys()),
            format_func=lambda x: f"{x.upper()} · {MATERIALS_DB[x]['name']}",
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
            _err = float(str(f['hc_error_pct']).replace('%',''))
            _status = '✓ pass' if abs(_err) <= 10 else ('~ warn' if abs(_err) <= 20 else '✗ fail')
            rows_sw.append({
                'Geometry':        _uval.GEOM_LABELS[gid].split('(')[0].strip(),
                'ΔN':              f"{f['delta_Nd']:+.3f}",
                'factor_hc (app)': f['factor_hc_app'],
                'factor_hc (SW)':  f['factor_hc_sw'],
                'Δ Hc %':          f"{f['hc_error_pct']}%",
                'Status':          _status,
                'factor_mr (app)': f['factor_mr_app'],
                'factor_mr (SW)':  f['factor_mr_sw'],
            })
        df_sw = pd.DataFrame(rows_sw)
        st.dataframe(df_sw, use_container_width=True, hide_index=True)

        st.markdown(
            '<div class="sci-info">'
            '<strong>Note:</strong> SW model assumes coherent rotation '
            '(valid for d &lt; d_crit single-domain). For larger particles, '
            'domain-wall motion reduces real Hc below H_sw.'
            '</div>',
            unsafe_allow_html=True)

        # Material parameters strip
        p_sw = MATERIALS_DB[mat_sw]['params']
        Ms   = p_sw['Ms_MA_m'] * 1e6
        K1   = abs(p_sw['K1_kJ_m3']) * 1e3
        H_mca_mT   = round(2 * K1 / Ms * 1e3 / (4*np.pi*1e-7) / 1e3, 1) if Ms > 0 else 0
        H_shape_mT = round((1/3) * Ms * 4*np.pi*1e-7 * 1e3, 1)
        st.markdown(
            f'<div class="sci-info">'
            f'<strong>{MATERIALS_DB[mat_sw]["name"]}</strong> &nbsp;|&nbsp; '
            f'Ms = {p_sw["Ms_MA_m"]} MA/m &nbsp;|&nbsp; '
            f'K₁ = {p_sw["K1_kJ_m3"]} kJ/m³ &nbsp;|&nbsp; '
            f'H_K,mca ≈ {H_mca_mT} mT &nbsp;|&nbsp; '
            f'H_shape (sphere) ≈ {H_shape_mT} mT'
            f'</div>',
            unsafe_allow_html=True)

    # ── Sub-tab 3: Radar de anisotropía ───────────────────────────────────────
    with uv_tab3:
        st.markdown(
            '<div class="sci-info">'
            'ΔN &gt; 0: shape easy axis (increases Hc). &nbsp;'
            'ΔN &lt; 0: shape easy plane (reduces Hc). &nbsp;'
            'ΔN = 0: isotropic (sphere). &nbsp;'
            'Trace = N_x + N_y + N_z = 1 (flux conservation).'
            '</div>',
            unsafe_allow_html=True)
        if 'uval_radar_fig' not in st.session_state:
            v_rad = _uval.UbermagValidator(MATERIALS_DB, GEOMETRY_MODES)
            st.session_state['uval_radar_fig'] = v_rad.plot_Nd_radar()
        st.plotly_chart(
            st.session_state['uval_radar_fig'], use_container_width=True)

        # Tabla de Nd completa
        rows_nd = []
        for gid, Nd in _uval.GEOM_Nd.items():
            rows_nd.append({
                'Geometry':  _uval.GEOM_LABELS[gid].split('(')[0].strip(),
                'Reference': _uval.GEOM_LABELS[gid].split('(')[-1].replace(')','')
                             if '(' in _uval.GEOM_LABELS[gid] else '—',
                'N_x':  f"{Nd[0]:.4f}",
                'N_y':  f"{Nd[1]:.4f}",
                'N_z':  f"{Nd[2]:.4f}",
                'ΔN':   f"{Nd[0]-Nd[2]:+.4f}",
                'Trace':f"{sum(Nd):.4f}",
            })
        st.dataframe(pd.DataFrame(rows_nd), use_container_width=True, hide_index=True)

    # ── Sub-tab 4: Simulación OOMMF ───────────────────────────────────────────
    with uv_tab4:
        st.markdown('#### Full OOMMF simulation via `oommfc`')

        # ── Comprobación de disponibilidad ───────────────────────────────────
        _oommf_avail = False
        try:
            import oommfc as _oc
            _oc.oommf.DockerOOMMFRunner(image='ubermag/oommf')
            _oommf_avail = True
        except Exception:
            pass

        if _oommf_avail:
            st.markdown(
                '<div class="sci-ok" style="padding:8px 12px;border-radius:4px;'
                'border-left:3px solid var(--ok);">'
                '<strong>oommfc available</strong> — Docker with '
                '<code>ubermag/oommf</code> detected. Full OOMMF simulation will run.'
                '</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                '<div class="sci-info">'
                '<strong>OOMMF/Docker not detected</strong> in this environment — '
                'analytical <strong>Stoner-Wohlfarth</strong> simulation with Ubermag '
                'shape correction will be used as fallback. '
                'To enable OOMMF: <code>docker pull ubermag/oommf</code>'
                '</div>',
                unsafe_allow_html=True)

        st.divider()

        # ── Controles de la simulación ───────────────────────────────────────
        _col_s1, _col_s2, _col_s3 = st.columns([1, 1, 1])
        with _col_s1:
            _sim_geom = st.selectbox(
                'Geometry',
                list(GEOMETRY_MODES.keys()),
                format_func=lambda g: GEOMETRY_MODES[g]['name'],
                key='oommf_geom',
            )
            _sim_mat = st.selectbox(
                'Material',
                list(MATERIALS_DB.keys()),
                format_func=lambda m: f"{m.upper()} · {MATERIALS_DB[m]['name']}",
                key='oommf_mat',
            )
        with _col_s2:
            _sim_d    = st.slider('Diameter (nm)',    5,  60,  20,  1,  key='oommf_d')
            _sim_Hmax = st.slider('Max field (mT)', 100, 800, 300, 50, key='oommf_Hmax')
        with _col_s3:
            _sim_steps = st.slider('Field steps',  20, 80, 40, 10, key='oommf_steps')
            _sim_cell  = st.slider('Cell size (nm)', 2,  5,  3,  1, key='oommf_cell')

        _run_sim = st.button('▶ Run Simulation', type='primary',
                             key='btn_run_oommf')

        _sim_key = (f'oommf_res_{_sim_geom}_{_sim_mat}_{_sim_d}'
                    f'_{_sim_Hmax}_{_sim_steps}_{_sim_cell}')

        # Recalcular si el usuario presionó el botón
        if _run_sim:
            st.session_state.pop(_sim_key, None)

        if _run_sim or _sim_key in st.session_state:

            if _sim_key not in st.session_state:
                with st.spinner('Running micromagnetic simulation…'):
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
            st.markdown(
                '<div class="sci-section-title" style="margin-top:8px">'
                'SIMULATION RESULTS</div>',
                unsafe_allow_html=True)
            _mc1, _mc2, _mc3, _mc4 = st.columns(4)
            _mc1.metric('Hc simulated (mT)', f"{_res['Hc_mT']:.1f}")
            _mc2.metric('Mr / Ms',           f"{_res['Mr']:.3f}")
            _mc3.metric('Active cells',      _res.get('n_cells', '—'))
            _mc4.metric('Engine',            _res.get('runner', '—'))

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
                mode='lines', name='H ↓ descending',
                line=dict(color='#38bdf8', width=2.5),
                hovertemplate='H=%{x:.1f} mT<br>M/Ms=%{y:.4f}<extra>↓ branch</extra>',
            ))

            # Rama ascendente
            _fig_h.add_trace(go.Scatter(
                x=_H[_nh:], y=_M[_nh:],
                mode='lines', name='H ↑ ascending',
                line=dict(color='#fb923c', width=2.5),
                hovertemplate='H=%{x:.1f} mT<br>M/Ms=%{y:.4f}<extra>↑ branch</extra>',
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
                name='Remanence (H=0)',
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
            _geom_name = GEOMETRY_MODES[_sim_geom]['name']
            _apply_plotly_theme(
                _fig_h,
                title=f'Hysteresis Loop — {_geom_name}  ·  {_mat_name}  ·  {_sim_d} nm',
                xaxis_title='Applied field  H  (mT)',
                yaxis_title='Reduced magnetization  M / Ms',
            )
            _fig_h.update_layout(
                height=450,
                yaxis=dict(range=[-1.18, 1.18]),
            )
            st.plotly_chart(_fig_h, use_container_width=True)

            # ── Comparación ML predicho vs. Simulado ─────────────────────────
            st.markdown(
                '<div class="sci-section-title" style="margin-top:8px">'
                'ML PREDICTED vs. SIMULATED</div>',
                unsafe_allow_html=True)
            _Hc_ml, _Mr_ml = MODELS.predict_fast(
                float(_sim_d), _sim_mat,
                geom_factor_hc=GEOMETRY_MODES[_sim_geom]['factor_hc'],
                geom_factor_mr=GEOMETRY_MODES[_sim_geom]['factor_mr'],
            )
            _Hc_err = abs(_res['Hc_mT'] - _Hc_ml) / max(_res['Hc_mT'], 1.0) * 100
            _Mr_err = abs(_res['Mr']    - _Mr_ml)  / max(_res['Mr'],    0.01) * 100
            _cmp_df = pd.DataFrame({
                'Source':  ['Simulation (OOMMF/SW)', 'ML Ensemble (predicted)'],
                'Hc (mT)': [f"{_res['Hc_mT']:.1f}", f"{_Hc_ml:.1f}"],
                'Mr / Ms': [f"{_res['Mr']:.3f}",     f"{_Mr_ml:.3f}"],
                'Engine':  [_res.get('runner', '—'), 'GBR + RF + MLP'],
                'Δ Hc':    ['—', f'{_Hc_err:.1f} %'],
                'Δ Mr':    ['—', f'{_Mr_err:.1f} %'],
            })
            st.dataframe(_cmp_df, use_container_width=True, hide_index=True)
            st.markdown(
                f'<div class="sci-info" style="font-size:10px;">'
                f'Relative error — Hc: <strong>{_Hc_err:.1f} %</strong> &nbsp;·&nbsp; '
                f'Mr: <strong>{_Mr_err:.1f} %</strong> &nbsp;—&nbsp; '
                f'model trained on LLG hysteresis + reference OOMMF data'
                f'</div>',
                unsafe_allow_html=True)

            # ── Validación contra datos reales (12nm.ipynb) ───────────────────
            if _REAL_DATA_OK and _ref_data is not None:
                _Hc_real, _Mr_real = _ref_data.extract_hc_mr()
                with st.expander(
                    'OOMMF validation — real data (2 Fe spheres · 12nm.ipynb)',
                    expanded=(_sim_mat == 'fe')
                ):
                    st.info(
                        '**Referencia real**: simulación con ExeOOMMFRunner de 2 esferas '
                        f'de Fe (r=21 nm, sep=6 nm, celda=3 nm, ±400 mT).  '
                        f'Hc_real = **{_Hc_real:.1f} mT**  ·  Mr_real = **{_Mr_real:.4f}**'
                    )
                    _val_df = pd.DataFrame({
                        'Source': [
                            'OOMMF real (12nm.ipynb)',
                            f'Current simulation ({_res.get("runner","SW")})',
                            'ML Ensemble',
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
                        title='Hysteresis: Real OOMMF vs. current simulation',
                        xaxis_title='Applied field  H  (mT)',
                        yaxis_title='Reduced magnetization  M / Ms',
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
                with st.expander('UBERMAG REPRODUCIBLE CODE — `12nm.ipynb`'):
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
            with st.expander('MICROMAGNETIC SYSTEM PARAMETERS'):
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
                        f'  Initial M: +z (saturated state)\n'
                        f'\nDriver:\n'
                        f'  MinDriver (field-step relaxation)\n'
                        f'  Sweep: +{_sim_Hmax} → −{_sim_Hmax} → +{_sim_Hmax} mT '
                        f'({_sim_steps} steps)\n'
                    )
                    st.code(_sys_str, language='text')
                except Exception as _e:
                    st.info(f'Parameters unavailable (discretisedfield required): {_e}')

        st.divider()

        # ── Tabla de factores validados con Ubermag ───────────────────────────
        st.markdown('##### Validated factors from Ubermag (used in the simulator)')
        _rows_vf = []
        for _gid, _vf in _uval.VALIDATED_FACTORS.items():
            _rows_vf.append({
                'Geometry':          _uval.GEOM_LABELS[_gid].split('(')[0].strip(),
                'factor_hc':         _vf['factor_hc'],
                'factor_mr':         _vf['factor_mr'],
                'N_z':               _vf['Nd_z'],
                'N_x':               _vf['Nd_x'],
                'Physical ref.':     _vf['ref'],
            })
        st.dataframe(
            pd.DataFrame(_rows_vf), use_container_width=True, hide_index=True)
        st.success(
            'The factors above are the values **currently used** '
            'in `GEOMETRY_MODES`, derived from analytical demagnetization calculations '
            'validated with the Ubermag stack.'
        )

        st.divider()

        # ══════════════════════════════════════════════════════════════════════
        # ── 📂 Gestión dinámica de datos OOMMF ────────────────────────────────
        # ══════════════════════════════════════════════════════════════════════
        st.markdown('#### OOMMF Data — Dynamic Loading &amp; Training')
        st.markdown(
            'The simulator automatically detects new files in `oommf_data/`. '
            'Upload `.txt` (fd/mg) or `.ipynb` files to expand the calibration '
            'dataset and improve ML predictions.'
        )

        # ── Estado actual del dataset ─────────────────────────────────────────
        if _ref_data is not None:
            _dssum = _ref_data.dataset_summary()
        else:
            _dssum = {}

        _ds1, _ds2, _ds3, _ds4, _ds5 = st.columns(5)
        _ds1.metric('Hysteresis cycles',  _dssum.get('n_hysteresis', 0))
        _ds2.metric('Energy series',      _dssum.get('n_energies',   0))
        _ds3.metric('Notebooks (.ipynb)', _dssum.get('n_notebooks',  0))
        _ds4.metric('ML calibration pts', _dssum.get('n_calibration', 0))
        _ds5.metric('Materials detected', len(_dssum.get('materials', [])))

        # ── Browser de datasets ───────────────────────────────────────────────
        with st.expander('BROWSE AVAILABLE DATASETS', expanded=False):
            if _REAL_DATA_OK and _ref_data is not None:
                import oommf_data_manager as _odm
                _all_ds = _odm.scan_datasets(_dssum.get('data_dir'))

                # Histéresis
                if _all_ds['hysteresis']:
                    st.markdown('**Hysteresis cycles detected**')
                    _h_rows = []
                    for _h in _all_ds['hysteresis']:
                        _h_rows.append({
                            'File':       _h['filename'],
                            'Points':     _h['n_points'],
                            'Hc (mT)':   f"{_h.get('Hc_mT', '?'):.1f}",
                            'Mr / Ms':   f"{_h.get('Mr_Ms', '?'):.4f}",
                            'H_max (mT)': f"{_h.get('H_max_mT', '?'):.0f}",
                        })
                    st.dataframe(pd.DataFrame(_h_rows),
                                 use_container_width=True, hide_index=True)

                # Energies
                if _all_ds['energies']:
                    st.markdown('**Energy series detected**')
                    _e_rows = [
                        {'File': e['filename'], 'Type': e['dtype'],
                         'Label': e['label'], 'Points': e['n_points']}
                        for e in _all_ds['energies']
                    ]
                    st.dataframe(pd.DataFrame(_e_rows),
                                 use_container_width=True, hide_index=True)

                # Notebooks
                if _all_ds['notebooks']:
                    st.markdown('**Notebooks detected**')
                    _nb_rows = [
                        {
                            'File':      nb.get('source_nb', '?'),
                            'Material':  nb.get('material_guess', '?'),
                            'Ms (MA/m)': f"{nb.get('Ms_Am', 0)/1e6:.2f}",
                            'K (kJ/m³)': f"{nb.get('K1_Jm3', 0)/1e3:.0f}",
                            'r (nm)':    f"{nb.get('radius_nm', '?')}",
                            'sep (nm)':  f"{nb.get('separation_nm', '?')}",
                            'cell (nm)': f"{nb.get('cell_nm', '?')}",
                            'Runner':    nb.get('runner', '?'),
                        }
                        for nb in _all_ds['notebooks']
                    ]
                    st.dataframe(pd.DataFrame(_nb_rows),
                                 use_container_width=True, hide_index=True)

                # Calibration points
                _cal_pts = _odm.load_calibration_db()
                if _cal_pts:
                    st.markdown('**Saved ML calibration points**')
                    st.dataframe(
                        pd.DataFrame(_cal_pts),
                        use_container_width=True, hide_index=True,
                    )
                else:
                    st.caption(
                        'No calibration points saved yet. '
                        'Upload a hysteresis cycle with material and diameter '
                        'specified to generate calibration.'
                    )
            else:
                st.info('OOMMF data unavailable. Upload files to activate.')

        # ── Upload new files ──────────────────────────────────────────────────
        with st.expander('Upload OOMMF / Ubermag data', expanded=False):
            st.markdown(
                'Upload `.txt` files (format `fd  mg`, tab-separated) or '
                '`.ipynb` Jupyter notebooks to expand the dataset. '
                'The system automatically classifies and parses each file.'
            )
            _up_col1, _up_col2 = st.columns([2, 1])
            with _up_col1:
                _uploaded = st.file_uploader(
                    'Select file(s)',
                    type=['txt', 'ipynb'],
                    accept_multiple_files=True,
                    key='oommf_uploader',
                    help='fd = applied field (mT) · mg = measured quantity',
                )
            with _up_col2:
                _up_mat = st.selectbox(
                    'Material (optional)',
                    ['— auto —'] + list(MATERIALS_DB.keys()),
                    format_func=lambda x: (
                        '— infer from file —' if x == '— auto —'
                        else f"{MATERIALS_DB[x]['emoji']} {MATERIALS_DB[x]['name']}"
                    ),
                    key='up_mat',
                )
                _up_d = st.number_input(
                    'Particle diameter (nm)',
                    min_value=1.0, max_value=200.0,
                    value=42.0, step=1.0, key='up_d',
                    help='Used to register the ML calibration point',
                )
                _up_geom = st.selectbox(
                    'Geometry',
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
                                f'**{_fname}** loaded · Type: `{dtype}` · '
                                f'Hc = **{hp["Hc_mT"]:.1f} mT** · '
                                f'Mr = **{hp["Mr_Ms"]:.4f}**'
                            )
                            if _r.get('calibration_saved'):
                                st.info(
                                    f'Calibration point saved: '
                                    f'{_up_d:.0f} nm · {_up_mat_id or "auto"} · '
                                    f'{_up_geom}'
                                )
                        else:
                            st.success(f'{_fname} · Type: `{dtype}` · {msg}')
                    else:
                        st.error(f'{_fname}: {_r.get("message","Error")}')

        # ── ML Prediction with OOMMF calibration ─────────────────────────────
        with st.expander(
            'ML Prediction with real-data calibration',
            expanded=False,
        ):
            st.markdown(
                'Combines the ML ensemble with real OOMMF calibration points '
                'using Gaussian interpolation in the parameter space.'
            )
            _pcal_mat = st.selectbox(
                'Material',
                list(MATERIALS_DB.keys()),
                format_func=lambda m: (
                    f"{MATERIALS_DB[m]['emoji']} {MATERIALS_DB[m]['name']}"
                ),
                key='pcal_mat',
            )
            _pcal_d   = st.slider('Diameter (nm)', 5, 100, 42, key='pcal_d')
            _pcal_g   = st.selectbox(
                'Geometry', list(GEOMETRY_MODES.keys()),
                format_func=lambda g: f"{GEOMETRY_MODES[g]['emoji']}  {GEOMETRY_MODES[g]['name']}",
                key='pcal_geom',
            )

            _Hc_cal, _Mr_cal, _cal_on = MODELS.predict_with_calibration(
                d_nm   = float(_pcal_d),
                mat_id = _pcal_mat,
                geom_id = _pcal_g,
                geom_factor_hc = GEOMETRY_MODES[_pcal_g]['factor_hc'],
                geom_factor_mr = GEOMETRY_MODES[_pcal_g]['factor_mr'],
                T = T_K,
            )
            _Hc_base, _Mr_base = MODELS.predict_fast(
                float(_pcal_d), _pcal_mat,
                GEOMETRY_MODES[_pcal_g]['factor_hc'],
                GEOMETRY_MODES[_pcal_g]['factor_mr'],
                T=T_K,
            )
            _pc1, _pc2, _pc3, _pc4 = st.columns(4)
            _pc1.metric('Hc ML base (mT)',         f'{_Hc_base:.1f}')
            _pc2.metric('Mr ML base',               f'{_Mr_base:.3f}')
            _pc3.metric(
                'Hc calibrated (mT)', f'{_Hc_cal:.1f}',
                delta=f'{_Hc_cal-_Hc_base:+.1f}' if _cal_on else None,
            )
            _pc4.metric(
                'Mr calibrated', f'{_Mr_cal:.3f}',
                delta=f'{_Mr_cal-_Mr_base:+.3f}' if _cal_on else None,
            )
            if _cal_on:
                st.success(
                    'OOMMF calibration active — prediction adjusted with '
                    'nearby real data in parameter space.'
                )
            else:
                st.caption(
                    'No calibration active for this (material, diameter, geometry). '
                    'Upload OOMMF hysteresis data to activate it.'
                )
