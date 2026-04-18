"""
=============================================================================
 micromagnetic_simulator_v2.py — CLI for the Micromagnetic ML Simulator
 Phase 5  ·  GBR + RF + MLP Ensemble  ·  7-feature physics-based engine

 Generates publication-quality figures from the command line using the
 same MicromagneticMLEngine powering the Streamlit web app.

 Usage:
   python micromagnetic_simulator_v2.py
   python micromagnetic_simulator_v2.py --material fe --sizes 16 30 44 60 --pred 30
   python micromagnetic_simulator_v2.py --material co  --sizes 5 20 40 --pred 20 --temp 400
   python micromagnetic_simulator_v2.py --sizes 5 10 20 50 --dpi 300 --format svg
   python micromagnetic_simulator_v2.py --list-materials
   python micromagnetic_simulator_v2.py --help

 Requirements (see requirements.txt):
   numpy >= 1.24  |  matplotlib >= 3.7  |  scikit-learn >= 1.3
=============================================================================
"""

from __future__ import annotations

import argparse
import os
import sys
import time
import warnings

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

warnings.filterwarnings('ignore')

# ── Import the shared ML engine and material database ─────────────────────────
try:
    from ml_engine import MicromagneticMLEngine
    from temperature_model import apply_temperature_to_hc_mr, reduced_magnetization
    from materials_db import MATERIALS_DB, GEOMETRY_MODES
    _ENGINE_AVAILABLE = True
except ImportError as _e:
    print(f'[WARN] Could not import required modules: {_e}')
    print('       Make sure ml_engine.py, temperature_model.py and materials_db.py are in the same directory.')
    _ENGINE_AVAILABLE = False
    sys.exit(1)

# =============================================================================
#  DARK THEME  (consistent with Streamlit web app)
# =============================================================================

_DARK = {
    'figure.facecolor': '#0d1117',
    'axes.facecolor':   '#161b22',
    'axes.edgecolor':   '#30363d',
    'axes.labelcolor':  '#e6edf3',
    'xtick.color':      '#8b949e',
    'ytick.color':      '#8b949e',
    'text.color':       '#e6edf3',
    'grid.color':       '#30363d',
    'grid.alpha':       0.45,
    'font.family':      'sans-serif',
    'font.size':        9,
    'legend.facecolor': '#1c2433',
    'legend.edgecolor': '#30363d',
}

# Per-geometry color palette
_GEOM_COLORS = ['#38bdf8','#fb923c','#34d399','#f472b6',
                '#fbbf24','#a78bfa','#6ee7b7','#f87171']

# =============================================================================
#  PHYSICS HELPERS
# =============================================================================

def llg_hysteresis(
    Hc: float,
    Mr: float,
    H_max: float,
    noise_level: float = 0.008,
    n_pts: int = 300,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Simplified LLG hysteresis loop (tanh model + Gaussian noise)."""
    rng = np.random.default_rng(seed)
    H   = np.linspace(-H_max, H_max, n_pts)
    eps = rng.normal(0, noise_level, n_pts)
    M_up = Mr * np.tanh((H + Hc) / max(Hc, 1e-6)) + eps
    M_dn = Mr * np.tanh((H - Hc) / max(Hc, 1e-6)) - eps
    H_full  = np.concatenate([H, H[::-1]])
    M_full  = np.concatenate([M_up, M_dn[::-1]])
    return H_full, M_full, H, M_up, M_dn


def energy_landscape(Hc: float, H_max: float) -> dict:
    """Compute normalized energy contributions as a function of applied field."""
    H = np.linspace(-H_max, H_max, 300)
    norm = max(abs(Hc), 1.0)
    return {
        'H':        H,
        'zeeman':   -H / norm,
        'exchange':  0.35 * np.exp(-0.5 * (H / norm) ** 2),
        'demag':     0.20 * (1 - (H / H_max) ** 2),
        'aniso':     np.cos(np.pi * H / (2 * H_max)) ** 2,
    }


def is_extrapolation(d_nm: float, mat_id: str) -> bool:
    lo, hi = MATERIALS_DB[mat_id]['range']
    return d_nm < lo or d_nm > hi

# =============================================================================
#  BUILD FIGURE  (GridSpec 2×2)
# =============================================================================

def build_figure(
    mat_id: str,
    engine: MicromagneticMLEngine,
    sizes: list[float],
    pred: float,
    geom_id: str = 'sphere',
    T_K: float = 300.0,
    noise: float = 0.008,
    dpi: int = 170,
    fmt: str = 'png',
    output_dir: str = 'outputs',
) -> str:
    """
    Build the 2×2 GridSpec figure and save it to disk.

    Returns the output file path.
    """
    mat    = MATERIALS_DB[mat_id]
    gm     = GEOMETRY_MODES[geom_id]
    H_max  = mat['field_max']
    colors_sizes = _GEOM_COLORS[:len(sizes)]

    plt.rcParams.update(_DARK)
    fig = plt.figure(figsize=(18, 10), facecolor=_DARK['figure.facecolor'])
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            left=0.07, right=0.97, bottom=0.08, top=0.90,
                            wspace=0.30, hspace=0.42)

    ax_hyst  = fig.add_subplot(gs[0, 0])
    ax_enrg  = fig.add_subplot(gs[0, 1])
    ax_sweep = fig.add_subplot(gs[1, 0])
    ax_table = fig.add_subplot(gs[1, 1])

    # ── Panel 1: Hysteresis loops for each size ───────────────────────────────
    for sz, col in zip(sizes, colors_sizes):
        Hc, Mr, sHc, sMr = engine.predict(
            sz, mat_id,
            geom_factor_hc=gm['factor_hc'],
            geom_factor_mr=gm['factor_mr'],
            T=T_K,
        )
        # Apply temperature correction
        p = mat['params']
        Hc, Mr, _ = apply_temperature_to_hc_mr(
            Hc_ref_mT=Hc, Mr_ref=Mr,
            d_nm=sz, Ms_MA_m=p['Ms_MA_m'],
            K1_kJ_m3=p['K1_kJ_m3'], Tc_K=p['Tc_K'],
            T_K=T_K,
        )
        _, _, H, M_up, M_dn = llg_hysteresis(Hc, Mr, H_max, noise)
        lbl = f'd = {sz:.0f} nm' + (' ⚠' if is_extrapolation(sz, mat_id) else '')
        ax_hyst.plot(H, M_up, color=col, lw=1.6, label=lbl)
        ax_hyst.plot(H, M_dn, color=col, lw=1.6, ls='--', alpha=0.6)

    ax_hyst.axhline(0, color='#475569', lw=0.7)
    ax_hyst.axvline(0, color='#475569', lw=0.7)
    ax_hyst.set_xlabel('Applied field H (mT)')
    ax_hyst.set_ylabel('Reduced magnetization M / Ms')
    ax_hyst.set_title(f'Hysteresis Loops — {mat["name"]}  ·  {gm["name"]}', fontsize=9)
    ax_hyst.legend(fontsize=7, loc='lower right')
    ax_hyst.grid(True)

    # ── Panel 2: Energy landscape for pred size ───────────────────────────────
    Hc_p, Mr_p, _, _ = engine.predict(pred, mat_id,
                                       geom_factor_hc=gm['factor_hc'],
                                       geom_factor_mr=gm['factor_mr'],
                                       T=T_K)
    Hc_p, Mr_p, _ = apply_temperature_to_hc_mr(
        Hc_ref_mT=Hc_p, Mr_ref=Mr_p,
        d_nm=pred, Ms_MA_m=mat['params']['Ms_MA_m'],
        K1_kJ_m3=mat['params']['K1_kJ_m3'], Tc_K=mat['params']['Tc_K'],
        T_K=T_K,
    )
    en = energy_landscape(Hc_p, H_max)
    en_styles = [
        ('zeeman',   '#38bdf8', '-',   'Zeeman'),
        ('exchange', '#f472b6', '-',   'Exchange'),
        ('demag',    '#fbbf24', '--',  'Demagnetization'),
        ('aniso',    '#34d399', '-.',  'Anisotropy'),
    ]
    for key, col, ls, lbl in en_styles:
        ax_enrg.plot(en['H'], en[key], color=col, lw=2.0, ls=ls, label=lbl)
    ax_enrg.axhline(0, color='#475569', lw=0.7)
    ax_enrg.set_xlabel('Applied field H (mT)')
    ax_enrg.set_ylabel('E / E₀  (a.u.)')
    ax_enrg.set_title(f'Magnetic Energy Landscape  ·  d = {pred:.0f} nm', fontsize=9)
    ax_enrg.legend(loc='upper right', fontsize=7)
    ax_enrg.grid(True)

    # ── Panel 3: Hc vs size sweep (all geometries, Ensemble) ─────────────────
    lo, hi = mat['range']
    sweep_sizes = np.linspace(max(2, lo - 5), hi + 5, 40)
    Hc_base, _ = engine.predict_batch(sweep_sizes, mat_id, T=T_K)
    for (gid, gdata), gcol in zip(GEOMETRY_MODES.items(), _GEOM_COLORS):
        hc_g = Hc_base * gdata['factor_hc']
        ax_sweep.plot(sweep_sizes, hc_g, color=gcol, lw=1.5, label=gdata['name'])
    ax_sweep.axvline(pred, color='#f1f5f9', lw=1.0, ls=':', alpha=0.8)
    ax_sweep.axvspan(lo, hi, alpha=0.07, color='#38bdf8')
    ax_sweep.set_xlabel('Size (nm)')
    ax_sweep.set_ylabel('Hc (mT)')
    ax_sweep.set_title(f'Hc vs Size — {mat["name"]} — all geometries', fontsize=9)
    ax_sweep.legend(fontsize=6, ncol=2)
    ax_sweep.grid(True)

    # ── Panel 4: Comparison table (all geometries at pred size) ──────────────
    ax_table.axis('off')
    ext_flag = ' ⚠' if is_extrapolation(pred, mat_id) else ''
    headers  = ['Geometry', 'Hc (mT)', 'Mr/Ms', 'f_Hc', 'f_Mr']
    rows: list[list[str]] = []
    for gid, gdata in GEOMETRY_MODES.items():
        Hc_g, Mr_g, _, _ = engine.predict(
            pred, mat_id,
            geom_factor_hc=gdata['factor_hc'],
            geom_factor_mr=gdata['factor_mr'],
            T=T_K,
        )
        Hc_g, Mr_g, barrier_g = apply_temperature_to_hc_mr(
            Hc_ref_mT=Hc_g, Mr_ref=Mr_g,
            d_nm=pred, Ms_MA_m=mat['params']['Ms_MA_m'],
            K1_kJ_m3=mat['params']['K1_kJ_m3'], Tc_K=mat['params']['Tc_K'],
            T_K=T_K,
        )
        is_sel   = (gid == geom_id)
        prefix   = '* ' if is_sel else '  '
        spm_flag = '  SPM' if (isinstance(barrier_g, float) and barrier_g < 25) else ''
        rows.append([
            f'{prefix}{gdata["name"]}{ext_flag}{spm_flag}',
            f'{Hc_g:.1f}',
            f'{Mr_g:.3f}',
            f'{gdata["factor_hc"]:.2f}',
            f'{gdata["factor_mr"]:.2f}',
        ])
    tbl = ax_table.table(cellText=rows, colLabels=headers,
                         loc='center', cellLoc='center')
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(7.5)
    tbl.scale(1, 1.7)
    for (row, col), cell in tbl.get_celld().items():
        if row == 0:
            cell.set_facecolor('#0f2744')
            cell.set_text_props(color='#93c5fd', fontweight='bold')
        elif row % 2 == 0:
            cell.set_facecolor('#0d1425')
            cell.set_text_props(color='#cbd5e1')
        else:
            cell.set_facecolor('#111827')
            cell.set_text_props(color='#94a3b8')
        cell.set_edgecolor('#1e3a5f')
        cell.set_linewidth(0.5)
    ax_table.set_title(
        f'Geometry Comparison  ·  d = {pred:.0f} nm  ·  {mat["name"]}  ·  T = {T_K:.0f} K',
        fontsize=9, color='#cbd5e1', pad=6, fontweight='semibold')

    fig.suptitle(
        f'Micromagnetic ML Simulator v5.0  ·  GBR + RF + MLP Ensemble  ·  {mat["name"]}',
        fontsize=10, color='#64748b', y=0.97,
        fontweight='normal', fontstyle='italic')

    # ── Save ──────────────────────────────────────────────────────────────────
    fname   = f'sim_{mat_id}_{geom_id}_{pred:.0f}nm_T{T_K:.0f}K.{fmt}'
    outpath = os.path.join(output_dir, fname)
    fig.savefig(outpath, dpi=dpi, bbox_inches='tight',
                facecolor=fig.get_facecolor())
    plt.close(fig)
    return outpath

# =============================================================================
#  CLI ARGUMENT PARSER
# =============================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog='micromagnetic_simulator_v2.py',
        description=(
            'Micromagnetic ML Simulator CLI — Phase 5\n'
            'Generates hysteresis, energy, and comparison figures\n'
            'using the GBR + RF + MLP ensemble (7 physics-based features).'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            'Examples:\n'
            '  python micromagnetic_simulator_v2.py\n'
            '  python micromagnetic_simulator_v2.py --material fe --sizes 16 30 44 60\n'
            '  python micromagnetic_simulator_v2.py --material co --sizes 5 20 40 --pred 20\n'
            '  python micromagnetic_simulator_v2.py --sizes 5 10 20 50 --dpi 300 --format svg\n'
            '  python micromagnetic_simulator_v2.py --material fe3o4 --temp 500 --pred 15\n'
            '  python micromagnetic_simulator_v2.py --list-materials\n'
        )
    )

    parser.add_argument(
        '--material', '-m',
        type=str,
        choices=list(MATERIALS_DB.keys()),
        default='fe',
        help='Material to simulate (default: fe)',
    )
    parser.add_argument(
        '--sizes', '-s',
        type=float,
        nargs='+',
        default=[16, 30, 44, 60],
        help='Particle sizes [nm] for the hysteresis panel (default: 16 30 44 60)',
    )
    parser.add_argument(
        '--pred', '-p',
        type=float,
        default=None,
        help='Size [nm] for the energy landscape and geometry table. '
             'Defaults to the first value in --sizes.',
    )
    parser.add_argument(
        '--geom', '-g',
        type=str,
        choices=list(GEOMETRY_MODES.keys()),
        default='sphere',
        help='Active geometry for the hysteresis panel (default: sphere)',
    )
    parser.add_argument(
        '--temp', '-T',
        type=float,
        default=300.0,
        help='Simulation temperature [K] with Callen-Callen correction (default: 300)',
    )
    parser.add_argument(
        '--output', '-o',
        type=str,
        default=None,
        help='Output directory for figures (default: ./outputs/ next to this script)',
    )
    parser.add_argument(
        '--dpi',
        type=int,
        default=170,
        help='Export resolution in DPI (default: 170)',
    )
    parser.add_argument(
        '--format', '-f',
        type=str,
        choices=['png', 'svg', 'pdf', 'eps'],
        default='png',
        dest='fmt',
        help='Export format (default: png)',
    )
    parser.add_argument(
        '--noise',
        type=float,
        default=0.008,
        help='Gaussian noise level for the LLG loop (default: 0.008)',
    )
    parser.add_argument(
        '--list-materials',
        action='store_true',
        help='Print available materials and exit.',
    )

    return parser.parse_args()

# =============================================================================
#  ENTRY POINT
# =============================================================================

def main() -> None:
    t_start = time.perf_counter()
    args    = parse_args()

    if args.list_materials:
        print('\nAvailable materials:\n')
        for mid, mdata in MATERIALS_DB.items():
            lo, hi = mdata['range']
            print(f'  {mid:<14} {mdata["name"]:<30} range: {lo}–{hi} nm')
        print()
        return

    # ── Output directory ──────────────────────────────────────────────────────
    if args.output is None:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = os.path.join(script_dir, 'outputs')
    else:
        output_dir = os.path.normpath(args.output)
    os.makedirs(output_dir, exist_ok=True)

    pred = args.pred if args.pred is not None else args.sizes[0]
    mat  = MATERIALS_DB[args.material]
    sep  = '─' * 64

    # ── Header ────────────────────────────────────────────────────────────────
    print(sep)
    print(f'  Micromagnetic ML Simulator  v5.0  —  Phase 5')
    print(sep)
    print(f'  Material     : {mat["name"]}')
    print(f'  Sizes (nm)   : {args.sizes}')
    print(f'  Pred. size   : {pred} nm')
    print(f'  Geometry     : {GEOMETRY_MODES[args.geom]["name"]}')
    print(f'  Temperature  : {args.temp:.0f} K')
    print(f'  ML range     : {mat["range"][0]}–{mat["range"][1]} nm')
    print(f'  Output dir   : {output_dir}')
    print(f'  DPI / Format : {args.dpi} / {args.fmt.upper()}')
    print(f'  LLG noise    : {args.noise}')
    print(sep)

    # ── Train ML engine ───────────────────────────────────────────────────────
    print('  Training GBR + RF + MLP ensemble…')
    t_ml = time.perf_counter()
    engine = MicromagneticMLEngine(MATERIALS_DB, T_sim=args.temp)
    engine.train()
    t_ml = time.perf_counter() - t_ml
    metrics = engine.get_metrics(args.material)
    r2_vals = metrics.get('r2_cv_hc', {})
    r2_str  = '  '.join(f'{nm}={v:.3f}' for nm, v in r2_vals.items())
    print(f'  ✓ Ensemble trained  ({t_ml*1000:.1f} ms)  |  R²(Hc): {r2_str}')

    # ── Check extrapolation ───────────────────────────────────────────────────
    for sz in args.sizes + [pred]:
        if is_extrapolation(sz, args.material):
            print(f'  ⚠ WARNING: {sz:.1f} nm is outside the training range '
                  f'[{mat["range"][0]}–{mat["range"][1]} nm] — extrapolation')

    # ── Quick prediction summary ──────────────────────────────────────────────
    gm   = GEOMETRY_MODES[args.geom]
    Hc_p, Mr_p, sHc_p, _ = engine.predict(
        pred, args.material,
        geom_factor_hc=gm['factor_hc'],
        geom_factor_mr=gm['factor_mr'],
        T=args.temp,
    )
    Hc_p, Mr_p, barrier_p = apply_temperature_to_hc_mr(
        Hc_ref_mT=Hc_p, Mr_ref=Mr_p,
        d_nm=pred, Ms_MA_m=mat['params']['Ms_MA_m'],
        K1_kJ_m3=mat['params']['K1_kJ_m3'], Tc_K=mat['params']['Tc_K'],
        T_K=args.temp,
    )
    spm_note = '  (SPM regime!)' if barrier_p < 25 else ''
    print(f'  Prediction at {pred:.1f} nm / {args.temp:.0f} K  →  '
          f'Hc = {Hc_p:.1f} ± {sHc_p:.1f} mT  |  '
          f'Mr/Ms = {Mr_p:.3f}  |  '
          f'E_b/k_BT = {barrier_p:.1f}{spm_note}')

    # ── Generate figure ───────────────────────────────────────────────────────
    print('  Generating figure…')
    t_fig = time.perf_counter()
    outpath = build_figure(
        mat_id    = args.material,
        engine    = engine,
        sizes     = args.sizes,
        pred      = pred,
        geom_id   = args.geom,
        T_K       = args.temp,
        noise     = args.noise,
        dpi       = args.dpi,
        fmt       = args.fmt,
        output_dir= output_dir,
    )
    t_fig = time.perf_counter() - t_fig

    # ── Summary ───────────────────────────────────────────────────────────────
    t_total = time.perf_counter() - t_start
    print(sep)
    print(f'  ✓ Figure saved    : {outpath}')
    print(f'  ML training time  : {t_ml*1000:.1f} ms')
    print(f'  Figure build time : {t_fig*1000:.0f} ms')
    print(f'  Total time        : {t_total*1000:.0f} ms')
    print(sep)


if __name__ == '__main__':
    main()
