"""
viz3d.py — Visualizaciones 3D interactivas con Plotly
Simulador Micromagnético ML · Fase 3

Funciones:
  · surface_energy_3d     : Superficie E(H, tamaño) — Zeeman + Intercambio
  · magnetization_vectors : Vectores M sobre esfera (Plotly Cone)
  · hc_heatmap            : Mapa de calor Hc(material, tamaño)
  · polar_anisotropy      : Gráfica polar E_K(θ) por material
  · hysteresis_3d_stack   : Stack 3D de lazos de histéresis vs tamaño
"""

from __future__ import annotations
from typing import Callable

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# ── Tema base compartido ─────────────────────────────────────────────────────
_BG       = '#0f172a'
_PANEL    = '#1e293b'
_TEXT     = '#f1f5f9'
_SUBTEXT  = '#94a3b8'
_BORDER   = '#334155'

_LAYOUT_BASE = dict(
    paper_bgcolor=_BG,
    font=dict(color=_TEXT, size=11),
    margin=dict(l=10, r=10, t=50, b=10),
)


# ═══════════════════════════════════════════════════════════════════════════════
#  1 · SUPERFICIE DE ENERGÍA 3D  E(H, tamaño)
# ═══════════════════════════════════════════════════════════════════════════════

def surface_energy_3d(
    mat_id: str,
    models,
    MATERIALS_DB: dict,
    predict_fn: Callable,
    n_sizes: int = 20,
    n_H: int = 80,
) -> go.Figure:
    """
    Superficie 3D: energía total E(H, d) = Zeeman + Intercambio + Desmagnetización.

    Ejes:
      x → Campo H (mT)
      y → Tamaño de partícula d (nm)
      z → Energía normalizada E/E₀
    """
    mat     = MATERIALS_DB[mat_id]
    lo, hi  = mat['range']
    H_max   = mat['field_max']

    sizes = np.linspace(lo, hi, n_sizes)
    H     = np.linspace(-H_max, H_max, n_H)

    Z_zeeman   = np.zeros((n_sizes, n_H))
    Z_exchange = np.zeros((n_sizes, n_H))
    Z_demag    = np.zeros((n_sizes, n_H))
    Z_aniso    = np.zeros((n_sizes, n_H))

    # Predicción vectorizada cuando el engine lo soporta (mucho más rápido)
    if hasattr(models, 'predict_batch'):
        Hc_arr, _ = models.predict_batch(sizes, mat_id)
        Hc_arr    = np.maximum(Hc_arr, 0.5)
    else:
        Hc_arr = np.array([max(predict_fn(s, mat_id, 'sphere', models)[0], 0.5)
                           for s in sizes])

    for i, Hc in enumerate(Hc_arr):
        Z_zeeman[i]   = -H / H_max
        Z_exchange[i] =  0.30 * np.exp(-np.abs(H) / (Hc * 2))
        Z_demag[i]    =  0.10 * (H / H_max) ** 2
        Z_aniso[i]    =  0.20 * np.cos(np.pi * H / H_max) ** 2

    Z_total = Z_zeeman + Z_exchange + Z_demag + Z_aniso

    fig = go.Figure()

    # Superficie total
    fig.add_trace(go.Surface(
        x=H, y=sizes, z=Z_total,
        colorscale='Viridis',
        opacity=0.92,
        colorbar=dict(
            title=dict(text='E/E₀', font=dict(color=_TEXT)),
            tickfont=dict(color=_TEXT),
        ),
        name='E total',
        hovertemplate=(
            'H = %{x:.0f} mT<br>'
            'd = %{y:.1f} nm<br>'
            'E/E₀ = %{z:.3f}<extra></extra>'
        ),
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text=f'Superficie de Energía  E(H, d) — {mat["name"]}',
            font=dict(size=13, color=_TEXT),
        ),
        scene=dict(
            bgcolor=_PANEL,
            xaxis=dict(title='H (mT)',       color=_SUBTEXT, gridcolor=_BORDER),
            yaxis=dict(title='Tamaño (nm)',   color=_SUBTEXT, gridcolor=_BORDER),
            zaxis=dict(title='E / E₀ (u.a.)', color=_SUBTEXT, gridcolor=_BORDER),
            camera=dict(eye=dict(x=1.6, y=-1.6, z=0.9)),
        ),
        height=520,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  2 · VECTORES DE MAGNETIZACIÓN SOBRE ESFERA
# ═══════════════════════════════════════════════════════════════════════════════

def magnetization_vectors(
    mat_id: str,
    d_nm: float,
    models: dict,
    MATERIALS_DB: dict,
    predict_fn: Callable,
    n_theta: int = 14,
    n_phi: int = 14,
) -> go.Figure:
    """
    Muestra vectores de magnetización sobre la superficie de una nanoesfera.
    Color = componente Mz (azul = –Ms, rojo = +Ms).
    """
    mat      = MATERIALS_DB[mat_id]
    Hc, Mr   = predict_fn(d_nm, mat_id, 'sphere', models)

    # Puntos sobre la esfera
    theta = np.linspace(0.1, np.pi - 0.1, n_theta)
    phi   = np.linspace(0, 2 * np.pi,     n_phi,  endpoint=False)
    θ, φ  = np.meshgrid(theta, phi)

    x = np.sin(θ) * np.cos(φ)
    y = np.sin(θ) * np.sin(φ)
    z = np.cos(θ)

    # Campo de magnetización simplificado (estado de remanencia)
    # M ≈ Mr·ẑ perturbado por variación azimutal (muros de dominio simplificados)
    wall_factor = np.exp(-2 * np.abs(np.cos(φ)))          # muro en φ=π/2, 3π/2
    Mx = 0.08 * Mr * np.sin(φ) * wall_factor
    My = 0.08 * Mr * np.cos(φ) * wall_factor
    Mz = Mr   * np.cos(θ) * (1 - 0.15 * wall_factor)

    fig = go.Figure()

    # Superficie de la esfera (semitransparente)
    θ_surf, φ_surf = np.mgrid[0:np.pi:60j, 0:2*np.pi:60j]
    fig.add_trace(go.Surface(
        x=np.sin(θ_surf) * np.cos(φ_surf),
        y=np.sin(θ_surf) * np.sin(φ_surf),
        z=np.cos(θ_surf),
        colorscale=[[0, '#1e293b'], [1, '#334155']],
        opacity=0.18,
        showscale=False,
        hoverinfo='skip',
        name='Sphere',
    ))

    # Conos (vectores M)
    fig.add_trace(go.Cone(
        x=x.flatten(), y=y.flatten(), z=z.flatten(),
        u=Mx.flatten(), v=My.flatten(), w=Mz.flatten(),
        colorscale='RdBu_r',
        cmin=-Mr, cmax=Mr,
        sizemode='absolute',
        sizeref=0.28,
        anchor='tail',
        colorbar=dict(
            title=dict(text='Mz/Ms', font=dict(color=_TEXT)),
            tickfont=dict(color=_TEXT),
        ),
        hovertemplate=(
            'x=%{x:.2f}, y=%{y:.2f}, z=%{z:.2f}<br>'
            'Mz/Ms=%{w:.3f}<extra></extra>'
        ),
        name='M',
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text=(f'Vectores de Magnetización — {mat["name"]}  '
                  f'@  {d_nm:.0f} nm  |  Hc={Hc:.1f} mT  Mr/Ms={Mr:.3f}'),
            font=dict(size=12, color=_TEXT),
        ),
        scene=dict(
            bgcolor=_PANEL,
            xaxis=dict(title='x', color=_SUBTEXT, gridcolor=_BORDER,
                       range=[-1.3, 1.3]),
            yaxis=dict(title='y', color=_SUBTEXT, gridcolor=_BORDER,
                       range=[-1.3, 1.3]),
            zaxis=dict(title='z (eje fácil)', color=_SUBTEXT,
                       gridcolor=_BORDER, range=[-1.3, 1.3]),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.5, y=1.2, z=0.8)),
        ),
        height=540,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  3 · MAPA DE CALOR  Hc(material, tamaño)
# ═══════════════════════════════════════════════════════════════════════════════

def hc_heatmap(
    models,
    MATERIALS_DB: dict,
    predict_fn: Callable,
    geom: str = 'sphere',
    n_sizes: int = 20,
) -> go.Figure:
    """
    Mapa de calor: Hc [mT] como función del material y el tamaño de partícula.
    Las celdas fuera del rango de entrenamiento se muestran como NaN (gris).
    Usa predict_batch cuando está disponible (una pasada por material → rápido).
    """
    sizes_common = np.linspace(5, 120, n_sizes)
    mat_names    = []
    Z            = []
    text_mat     = []

    for mat_id, mat in MATERIALS_DB.items():
        lo, hi = mat['range']
        # Seleccionar solo los tamaños dentro del rango del material
        valid_mask = (sizes_common >= lo) & (sizes_common <= hi)
        valid_sizes = sizes_common[valid_mask]

        # Predicción vectorizada o punto a punto según capacidad del engine
        if hasattr(models, 'predict_batch') and len(valid_sizes) > 0:
            Hc_valid, _ = models.predict_batch(valid_sizes, mat_id)
        else:
            Hc_valid = np.array([
                predict_fn(s, mat_id, geom, models)[0]
                for s in valid_sizes
            ])

        row   = []
        t_row = []
        vi = 0
        for s, in_range in zip(sizes_common, valid_mask):
            if in_range:
                Hc = float(Hc_valid[vi]); vi += 1
                row.append(round(Hc, 1))
                t_row.append(f'{Hc:.0f}')
            else:
                row.append(None)
                t_row.append('—')
        Z.append(row)
        text_mat.append(t_row)
        mat_names.append(mat['name'])

    fig = go.Figure(data=go.Heatmap(
        z=Z,
        x=[f'{s:.0f}' for s in sizes_common],
        y=mat_names,
        colorscale='Plasma',
        colorbar=dict(
            title=dict(text='Hc (mT)', font=dict(color=_TEXT)),
            tickfont=dict(color=_TEXT),
        ),
        text=text_mat,
        texttemplate='%{text}',
        textfont=dict(size=8, color='white'),
        hoverongaps=False,
        hovertemplate='Material: %{y}<br>Tamaño: %{x} nm<br>Hc = %{z:.1f} mT<extra></extra>',
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text=f'Hc Heatmap (mT) — Geometry: {"Sphere" if geom=="sphere" else "Cuboid"}',
            font=dict(size=13, color=_TEXT),
        ),
        xaxis=dict(title='Tamaño (nm)', color=_SUBTEXT,
                   gridcolor=_BORDER, tickangle=-45),
        yaxis=dict(title='Material',    color=_SUBTEXT, gridcolor=_BORDER),
        plot_bgcolor=_PANEL,
        height=360,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  4 · GRÁFICA POLAR  E_K(θ) — Energía de anisotropía
# ═══════════════════════════════════════════════════════════════════════════════

def polar_anisotropy(
    MATERIALS_DB: dict,
    mat_ids: list[str] | None = None,
) -> go.Figure:
    """
    Gráfica polar de la energía de anisotropía magnetocristalina E_K(θ).
    Modelo uniaxial: E_K = K₁·sin²(θ), normalizado a K₁_max entre materiales.
    """
    if mat_ids is None:
        mat_ids = list(MATERIALS_DB.keys())

    theta_deg = np.linspace(0, 360, 361)
    theta_rad = np.radians(theta_deg)

    # Valor máximo para normalización global
    K1_max = max(MATERIALS_DB[m]['params']['K1_kJ_m3'] for m in mat_ids)

    fig = go.Figure()

    for mat_id in mat_ids:
        mat = MATERIALS_DB[mat_id]
        K1  = mat['params']['K1_kJ_m3']
        # E_K uniaxial normalizada globalmente
        E_K = (K1 / max(K1_max, 1e-9)) * np.sin(theta_rad) ** 2
        fig.add_trace(go.Scatterpolar(
            r=E_K,
            theta=theta_deg,
            mode='lines',
            name=f"{mat['emoji']}  {mat['name']}  (K₁={K1} kJ/m³)",
            line=dict(color=mat['color'], width=2.5),
            fill='toself',
            fillcolor=mat['color'],
            opacity=0.10,
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text='Energía de Anisotropía  E_K(θ) — Comparación de Materiales',
            font=dict(size=13, color=_TEXT),
        ),
        polar=dict(
            bgcolor=_PANEL,
            angularaxis=dict(color=_SUBTEXT, linecolor=_BORDER,
                             gridcolor=_BORDER),
            radialaxis=dict(color=_SUBTEXT, linecolor=_BORDER,
                            gridcolor=_BORDER, title='E_K / K₁_max'),
        ),
        legend=dict(bgcolor=_PANEL, bordercolor=_BORDER, borderwidth=1,
                    font=dict(size=10)),
        height=480,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  5 · STACK 3D DE LAZOS DE HISTÉRESIS vs TAMAÑO
# ═══════════════════════════════════════════════════════════════════════════════

def hysteresis_3d_stack(
    mat_id: str,
    models,
    MATERIALS_DB: dict,
    predict_fn: Callable,
    llg_fn: Callable,
    n_sizes: int = 8,
    n_pts: int = 200,
) -> go.Figure:
    """
    Stack 3D de lazos de histéresis:
      x → Campo H (mT)
      y → Tamaño de partícula d (nm)   [profundidad]
      z → Magnetización M/Ms

    Permite ver cómo evoluciona el lazo al variar el tamaño.
    """
    mat    = MATERIALS_DB[mat_id]
    lo, hi = mat['range']
    H_max  = mat['field_max']
    sizes  = np.linspace(lo, hi, n_sizes)

    fig = go.Figure()

    colorscale_vals = np.linspace(0, 1, n_sizes)

    # Pre-calcular Hc y Mr para todos los tamaños en una sola pasada
    if hasattr(models, 'predict_batch'):
        Hc_all, Mr_all = models.predict_batch(sizes, mat_id)
    else:
        _preds = [predict_fn(s, mat_id, 'sphere', models) for s in sizes]
        Hc_all = np.array([p[0] for p in _preds])
        Mr_all = np.array([p[1] for p in _preds])

    for idx, (s, cv) in enumerate(zip(sizes, colorscale_vals)):
        Hc, Mr = float(Hc_all[idx]), float(Mr_all[idx])
        H, M_up, M_dn = llg_fn(Hc, Mr, H_max=H_max, n_pts=n_pts, seed=42)

        # Curva cerrada: up + down invertida
        H_closed   = np.concatenate([H,   H[::-1]])
        M_closed   = np.concatenate([M_up, M_dn[::-1]])
        size_plane = np.full_like(H_closed, s)

        # Color interpolado
        r = int(56  + cv * (251 - 56))
        g = int(189 + cv * (146 - 189))
        b = int(248 + cv * (48  - 248))
        color = f'rgb({r},{g},{b})'

        fig.add_trace(go.Scatter3d(
            x=H_closed, y=size_plane, z=M_closed,
            mode='lines',
            line=dict(color=color, width=2),
            name=f'd = {s:.0f} nm',
            hovertemplate=(
                f'd = {s:.0f} nm<br>'
                'H = %{x:.0f} mT<br>'
                'M/Ms = %{z:.3f}<extra></extra>'
            ),
        ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text=f'Stack 3D de Histéresis — {mat["name"]}  ({lo}–{hi} nm)',
            font=dict(size=13, color=_TEXT),
        ),
        scene=dict(
            bgcolor=_PANEL,
            xaxis=dict(title='H (mT)',        color=_SUBTEXT, gridcolor=_BORDER),
            yaxis=dict(title='Tamaño (nm)',    color=_SUBTEXT, gridcolor=_BORDER),
            zaxis=dict(title='M / Ms',         color=_SUBTEXT, gridcolor=_BORDER),
            camera=dict(eye=dict(x=1.8, y=-1.4, z=0.7)),
        ),
        legend=dict(bgcolor=_PANEL, bordercolor=_BORDER, borderwidth=1,
                    font=dict(size=9), tracegroupgap=2),
        height=540,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  6 · GEOMETRÍA 3D VÓXEL  (inspirado en simulaciones MuMax3)
# ═══════════════════════════════════════════════════════════════════════════════

def voxel_geometry_3d(
    geom_id: str,
    d_nm: float,
    GEOMETRY_MODES: dict,
    n: int = 26,
) -> go.Figure:
    """
    Representación 3D voxelizada de la geometría de nanopartícula seleccionada.
    Cada punto es un 'vóxel' coloreado por la componente mz (estado remanente).

    Inspirado en las visualizaciones de geometría discreta de MuMax3 /
    Galvis, Mesa et al. (Results in Physics, 2025).
    """
    gm = GEOMETRY_MODES[geom_id]
    r  = d_nm / 2.0   # semidiámetro en nm

    span  = r * 1.15
    coord = np.linspace(-span, span, n)
    X, Y, Z = np.meshgrid(coord, coord, coord)

    # ── Máscara de geometría ─────────────────────────────────────────────────
    if geom_id == 'sphere':
        mask = X**2 + Y**2 + Z**2 <= r**2

    elif geom_id == 'cuboid':
        mask = (np.abs(X) <= r) & (np.abs(Y) <= r * 0.80) & (np.abs(Z) <= r * 0.55)

    elif geom_id == 'cylinder_disk':
        mask = (X**2 + Y**2 <= r**2) & (np.abs(Z) <= r * 0.32)

    elif geom_id == 'cylinder_rod':
        mask = (X**2 + Y**2 <= (r * 0.38)**2) & (np.abs(Z) <= r)

    elif geom_id == 'ellipsoid_prolate':
        a2 = (r * 0.62)**2
        mask = X**2 / a2 + Y**2 / a2 + Z**2 / r**2 <= 1

    elif geom_id == 'ellipsoid_oblate':
        c2 = (r * 0.38)**2
        mask = X**2 / r**2 + Y**2 / r**2 + Z**2 / c2 <= 1

    elif geom_id == 'torus':
        R_maj = r * 0.60
        r_tub = r * 0.32
        mask = (np.sqrt(X**2 + Y**2) - R_maj)**2 + Z**2 <= r_tub**2

    elif geom_id == 'core_shell':
        r_out   = r
        r_inner = r * 0.55
        mask    = X**2 + Y**2 + Z**2 <= r_out**2

    else:
        mask = X**2 + Y**2 + Z**2 <= r**2

    xs = X[mask].flatten()
    ys = Y[mask].flatten()
    zs = Z[mask].flatten()

    # ── Coloración por mz (estado remanente) ────────────────────────────────
    mz = zs / (r + 1e-9)    # normalizado –1 … +1

    if geom_id == 'core_shell':
        r_inner = r * 0.55
        in_core = xs**2 + ys**2 + zs**2 <= r_inner**2
        # core: brillo pleno · cáscara: atenuada
        mz = np.where(in_core, np.sign(zs + 1e-9) * 0.90, mz * 0.35)

    marker_size = max(2.2, 105.0 / n)

    fig = go.Figure(data=go.Scatter3d(
        x=xs, y=ys, z=zs,
        mode='markers',
        marker=dict(
            size=marker_size,
            color=mz,
            colorscale='RdBu_r',
            cmin=-1, cmax=1,
            colorbar=dict(
                title=dict(text='mz / Ms', font=dict(color=_TEXT, size=11)),
                tickfont=dict(color=_TEXT),
                len=0.72,
            ),
            opacity=0.90,
            line=dict(width=0),
        ),
        hovertemplate=(
            'x=%{x:.1f} nm<br>y=%{y:.1f} nm<br>z=%{z:.1f} nm<extra></extra>'
        ),
        name=gm['name'],
    ))

    fig.update_layout(
        **_LAYOUT_BASE,
        title=dict(
            text=(f'{gm["name"]}  —  d = {d_nm:.0f} nm'
                  '<br><sup>Voxels colored by mz (remanent state)</sup>'),
            font=dict(size=13, color=_TEXT),
        ),
        scene=dict(
            bgcolor=_PANEL,
            xaxis=dict(title='x (nm)', color=_SUBTEXT, gridcolor=_BORDER),
            yaxis=dict(title='y (nm)', color=_SUBTEXT, gridcolor=_BORDER),
            zaxis=dict(title='z (nm)', color=_SUBTEXT, gridcolor=_BORDER),
            aspectmode='cube',
            camera=dict(eye=dict(x=1.55, y=-1.40, z=1.05)),
        ),
        height=540,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  7 · COMPONENTES DE ENERGÍA — 4 paneles  (inspirado en Fig. 3 del artículo)
# ═══════════════════════════════════════════════════════════════════════════════

def energy_components_4panel(
    mat_id: str,
    models,
    MATERIALS_DB: dict,
    predict_fn: Callable,
    n_sizes: int = 6,
    n_H: int = 200,
) -> go.Figure:
    """
    2×2 subplots con las cuatro componentes de energía vs H para n_sizes tamaños.

    Paneles:
      TL → Anisotropía efectiva  E_K(H)
      TR → Desmagnetización      E_D(H)
      BL → Intercambio           E_ex(H)
      BR → Zeeman                E_Z(H)

    Las magnitudes (en Joules) se estiman con los parámetros físicos del material
    y escalan con el volumen de la partícula V ∝ d³.
    Inspirado en Fig. 3 de Galvis, Mesa et al. (Results in Physics, 2025).
    """
    mat    = MATERIALS_DB[mat_id]
    lo, hi = mat['range']
    H_max  = mat['field_max']
    p      = mat['params']

    mu0 = 4.0 * np.pi * 1e-7
    K1  = abs(p['K1_kJ_m3']) * 1e3     # J/m³
    Ms  = p['Ms_MA_m']  * 1e6          # A/m
    A   = p['A_pJ_m']   * 1e-12        # J/m
    lam = p['lambda_ex_nm'] * 1e-9     # m (exchange length)

    H_mT  = np.linspace(-H_max, H_max, n_H)
    sizes = np.linspace(lo, hi, n_sizes)

    # Pre-calcular Hc para todos los tamaños en una sola pasada (batch)
    if hasattr(models, 'predict_batch'):
        Hc_batch, _ = models.predict_batch(sizes, mat_id)
        Hc_batch    = np.maximum(Hc_batch, 1.0)
    else:
        Hc_batch = np.array([
            max(predict_fn(s, mat_id, 'sphere', models)[0], 1.0)
            for s in sizes
        ])

    # Colormap azul → naranja interpolado
    def _interp_color(i, n):
        t = i / max(n - 1, 1)
        r = int(56  + t * (251 - 56))
        g = int(189 + t * (146 - 189))
        b = int(248 + t * (48  - 248))
        return f'rgb({r},{g},{b})'

    panel_titles = [
        'Energía de Anisotropía Efectiva (J)',
        'Energía de Desmagnetización (J)',
        'Energía de Intercambio (J)',
        'Energía de Zeeman (J)',
    ]
    positions = [(1, 1), (1, 2), (2, 1), (2, 2)]

    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=panel_titles,
        horizontal_spacing=0.14,
        vertical_spacing=0.20,
    )

    for i, s_nm in enumerate(sizes):
        d_m = s_nm * 1e-9
        V   = (4.0 / 3.0) * np.pi * (d_m / 2.0) ** 3   # m³

        Hc_mT = float(Hc_batch[i])

        # Magnetización normalizada (promedio de ramas up/down)
        m = np.tanh(H_mT / Hc_mT)

        # ── Energías (Joules) ────────────────────────────────────────────────
        E_aniso  = K1 * V * (1.0 - m ** 2)
        Nd       = 1.0 / 3.0                             # factor desmagnetizante esfera
        E_demag  = 0.5 * mu0 * Nd * (Ms * m) ** 2 * V
        E_exch   = A * V / (lam ** 2) * (1.0 - np.abs(m))
        H_SI     = H_mT * 1e-3 / mu0                    # mT → A/m
        E_zeeman = np.abs(mu0 * Ms * m * V * H_SI)

        color  = _interp_color(i, n_sizes)
        label  = f'{s_nm:.0f} nm'

        for (r_pos, c_pos), E in zip(positions,
                                     [E_aniso, E_demag, E_exch, E_zeeman]):
            fig.add_trace(
                go.Scatter(
                    x=H_mT, y=E,
                    mode='lines',
                    line=dict(color=color, width=1.8),
                    name=label,
                    legendgroup=label,
                    showlegend=(r_pos == 1 and c_pos == 1),
                    hovertemplate=(
                        f'd = {s_nm:.0f} nm<br>'
                        'H = %{x:.0f} mT<br>'
                        'E = %{y:.3e} J<extra></extra>'
                    ),
                ),
                row=r_pos, col=c_pos,
            )

    # ── Ejes ─────────────────────────────────────────────────────────────────
    for r_pos, c_pos in positions:
        fig.update_xaxes(
            title_text='H (mT)', color=_SUBTEXT,
            gridcolor=_BORDER, zerolinecolor=_BORDER,
            showgrid=True,
            row=r_pos, col=c_pos,
        )
        fig.update_yaxes(
            title_text='Energía (J)', color=_SUBTEXT,
            gridcolor=_BORDER, zerolinecolor=_BORDER,
            exponentformat='e', showexponent='all',
            showgrid=True,
            row=r_pos, col=c_pos,
        )

    # Títulos de subtítulos en color claro
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(color=_SUBTEXT, size=11)

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_PANEL,
        font=dict(color=_TEXT, size=11),
        margin=dict(l=60, r=20, t=90, b=50),
        title=dict(
            text=(f'Componentes de Energía vs H — {mat["name"]}'
                  '<br><sup>Anisotropía · Desmagnetización · Intercambio · Zeeman</sup>'),
            font=dict(size=13, color=_TEXT),
        ),
        legend=dict(
            bgcolor=_PANEL, bordercolor=_BORDER, borderwidth=1,
            font=dict(size=10, color=_TEXT),
            title=dict(text='Tamaño', font=dict(color=_SUBTEXT, size=10)),
        ),
        height=640,
    )
    return fig


# ═══════════════════════════════════════════════════════════════════════════════
#  8 · MAPA 2D DE MAGNETIZACIÓN — 4 estados magnéticos  (Fig. 4 del artículo)
# ═══════════════════════════════════════════════════════════════════════════════

def magnetization_map_2d(
    mat_id: str,
    d_nm: float,
    models: dict,
    MATERIALS_DB: dict,
    predict_fn: Callable,
    n_grid: int = 42,
) -> go.Figure:
    """
    4 paneles con el mapa 2D de la componente mz (corte transversal XY) en
    cuatro estados magnéticos distintos, con vectores de magnetización superpuestos.

    Estados:
      (a) Saturación  +H_max  → mz ≈ +1, uniforme
      (b) Remanencia  H = 0   → mz ≈ Mr, leve vórtice en bordes
      (c) Conmutación H ≈ −Hc → pared de dominio central
      (d) Saturación  −H_max  → mz ≈ −1, uniforme

    Inspirado en simulaciones MuMax3 de la Fig. 4 de Galvis, Mesa et al. (2025).
    """
    mat    = MATERIALS_DB[mat_id]
    H_max  = mat['field_max']
    Hc, Mr = predict_fn(d_nm, mat_id, 'sphere', models)
    r      = d_nm / 2.0   # nm

    xy        = np.linspace(-r, r, n_grid)
    X, Y      = np.meshgrid(xy, xy)
    dist_sq   = X ** 2 + Y ** 2
    inside    = dist_sq <= r ** 2

    # ── Patrones de mz ───────────────────────────────────────────────────────
    def _build_mz(pattern: str, mz_mean: float) -> np.ndarray:
        mz = np.full((n_grid, n_grid), np.nan)
        if pattern == 'uniform':
            mz[inside] = mz_mean
        elif pattern == 'remanence':
            # Leve curvatura tipo burbuja en los bordes ecuatoriales
            norm_r = np.sqrt(dist_sq) / r
            mz[inside] = mz_mean * (
                1.0 - 0.18 * norm_r[inside] ** 3
            )
        elif pattern == 'switching':
            # Pared de dominio circular en r ≈ 0.45·R
            wall_r = r * 0.45
            dist   = np.sqrt(dist_sq)
            signed = (dist - wall_r) / (r * 0.12)
            mz[inside] = np.tanh(signed[inside]) * mz_mean if mz_mean != 0 else \
                         np.tanh(signed[inside]) * Mr
        return mz

    def _build_arrows(pattern: str, mz_mean: float):
        """Devuelve vectores planos (Mx, My) para flechas superpuestas."""
        Mx = np.zeros((n_grid, n_grid))
        My = np.zeros((n_grid, n_grid))
        if pattern == 'uniform':
            My[inside] = 0.06 * np.sign(mz_mean if mz_mean != 0 else 1)
        elif pattern == 'remanence':
            # Ligera curvatura antihoraria
            Mx[inside] = -0.07 * Y[inside] / r
            My[inside] =  0.07 * X[inside] / r
        elif pattern == 'switching':
            # Flechas radiales cerca de la pared de dominio
            dist = np.sqrt(dist_sq)
            Mx[inside] = 0.14 * X[inside] / (dist[inside] + 1e-9)
            My[inside] = 0.14 * Y[inside] / (dist[inside] + 1e-9)
        return Mx, My

    states = [
        ('uniform',   Mr,    f'(a) Saturación  +{H_max:.0f} mT'),
        ('remanence', Mr,    '(b) Remanencia  H = 0'),
        ('switching', 0.0,   f'(c) Conmutación  −{Hc:.0f} mT'),
        ('uniform',  -Mr,    f'(d) Saturación  −{H_max:.0f} mT'),
    ]

    fig = make_subplots(
        rows=1, cols=4,
        subplot_titles=[s[2] for s in states],
        horizontal_spacing=0.035,
    )

    stride = max(1, n_grid // 13)

    for col_i, (pattern, mz_val, _title) in enumerate(states, start=1):
        mz  = _build_mz(pattern, mz_val)
        Mx, My = _build_arrows(pattern, mz_val)

        # ── Heatmap mz ───────────────────────────────────────────────────────
        fig.add_trace(
            go.Heatmap(
                z=mz, x=xy, y=xy,
                colorscale='RdBu_r',
                zmin=-1, zmax=1,
                showscale=(col_i == 4),
                colorbar=dict(
                    title=dict(text='mz', font=dict(color=_TEXT, size=12)),
                    tickfont=dict(color=_TEXT),
                    len=0.82, x=1.01,
                ) if col_i == 4 else {},
                hovertemplate=(
                    'x=%{x:.1f} nm<br>y=%{y:.1f} nm<br>'
                    'mz=%{z:.3f}<extra></extra>'
                ),
                name=_title,
            ),
            row=1, col=col_i,
        )

        # ── Vectores (flechas tipo quiver) ───────────────────────────────────
        xs_a, ys_a = [], []
        scale = r * 0.20
        for iy in range(0, n_grid, stride):
            for ix in range(0, n_grid, stride):
                if not inside[iy, ix]:
                    continue
                x0, y0 = X[iy, ix], Y[iy, ix]
                dx_a   = Mx[iy, ix] * scale
                dy_a   = My[iy, ix] * scale
                xs_a  += [x0, x0 + dx_a, None]
                ys_a  += [y0, y0 + dy_a, None]

        fig.add_trace(
            go.Scatter(
                x=xs_a, y=ys_a,
                mode='lines',
                line=dict(color='black', width=0.9),
                showlegend=False,
                hoverinfo='skip',
            ),
            row=1, col=col_i,
        )

        # ── Contorno de la partícula ──────────────────────────────────────────
        theta_c = np.linspace(0, 2 * np.pi, 200)
        fig.add_trace(
            go.Scatter(
                x=r * np.cos(theta_c),
                y=r * np.sin(theta_c),
                mode='lines',
                line=dict(color='white', width=1.4, dash='dot'),
                showlegend=False,
                hoverinfo='skip',
            ),
            row=1, col=col_i,
        )

        # ── Ejes ─────────────────────────────────────────────────────────────
        fig.update_xaxes(
            title_text='x (nm)', color=_SUBTEXT,
            gridcolor=_BORDER,
            range=[-r * 1.12, r * 1.12],
            row=1, col=col_i,
        )
        fig.update_yaxes(
            title_text='y (nm)' if col_i == 1 else '',
            color=_SUBTEXT, gridcolor=_BORDER,
            range=[-r * 1.12, r * 1.12],
            scaleanchor=f'x{col_i}' if col_i > 1 else 'x',
            scaleratio=1,
            row=1, col=col_i,
        )

    # Títulos en color claro
    for ann in fig['layout']['annotations']:
        ann['font'] = dict(color=_SUBTEXT, size=10)

    fig.update_layout(
        paper_bgcolor=_BG,
        plot_bgcolor=_PANEL,
        font=dict(color=_TEXT, size=11),
        margin=dict(l=55, r=70, t=80, b=50),
        title=dict(
            text=(f'Mapa 2D de Magnetización — {mat["name"]}  @  {d_nm:.0f} nm'
                  '<br><sup>Corte XY · mz coloreado · vectores superpuestos</sup>'),
            font=dict(size=13, color=_TEXT),
        ),
        height=430,
    )
    return fig
