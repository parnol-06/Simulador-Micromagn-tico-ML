"""
report.py — Scientific PDF Report Generator
Micromagnetic ML Simulator · Phase 5

Generates a scientific PDF report containing:
  · Cover page with simulation metadata
  · Physical parameters of the material
  · ML prediction table (all geometries)
  · Embedded simulation figures (PNG)
  · Mathematical foundations section
  · SQLite simulation history table

Dependency: reportlab
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any
from xml.sax.saxutils import escape as _xml_escape

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from reportlab.lib             import colors
from reportlab.lib.enums       import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.lib.pagesizes   import A4
from reportlab.lib.styles      import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units       import cm
from reportlab.platypus        import (
    BaseDocTemplate, Frame, Image, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)
from reportlab.platypus.flowables import HRFlowable

def _esc(value: Any) -> str:
    """Escape a value for safe use inside ReportLab Paragraph (XML context).

    ReportLab interprets XML tags inside Paragraph text. Without escaping,
    user-controlled strings could inject markup (e.g. ``<b>``, ``<br/>``).
    """
    return _xml_escape(str(value))


# ─── Color palette ───────────────────────────────────────────────────────────
C_DARK   = colors.HexColor('#0f172a')
C_PANEL  = colors.HexColor('#1e293b')
C_ACCENT = colors.HexColor('#38bdf8')
C_TEXT   = colors.HexColor('#1e293b')
C_GRAY   = colors.HexColor('#64748b')
C_WHITE  = colors.white
C_WARN   = colors.HexColor('#f59e0b')

# ─── Styles ───────────────────────────────────────────────────────────────────

def _build_styles() -> dict:
    styles = {
        'title': ParagraphStyle(
            'title',
            fontSize=22, leading=28, alignment=TA_CENTER,
            textColor=C_DARK, fontName='Helvetica-Bold', spaceAfter=6,
        ),
        'subtitle': ParagraphStyle(
            'subtitle',
            fontSize=13, leading=18, alignment=TA_CENTER,
            textColor=C_GRAY, fontName='Helvetica', spaceAfter=4,
        ),
        'section': ParagraphStyle(
            'section',
            fontSize=13, leading=18, alignment=TA_LEFT,
            textColor=C_DARK, fontName='Helvetica-Bold',
            spaceBefore=14, spaceAfter=6,
            borderPad=4,
        ),
        'body': ParagraphStyle(
            'body',
            fontSize=10, leading=14, alignment=TA_LEFT,
            textColor=C_TEXT, fontName='Helvetica', spaceAfter=4,
        ),
        'caption': ParagraphStyle(
            'caption',
            fontSize=8, leading=11, alignment=TA_CENTER,
            textColor=C_GRAY, fontName='Helvetica-Oblique', spaceAfter=8,
        ),
        'mono': ParagraphStyle(
            'mono',
            fontSize=9, leading=13, alignment=TA_LEFT,
            textColor=colors.HexColor('#1d4ed8'),
            fontName='Courier', spaceAfter=2,
        ),
        'warn': ParagraphStyle(
            'warn',
            fontSize=9, leading=13, alignment=TA_LEFT,
            textColor=C_WARN, fontName='Helvetica-Bold', spaceAfter=4,
        ),
        'right': ParagraphStyle(
            'right',
            fontSize=8, leading=11, alignment=TA_RIGHT,
            textColor=C_GRAY, fontName='Helvetica',
        ),
    }
    return styles


# ─── Table styles ─────────────────────────────────────────────────────────────

_TABLE_STYLE = TableStyle([
    ('BACKGROUND',  (0, 0), (-1, 0),  C_DARK),
    ('TEXTCOLOR',   (0, 0), (-1, 0),  C_WHITE),
    ('FONTNAME',    (0, 0), (-1, 0),  'Helvetica-Bold'),
    ('FONTSIZE',    (0, 0), (-1, 0),  9),
    ('ALIGN',       (0, 0), (-1, -1), 'CENTER'),
    ('VALIGN',      (0, 0), (-1, -1), 'MIDDLE'),
    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.HexColor('#f8fafc'),
                                           colors.HexColor('#e2e8f0')]),
    ('FONTNAME',    (0, 1), (-1, -1), 'Helvetica'),
    ('FONTSIZE',    (0, 1), (-1, -1), 9),
    ('GRID',        (0, 0), (-1, -1), 0.4, colors.HexColor('#cbd5e1')),
    ('TOPPADDING',  (0, 0), (-1, -1), 5),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
    ('LEFTPADDING', (0, 0), (-1, -1), 8),
    ('RIGHTPADDING', (0, 0), (-1, -1), 8),
])

_PARAMS_STYLE = TableStyle([
    ('BACKGROUND',  (0, 0), (0, -1),  colors.HexColor('#1e293b')),
    ('TEXTCOLOR',   (0, 0), (0, -1),  C_WHITE),
    ('FONTNAME',    (0, 0), (0, -1),  'Helvetica-Bold'),
    ('FONTNAME',    (1, 0), (1, -1),  'Courier'),
    ('FONTSIZE',    (0, 0), (-1, -1), 9),
    ('ALIGN',       (0, 0), (0, -1),  'LEFT'),
    ('ALIGN',       (1, 0), (1, -1),  'RIGHT'),
    ('ROWBACKGROUNDS', (0, 0), (-1, -1), [colors.HexColor('#f8fafc'),
                                           colors.HexColor('#e2e8f0')]),
    ('GRID',        (0, 0), (-1, -1), 0.4, colors.HexColor('#cbd5e1')),
    ('TOPPADDING',  (0, 0), (-1, -1), 4),
    ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ('LEFTPADDING', (0, 0), (-1, -1), 8),
])


# ─── Page header / footer ─────────────────────────────────────────────────────

def _header_footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    # Top band
    canvas.setFillColor(C_DARK)
    canvas.rect(0, h - 1.2 * cm, w, 1.2 * cm, fill=1, stroke=0)
    canvas.setFillColor(C_ACCENT)
    canvas.rect(0, h - 1.2 * cm, 0.5 * cm, 1.2 * cm, fill=1, stroke=0)
    canvas.setFillColor(C_WHITE)
    canvas.setFont('Helvetica-Bold', 9)
    canvas.drawString(1.0 * cm, h - 0.78 * cm,
                      'Micromagnetic ML Simulator  ·  Phase 5  ·  Simulation Report')
    canvas.setFont('Helvetica', 8)
    canvas.drawRightString(w - 1.0 * cm, h - 0.78 * cm,
                           datetime.now().strftime('%d/%m/%Y %H:%M'))
    # Footer
    canvas.setFillColor(colors.HexColor('#e2e8f0'))
    canvas.rect(0, 0, w, 0.8 * cm, fill=1, stroke=0)
    canvas.setFillColor(C_GRAY)
    canvas.setFont('Helvetica', 7.5)
    canvas.drawString(1.0 * cm, 0.28 * cm,
                      'Based on data from Galvis, Mesa et al. — '
                      'Results in Physics (2025) & Comp. Mat. Sci. (2024)')
    canvas.drawRightString(w - 1.0 * cm, 0.28 * cm,
                           f'Page {doc.page}')
    canvas.restoreState()


# ─── Matplotlib figure helper ─────────────────────────────────────────────────

def _fig_to_image_flowable(fig: plt.Figure, width: float, height: float) -> Image:
    """Convert a matplotlib figure to a ReportLab Image flowable."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='#0f172a')
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=width, height=height, kind='proportional')


# ─── Main report generator ────────────────────────────────────────────────────

def generate_report(
    mat_id: str,
    mat_name: str,
    d_nm: float,
    preds: dict,
    mat_params: dict,
    mat_range: list,
    field_max: float,
    fig_main: plt.Figure | None,
    fig_energy: plt.Figure | None,
    noise_level: float = 0.008,
    extrapolation: bool = False,
    history_rows: list[dict] | None = None,
    T_K: float = 300.0,
    geom_name: str = '',
) -> bytes:
    """
    Generate the full PDF report and return its bytes.

    Parameters
    ----------
    mat_id       : material key ('fe', 'permalloy', …)
    mat_name     : human-readable material name
    d_nm         : selected particle size [nm]
    preds        : {'sphere': (Hc, Mr), …}  — predictions per geometry
    mat_params   : dict with K1, A, Ms, alpha, lambda_ex, Tc
    mat_range    : [lo, hi] nm training range
    field_max    : maximum applied field [mT]
    fig_main     : main matplotlib figure (GridSpec 2×2) or None
    fig_energy   : energy landscape matplotlib figure or None
    noise_level  : LLG noise level used in the simulation
    extrapolation: True if the selected size is outside the training range
    history_rows : SQLite history rows to include in the report (max 30)
    T_K          : simulation temperature [K]
    geom_name    : active geometry name (for cover metadata)
    """
    buf_pdf = io.BytesIO()
    styles  = _build_styles()
    w, h    = A4
    margin  = 1.8 * cm

    doc = BaseDocTemplate(
        buf_pdf,
        pagesize=A4,
        leftMargin=margin, rightMargin=margin,
        topMargin=1.6 * cm, bottomMargin=1.4 * cm,
    )
    frame = Frame(margin, 1.2 * cm, w - 2 * margin, h - 3.0 * cm, id='main')
    template = PageTemplate(id='main', frames=[frame],
                            onPage=_header_footer)
    doc.addPageTemplates([template])

    story: list = []

    # ── Cover page ────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph('Micromagnetic Simulation Report', styles['title']))
    story.append(Paragraph(
        'ML Engine · GBR + RF + MLP Ensemble · Simplified LLG',
        styles['subtitle']))
    story.append(Spacer(1, 0.4 * cm))
    story.append(HRFlowable(width='100%', thickness=2, color=C_ACCENT,
                             spaceAfter=12))

    # Identification sheet
    cover_data = [
        ['Field', 'Value'],
        ['Material',          _esc(mat_name)],
        ['Material ID',       _esc(mat_id.upper())],
        ['Particle size',     f'{d_nm:.1f} nm'],
        ['Active geometry',   _esc(geom_name) if geom_name else '—'],
        ['Temperature',       f'{T_K:.0f} K'],
        ['ML training range', f'{mat_range[0]}–{mat_range[1]} nm'],
        ['Max field',         f'{field_max:.0f} mT'],
        ['LLG noise level',   f'{noise_level:.3f}'],
        ['Extrapolation',     '⚠ YES — prediction outside range' if extrapolation else 'No'],
        ['Date / Time',       datetime.now().strftime('%d/%m/%Y %H:%M:%S')],
        ['Version',           'v5.0 — Phase 5 (GBR + RF + MLP Ensemble)'],
    ]
    tbl_cover = Table(cover_data, colWidths=[5 * cm, 10.5 * cm])
    tbl_cover.setStyle(_TABLE_STYLE)
    story.append(tbl_cover)
    story.append(Spacer(1, 0.6 * cm))

    if extrapolation:
        story.append(Paragraph(
            f'⚠ WARNING: Size {d_nm:.1f} nm is outside the training range '
            f'[{mat_range[0]}–{mat_range[1]} nm]. '
            'Results are extrapolations and should be interpreted with caution.',
            styles['warn']))

    # ── Main figure ───────────────────────────────────────────────────────────
    story.append(Paragraph(
        '1. Simulation — Hysteresis, Energy Landscape & ML Table',
        styles['section']))

    if fig_main is not None:
        story.append(_fig_to_image_flowable(
            fig_main, width=w - 2 * margin, height=12 * cm))
        story.append(Paragraph(
            f'Figure 1. Full micromagnetic simulation for {_esc(mat_name)} at {d_nm:.0f} nm. '
            '(a) LLG hysteresis loop. '
            '(b) Magnetic energy landscape (Zeeman, Exchange, Demagnetization, Anisotropy). '
            '(c) ML parameter comparison table across all geometries.',
            styles['caption']))

    # Optional energy figure
    if fig_energy is not None:
        story.append(Spacer(1, 0.3 * cm))
        story.append(_fig_to_image_flowable(
            fig_energy, width=w - 2 * margin, height=7 * cm))
        story.append(Paragraph(
            'Figure 2. Interactive energy landscape — individual energy contributions '
            'as a function of applied field H.',
            styles['caption']))

    # ── ML predictions ────────────────────────────────────────────────────────
    story.append(Paragraph('2. ML Model Predictions', styles['section']))
    story.append(Paragraph(
        f'Coercive field Hc (mT) and remanence ratio Mr/Ms predicted by the '
        f'GBR + RF + MLP ensemble for {_esc(mat_name)} at {d_nm:.1f} nm, '
        f'T = {T_K:.0f} K.',
        styles['body']))

    pred_data = [['Geometry', 'Hc (mT)', 'Mr/Ms', 'Status']]
    for geom_key, geom_label in preds.items():
        if isinstance(geom_label, tuple):
            Hc_v, Mr_v = geom_label
        else:
            continue
        pred_data.append([
            _esc(geom_key),
            f'{Hc_v:.2f}',
            f'{Mr_v:.4f}',
            '⚠ Extrapolation' if extrapolation else '✓ Interpolation',
        ])
    tbl_pred = Table(pred_data,
                     colWidths=[4 * cm, 3.5 * cm, 3.5 * cm, 4.5 * cm])
    tbl_pred.setStyle(_TABLE_STYLE)
    story.append(tbl_pred)
    story.append(Spacer(1, 0.4 * cm))

    # ── Physical parameters ───────────────────────────────────────────────────
    story.append(Paragraph('3. Material Physical Parameters', styles['section']))

    param_labels = {
        'K1_kJ_m3':     'Anisotropy constant  K₁ (kJ/m³)',
        'A_pJ_m':       'Exchange constant  A (pJ/m)',
        'Ms_MA_m':      'Saturation magnetization  Ms (MA/m)',
        'alpha':        'Gilbert damping factor  α',
        'lambda_ex_nm': 'Exchange length  λₑₓ (nm)',
        'Tc_K':         'Curie temperature  Tc (K)',
    }
    params_data = [['Parameter', 'Value']] + [
        [label, str(mat_params.get(key, '—'))]
        for key, label in param_labels.items()
    ]
    tbl_params = Table(params_data, colWidths=[10 * cm, 5.5 * cm])
    tbl_params.setStyle(_PARAMS_STYLE)
    story.append(tbl_params)
    story.append(Spacer(1, 0.4 * cm))

    # ── Key equations ─────────────────────────────────────────────────────────
    story.append(Paragraph('4. Mathematical Foundations', styles['section']))
    eqs = [
        ('Landau-Lifshitz-Gilbert equation (LLG):',
         'dM/dt = –γ(M × H_eff) + (α/Ms)[M × dM/dt]'),
        ('Zeeman energy:',
         'E_Z = –μ₀ Ms (M/Ms) · H · V'),
        ('Exchange energy:',
         'E_ex = A ∫|∇m|² dV'),
        ('Uniaxial anisotropy energy:',
         'E_K = K₁ V sin²(θ)'),
        ('Simplified LLG hysteresis (ascending branch):',
         'M(H) = Mr · tanh[(H + Hc) / Hc]  + ε(σ)'),
        ('Callen-Callen thermal scaling:',
         'Hc(T) = Hc(T₀) · [ms(T)/ms(T₀)]^(7/3),   ms(T) = (1 – (T/Tc)^1.5)^(1/3)'),
        ('SPM energy barrier (Néel criterion):',
         'E_b = K₁(T) · V,   K₁(T) = K₁(0) · ms(T)^(10/3)   →   SPM if E_b/k_BT < 25'),
    ]
    for label, eq in eqs:
        story.append(Paragraph(label, styles['body']))
        story.append(Paragraph(eq, styles['mono']))
        story.append(Spacer(1, 0.15 * cm))

    # ── Simulation history ────────────────────────────────────────────────────
    if history_rows:
        story.append(PageBreak())
        story.append(Paragraph('5. Simulation History (SQLite)', styles['section']))
        story.append(Paragraph(
            f'{len(history_rows)} simulation(s) recorded in the SQLite database.',
            styles['body']))

        hist_headers = ['#', 'Time', 'Material', 'Size (nm)',
                        'Hc (mT)', 'Mr/Ms', 'Extrapol.']
        hist_data = [hist_headers]
        for i, row in enumerate(history_rows[:30], 1):
            ts = row.get('timestamp', '')[:16].replace('T', ' ')
            hist_data.append([
                str(i),
                _esc(ts),
                _esc(row.get('material', '—')),
                f"{row.get('size_nm', 0):.1f}",
                f"{row.get('hc_sphere', 0):.1f}" if row.get('hc_sphere') else '—',
                f"{row.get('mr_sphere', 0):.3f}" if row.get('mr_sphere') else '—',
                '⚠' if row.get('extrapolation') else '✓',
            ])

        col_w = [x * cm for x in [0.8, 2.4, 4.0, 2.2, 2.2, 2.2, 1.6]]
        tbl_hist = Table(hist_data, colWidths=col_w, repeatRows=1)
        tbl_hist.setStyle(_TABLE_STYLE)
        story.append(tbl_hist)

    # ── References ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.6 * cm))
    story.append(HRFlowable(width='100%', thickness=1,
                             color=colors.HexColor('#cbd5e1'), spaceAfter=8))
    story.append(Paragraph('References', styles['section']))
    refs = [
        '[1] Galvis, Mesa, et al. "Micromagnetic simulation of Fe nanoparticles." '
        '<i>Results in Physics</i>, 2025.',
        '[2] Galvis, Mesa, Restrepo. "ML-assisted micromagnetic modeling of Permalloy nanodots." '
        '<i>Computational Materials Science</i>, 2024.',
        '[3] Landau, L. D. & Lifshitz, E. M. "On the theory of the dispersion of magnetic '
        'permeability in ferromagnetic bodies." <i>Phys. Z. Sowjetunion</i>, 8:153, 1935.',
        '[4] Gilbert, T. L. "A phenomenological theory of damping in ferromagnetic materials." '
        '<i>IEEE Trans. Magn.</i>, 40(6):3443–3449, 2004.',
        '[5] Callen, H. B. & Callen, E. "The present status of the temperature dependence '
        'of magnetocrystalline anisotropy, and the l(l+1)/2 power law." '
        '<i>J. Phys. Chem. Solids</i>, 27(8):1271–1285, 1966.',
        '[6] Pedregosa et al. "Scikit-learn: Machine learning in Python." '
        '<i>JMLR</i>, 12:2825–2830, 2011.',
    ]
    for ref in refs:
        story.append(Paragraph(ref, styles['body']))
        story.append(Spacer(1, 0.1 * cm))

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(story)
    buf_pdf.seek(0)
    return buf_pdf.read()
