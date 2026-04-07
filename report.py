"""
report.py — Generador de Reportes PDF
Simulador Micromagnético ML · Fase 3

Genera un reporte científico en PDF con:
  · Portada con metadata
  · Parámetros físicos del material
  · Tabla de predicciones ML (esfera vs cuboide)
  · Figura de simulación embebida (PNG)
  · Paisaje de energía
  · Sección de referencias

Dependencia: reportlab
"""

from __future__ import annotations

import io
from datetime import datetime
from typing import Any

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

# ─── Paleta de colores ────────────────────────────────────────────────────────
C_DARK   = colors.HexColor('#0f172a')
C_PANEL  = colors.HexColor('#1e293b')
C_ACCENT = colors.HexColor('#38bdf8')
C_TEXT   = colors.HexColor('#1e293b')
C_GRAY   = colors.HexColor('#64748b')
C_WHITE  = colors.white
C_WARN   = colors.HexColor('#f59e0b')

# ─── Estilos ─────────────────────────────────────────────────────────────────

def _build_styles() -> dict:
    base = getSampleStyleSheet()
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


# ─── Estilos de tabla ─────────────────────────────────────────────────────────

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


# ─── Cabecera / pie de página ─────────────────────────────────────────────────

def _header_footer(canvas, doc):
    canvas.saveState()
    w, h = A4
    # Banda superior
    canvas.setFillColor(C_DARK)
    canvas.rect(0, h - 1.2 * cm, w, 1.2 * cm, fill=1, stroke=0)
    canvas.setFillColor(C_ACCENT)
    canvas.rect(0, h - 1.2 * cm, 0.5 * cm, 1.2 * cm, fill=1, stroke=0)
    canvas.setFillColor(C_WHITE)
    canvas.setFont('Helvetica-Bold', 9)
    canvas.drawString(1.0 * cm, h - 0.78 * cm,
                      'Simulador Micromagnético ML  ·  Fase 3  ·  Reporte de Simulación')
    canvas.setFont('Helvetica', 8)
    canvas.drawRightString(w - 1.0 * cm, h - 0.78 * cm,
                           datetime.now().strftime('%d/%m/%Y %H:%M'))
    # Pie
    canvas.setFillColor(colors.HexColor('#e2e8f0'))
    canvas.rect(0, 0, w, 0.8 * cm, fill=1, stroke=0)
    canvas.setFillColor(C_GRAY)
    canvas.setFont('Helvetica', 7.5)
    canvas.drawString(1.0 * cm, 0.28 * cm,
                      'Basado en datos de Galvis, Mesa et al. — '
                      'Results in Physics (2025) & Comp. Mat. Sci. (2024)')
    canvas.drawRightString(w - 1.0 * cm, 0.28 * cm,
                           f'Página {doc.page}')
    canvas.restoreState()


# ─── Constructor de figura matplotlib ────────────────────────────────────────

def _fig_to_image_flowable(fig: plt.Figure, width: float, height: float) -> Image:
    """Convierte una figura matplotlib en un flowable de ReportLab."""
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=150, bbox_inches='tight',
                facecolor='#0f172a')
    buf.seek(0)
    plt.close(fig)
    return Image(buf, width=width, height=height, kind='proportional')


# ─── Generador principal ─────────────────────────────────────────────────────

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
) -> bytes:
    """
    Genera el reporte PDF completo y retorna los bytes del documento.

    Parámetros
    ----------
    mat_id       : clave del material ('fe', 'permalloy', …)
    mat_name     : nombre legible del material
    d_nm         : tamaño de partícula seleccionado
    preds        : {'sphere': (Hc, Mr), 'cuboid': (Hc, Mr)}
    mat_params   : dict con K1, A, Ms, alpha, lambda_ex, Tc
    mat_range    : [lo, hi] nm rango de entrenamiento
    field_max    : campo máximo (mT)
    fig_main     : figura matplotlib principal (GridSpec 2×2)
    fig_energy   : figura matplotlib de energía (opcional)
    noise_level  : nivel de ruido LLG usado
    extrapolation: True si el tamaño está fuera del rango
    history_rows : filas del historial SQLite para incluir en el reporte
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

    # ── Portada ───────────────────────────────────────────────────────────────
    story.append(Spacer(1, 1.5 * cm))
    story.append(Paragraph('Reporte de Simulación Micromagnética', styles['title']))
    story.append(Paragraph('Motor ML · GradientBoostingRegressor · LLG Simplificado',
                            styles['subtitle']))
    story.append(Spacer(1, 0.4 * cm))
    story.append(HRFlowable(width='100%', thickness=2, color=C_ACCENT,
                             spaceAfter=12))

    # Ficha de identificación
    ficha_data = [
        ['Campo', 'Valor'],
        ['Material',         mat_name],
        ['ID Material',      mat_id.upper()],
        ['Tamaño analizado', f'{d_nm:.1f} nm'],
        ['Rango ML',         f'{mat_range[0]}–{mat_range[1]} nm'],
        ['Campo máximo',     f'{field_max:.0f} mT'],
        ['Ruido LLG',        f'{noise_level:.3f}'],
        ['Extrapolación',    '⚠ SÍ — predicción fuera de rango' if extrapolation else 'No'],
        ['Fecha / Hora',     datetime.now().strftime('%d/%m/%Y %H:%M:%S')],
        ['Versión',          'v2.0 — Fase 3'],
    ]
    tbl_ficha = Table(ficha_data, colWidths=[5 * cm, 10.5 * cm])
    tbl_ficha.setStyle(_TABLE_STYLE)
    story.append(tbl_ficha)
    story.append(Spacer(1, 0.6 * cm))

    if extrapolation:
        story.append(Paragraph(
            f'⚠ ADVERTENCIA: El tamaño {d_nm:.1f} nm está fuera del rango '
            f'de entrenamiento [{mat_range[0]}–{mat_range[1]} nm]. '
            'Los resultados son una extrapolación y deben interpretarse con precaución.',
            styles['warn']))

    # ── Figura principal ──────────────────────────────────────────────────────
    story.append(Paragraph('1. Simulación — Histéresis, Energía y Tabla ML',
                            styles['section']))

    if fig_main is not None:
        story.append(_fig_to_image_flowable(fig_main,
                     width=w - 2 * margin, height=12 * cm))
        story.append(Paragraph(
            f'Figura 1. Simulación micromagnética completa para {mat_name} '
            f'a {d_nm:.0f} nm. (a) Lazo de histéresis LLG para esfera y cuboide. '
            '(b) Paisaje de energía (Zeeman, Intercambio, Desmagnetización, Anisotropía). '
            '(c) Tabla comparativa de parámetros ML.',
            styles['caption']))

    # ── Predicciones ML ───────────────────────────────────────────────────────
    story.append(Paragraph('2. Predicciones del Modelo ML', styles['section']))
    story.append(Paragraph(
        f'La siguiente tabla muestra los valores de campo coercitivo Hc (mT) y '
        f'magnetización de remanencia Mr/Ms predichos por el modelo '
        f'GradientBoostingRegressor para {mat_name} a {d_nm:.1f} nm.',
        styles['body']))

    pred_data = [['Geometría', 'Hc (mT)', 'Mr/Ms', 'Estado']]
    for geom_key, geom_label in [('sphere', 'Esfera'), ('cuboid', 'Cuboide')]:
        if geom_key in preds:
            Hc_v, Mr_v = preds[geom_key]
            pred_data.append([
                geom_label,
                f'{Hc_v:.2f}',
                f'{Mr_v:.4f}',
                '⚠ Extrapolación' if extrapolation else '✓ Interpolación',
            ])
    tbl_pred = Table(pred_data,
                     colWidths=[4 * cm, 3.5 * cm, 3.5 * cm, 4.5 * cm])
    tbl_pred.setStyle(_TABLE_STYLE)
    story.append(tbl_pred)
    story.append(Spacer(1, 0.4 * cm))

    # ── Parámetros físicos ────────────────────────────────────────────────────
    story.append(Paragraph('3. Parámetros Físicos del Material', styles['section']))

    param_labels = {
        'K1_kJ_m3':    'Constante de anisotropía  K₁ (kJ/m³)',
        'A_pJ_m':      'Constante de intercambio  A (pJ/m)',
        'Ms_MA_m':     'Magnetización de saturación  Ms (MA/m)',
        'alpha':       'Factor de amortiguamiento  α (Gilbert)',
        'lambda_ex_nm':'Longitud de intercambio  λₑₓ (nm)',
        'Tc_K':        'Temperatura de Curie  Tc (K)',
    }
    params_data = [['Parámetro', 'Valor']] + [
        [label, str(mat_params.get(key, '—'))]
        for key, label in param_labels.items()
    ]
    tbl_params = Table(params_data, colWidths=[10 * cm, 5.5 * cm])
    tbl_params.setStyle(_PARAMS_STYLE)
    story.append(tbl_params)
    story.append(Spacer(1, 0.4 * cm))

    # ── Ecuaciones clave ──────────────────────────────────────────────────────
    story.append(Paragraph('4. Fundamentos Matemáticos', styles['section']))
    eqs = [
        ('Ecuación de Landau-Lifshitz-Gilbert (LLG):',
         'dM/dt = –γ(M × H_eff) + (α/Ms)[M × dM/dt]'),
        ('Energía de Zeeman:',
         'E_Z = –μ₀ Ms (M/Ms) · H · V'),
        ('Energía de intercambio:',
         'E_ex = A ∫|∇m|² dV'),
        ('Energía de anisotropía uniaxial:',
         'E_K = K₁ V sin²(θ)'),
        ('Histéresis LLG simplificada (rama ↑):',
         'M(H) = Mr · tanh[(H + Hc) / Hc]  + ε(σ)'),
    ]
    for label, eq in eqs:
        story.append(Paragraph(label, styles['body']))
        story.append(Paragraph(eq, styles['mono']))
        story.append(Spacer(1, 0.15 * cm))

    # ── Historial de sesión ───────────────────────────────────────────────────
    if history_rows:
        story.append(PageBreak())
        story.append(Paragraph('5. Historial de Simulaciones (Sesión)', styles['section']))
        story.append(Paragraph(
            f'Se registraron {len(history_rows)} simulaciones en la base de datos SQLite '
            'durante la sesión actual.',
            styles['body']))

        hist_headers = ['#', 'Hora', 'Material', 'Tamaño (nm)',
                         'Hc Esfera', 'Mr Esfera', 'Extrapol.']
        hist_data = [hist_headers]
        for i, row in enumerate(history_rows[:30], 1):   # máx 30 filas
            ts = row.get('timestamp', '')[:16].replace('T', ' ')
            hist_data.append([
                str(i),
                ts,
                row.get('material', '—'),
                f"{row.get('size_nm', '—'):.1f}",
                f"{row.get('hc_sphere', '—'):.1f}" if row.get('hc_sphere') else '—',
                f"{row.get('mr_sphere', '—'):.3f}" if row.get('mr_sphere') else '—',
                '⚠' if row.get('extrapolation') else '✓',
            ])

        col_w = [0.8, 2.4, 4.0, 2.2, 2.2, 2.2, 1.6]
        col_w = [x * cm for x in col_w]
        tbl_hist = Table(hist_data, colWidths=col_w, repeatRows=1)
        tbl_hist.setStyle(_TABLE_STYLE)
        story.append(tbl_hist)

    # ── Referencias ───────────────────────────────────────────────────────────
    story.append(Spacer(1, 0.6 * cm))
    story.append(HRFlowable(width='100%', thickness=1,
                             color=colors.HexColor('#cbd5e1'), spaceAfter=8))
    story.append(Paragraph('Referencias', styles['section']))
    refs = [
        '[1] Galvis, Mesa, et al. "Micromagnetic simulation of Fe nanoparticles." '
        '<i>Results in Physics</i>, 2025.',
        '[2] Galvis, Mesa, Restrepo. "ML-assisted micromagnetic modeling of Permalloy nanodots." '
        '<i>Computational Materials Science</i>, 2024.',
        '[3] Landau, L. D. & Lifshitz, E. M. "On the theory of the dispersion of magnetic '
        'permeability in ferromagnetic bodies." <i>Phys. Z. Sowjetunion</i>, 8:153, 1935.',
        '[4] Gilbert, T. L. "A phenomenological theory of damping in ferromagnetic materials." '
        '<i>IEEE Trans. Magn.</i>, 40(6):3443–3449, 2004.',
        '[5] Pedregosa et al. "Scikit-learn: Machine learning in Python." '
        '<i>JMLR</i>, 12:2825–2830, 2011.',
    ]
    for ref in refs:
        story.append(Paragraph(ref, styles['body']))
        story.append(Spacer(1, 0.1 * cm))

    # ── Build ─────────────────────────────────────────────────────────────────
    doc.build(story)
    buf_pdf.seek(0)
    return buf_pdf.read()
