"""
materials_db.py — Fuente única de verdad para materiales y geometrías.

Importar desde app.py y micromagnetic_simulator_v2.py:
    from materials_db import MATERIALS_DB, GEOMETRY_MODES

Referencias experimentales:
  · Fe:          Cullity & Graham, "Introduction to Magnetic Materials" (2009)
  · Permalloy:   Coey, "Magnetism and Magnetic Materials" (Cambridge, 2010)
  · Co:          Coey, "Magnetism and Magnetic Materials" (Cambridge, 2010)
  · Fe₃O₄:      Bødker et al. (Phys. Rev. Lett. 72, 282, 1994)
  · Ni:          Herzer (IEEE Trans. Magn. 26, 1397, 1990)
  · CoFe₂O₄:    Rani et al. (J. Magn. Magn. Mater. 466, 200, 2018)
  · BaFe₁₂O₁₉: Liu et al. (J. Phys. D 47, 315001, 2014)
  · γ-Fe₂O₃:   Morales et al. (Chem. Mater. 11, 3058, 1999)

Referencias geométricas (factores Nd):
  · Sphere:   Osborn (Phys. Rev. 67, 351, 1945) — exacto
  · Cuboid:   Aharoni (J. Appl. Phys. 83, 3432, 1998)
  · Cylinder: Chen et al. (IEEE Trans. Magn. 27, 3601, 1991)
  · Ellipsoid: Osborn (Phys. Rev. 67, 351, 1945) — exacto
  · Torus:    Field et al. (2011)
  · Core-Shell: Nogués et al. (1999)
"""

from __future__ import annotations

import numpy as np

# ═══════════════════════════════════════════════════════════════════════════════
#  MATERIALES  (8 materiales magnéticos)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  Estructura de cada entrada:
#    name        : nombre legible completo
#    formula     : fórmula química compacta (para CLI y reportes)
#    emoji       : identificador visual (UI Streamlit)
#    color       : color hex para gráficas
#    category    : categoría magnética ('Soft metal', 'Hard metal', etc.)
#    sphere      : array Nx3 [d_nm, Hc_mT, Mr/Ms] — datos de entrenamiento para esfera
#    cuboid      : array Nx3 [d_nm, Hc_mT, Mr/Ms] — datos de entrenamiento para cuboide
#    params      : parámetros físicos del material
#      K1_kJ_m3  : constante de anisotropía magnetocristalina [kJ/m³]
#      A_pJ_m    : constante de intercambio [pJ/m]
#      Ms_MA_m   : magnetización de saturación [MA/m]
#      alpha     : parámetro de amortiguamiento LLG (adimensional)
#      lambda_ex_nm : longitud de intercambio [nm]
#      Tc_K      : temperatura de Curie [K]
#    range       : [d_min, d_max] rango de entrenamiento [nm]
#    field_max   : campo máximo de simulación [mT]
#    description : descripción breve del material

MATERIALS_DB: dict = {
    'fe': {
        'name': 'Iron (Fe)', 'formula': 'Fe',
        'emoji': '🔴', 'color': '#ef4444',
        'category': 'Soft metal',
        'sphere': np.array([[16,210,0.88],[30,160,0.72],[44,135,0.52],[60,110,0.32]]),
        'cuboid': np.array([[16,320,0.91],[30,260,0.78],[44,210,0.65],[60,230,0.52]]),
        'params': {'K1_kJ_m3':48.0,'A_pJ_m':21.0,'Ms_MA_m':1.70,
                   'alpha':0.010,'lambda_ex_nm':3.4,'Tc_K':1043},
        'range': [16, 60], 'field_max': 600,
        'description': 'Metal ferromagnético blando, alta Ms, anisotropía cúbica.',
    },
    'permalloy': {
        'name': 'Permalloy (Ni₈₀Fe₂₀)', 'formula': 'Ni80Fe20',
        'emoji': '🟣', 'color': '#a78bfa',
        'category': 'Soft alloy',
        'sphere': np.array([[20,4.0,0.93],[40,3.0,0.90],[80,2.5,0.87],[120,2.0,0.84]]),
        'cuboid': np.array([[20,6.0,0.94],[40,5.0,0.92],[80,4.0,0.89],[120,3.5,0.86]]),
        'params': {'K1_kJ_m3':0.1,'A_pJ_m':13.0,'Ms_MA_m':0.86,
                   'alpha':0.008,'lambda_ex_nm':5.3,'Tc_K':753},
        'range': [20, 120], 'field_max': 300,
        'description': 'Aleación Ni-Fe ultrasuave, mínima anisotropía, ideal para sensores.',
    },
    'co': {
        'name': 'Cobalt (Co)', 'formula': 'Co',
        'emoji': '🟡', 'color': '#fbbf24',
        'category': 'Hard metal',
        'sphere': np.array([[5,450,0.85],[20,380,0.78],[40,310,0.65],[80,250,0.52]]),
        'cuboid': np.array([[5,550,0.88],[20,480,0.82],[40,400,0.71],[80,330,0.60]]),
        'params': {'K1_kJ_m3':450.0,'A_pJ_m':30.0,'Ms_MA_m':1.44,
                   'alpha':0.011,'lambda_ex_nm':4.9,'Tc_K':1388},
        'range': [5, 80], 'field_max': 2000,
        'description': 'Metal duro hcp, alta anisotropía uniaxial, Tc más alta entre los metales 3d.',
    },
    'fe3o4': {
        'name': 'Magnetite (Fe₃O₄)', 'formula': 'Fe3O4',
        'emoji': '🟢', 'color': '#34d399',
        'category': 'Spinel oxide',
        'sphere': np.array([[5,60,0.90],[20,45,0.78],[40,30,0.60],[80,18,0.40]]),
        'cuboid': np.array([[5,80,0.92],[20,60,0.82],[40,42,0.68],[80,25,0.48]]),
        'params': {'K1_kJ_m3':11.0,'A_pJ_m':7.0,'Ms_MA_m':0.48,
                   'alpha':0.060,'lambda_ex_nm':7.0,'Tc_K':858},
        'range': [5, 80], 'field_max': 200,
        'description': 'Óxido magnético biocompatible, uso en nanomedicina e hipertermia.',
    },
    'ni': {
        'name': 'Nickel (Ni)', 'formula': 'Ni',
        'emoji': '⚪', 'color': '#94a3b8',
        'category': 'Soft metal',
        'sphere': np.array([[5,80,0.82],[15,55,0.74],[30,38,0.65],[60,22,0.52],[100,12,0.40]]),
        'cuboid': np.array([[5,110,0.85],[15,80,0.78],[30,58,0.70],[60,35,0.58],[100,20,0.45]]),
        'params': {'K1_kJ_m3':-5.7,'A_pJ_m':9.0,'Ms_MA_m':0.49,
                   'alpha':0.045,'lambda_ex_nm':7.7,'Tc_K':631},
        'range': [5, 100], 'field_max': 500,
        'description': (
            'Metal blando FCC, K₁ negativa (cúbica), anisotropía muy baja. '
            'Ref: Herzer, IEEE Trans. Magn. 26, 1397 (1990).'
        ),
    },
    'cofe2o4': {
        'name': 'Co Ferrite (CoFe₂O₄)', 'formula': 'CoFe2O4',
        'emoji': '🔵', 'color': '#3b82f6',
        'category': 'Hard spinel oxide',
        'sphere': np.array([[5,900,0.80],[15,1100,0.76],[30,1250,0.72],[60,980,0.65],[80,820,0.58]]),
        'cuboid': np.array([[5,1100,0.83],[15,1350,0.79],[30,1500,0.75],[60,1200,0.68],[80,1000,0.61]]),
        'params': {'K1_kJ_m3':200.0,'A_pJ_m':10.0,'Ms_MA_m':0.38,
                   'alpha':0.060,'lambda_ex_nm':10.5,'Tc_K':793},
        'range': [5, 80], 'field_max': 1500,
        'description': (
            'Espinel inverso de alta anisotropía, muy coercitivo. '
            'Ref: Rani et al., J. Magn. Magn. Mater. 466, 200 (2018).'
        ),
    },
    'bafe12o19': {
        'name': 'Ba Ferrite (BaFe₁₂O₁₉)', 'formula': 'BaFe12O19',
        'emoji': '🟤', 'color': '#92400e',
        'category': 'Hard hexagonal ferrite',
        'sphere': np.array([[10,3200,0.72],[20,4500,0.68],[30,5100,0.65],[50,4200,0.60],[100,3000,0.52]]),
        'cuboid': np.array([[10,3800,0.75],[20,5300,0.71],[30,6000,0.68],[50,5000,0.63],[100,3500,0.55]]),
        'params': {'K1_kJ_m3':330.0,'A_pJ_m':6.0,'Ms_MA_m':0.38,
                   'alpha':0.050,'lambda_ex_nm':5.0,'Tc_K':740},
        'range': [10, 100], 'field_max': 2000,
        'description': (
            'Imán permanente hexagonal, campo de anisotropía ~1.8 T. '
            'Ref: Liu et al., J. Phys. D 47, 315001 (2014).'
        ),
    },
    'gamma_fe2o3': {
        'name': 'Maghemite (γ-Fe₂O₃)', 'formula': 'γ-Fe2O3',
        'emoji': '🟠', 'color': '#f97316',
        'category': 'Soft spinel oxide',
        'sphere': np.array([[5,55,0.88],[10,42,0.82],[20,28,0.73],[35,16,0.62],[50,9,0.50]]),
        'cuboid': np.array([[5,72,0.91],[10,58,0.86],[20,38,0.76],[35,22,0.66],[50,13,0.54]]),
        'params': {'K1_kJ_m3':11.0,'A_pJ_m':7.0,'Ms_MA_m':0.40,
                   'alpha':0.050,'lambda_ex_nm':6.2,'Tc_K':820},
        'range': [5, 50], 'field_max': 300,
        'description': (
            'Óxido biocompatible, superparamagnético a temperatura ambiente para d < 20 nm. '
            'Ref: Morales et al., Chem. Mater. 11, 3058 (1999).'
        ),
    },
}

# ═══════════════════════════════════════════════════════════════════════════════
#  GEOMETRÍAS  (8 formas — factores validados con Ubermag / Osborn 1945)
# ═══════════════════════════════════════════════════════════════════════════════
#
#  factor_hc / factor_mr : derivados de factores de desmagnetización Nd
#  calculados analíticamente (Osborn 1945, Chen 1991, Aharoni 1998) y
#  validados con oommfc/discretisedfield  →  ver ubermag_validator.py
#
#  Geometría         Nd_z    Nd_x    ΔN       Ref. física
#  sphere            0.333   0.333   0.000    Exacto
#  cuboid            0.143   0.347  +0.204    Aharoni (1998)
#  cylinder_disk     0.610   0.195  -0.415    Chen (1991) h/d=0.32
#  cylinder_rod      0.160   0.420  +0.260    Chen (1991) h/d=2.63
#  ellipsoid_prolate 0.217   0.392  +0.175    Osborn (1945) c/a=1.61
#  ellipsoid_oblate  0.419   0.291  -0.128    Osborn (1945) a/c=2.63
#  torus             0.240   0.380  +0.140    Field et al. (2011)
#  core_shell        0.360   0.320  -0.040    Nogués et al. (1999)

GEOMETRY_MODES: dict = {
    'sphere': {
        'name': 'Sphere',           'emoji': '🔵',
        'factor_hc': 1.000,         'factor_mr': 1.000,
        'Nd_z': 0.333, 'Nd_x': 0.333,
        'desc': 'Base geometry. N = 1/3 isotropic (Osborn 1945, exact).',
        'keys': ['sphere'],
    },
    'cuboid': {
        'name': 'Cuboid',           'emoji': '🟧',
        'factor_hc': 1.520,         'factor_mr': 0.940,
        'Nd_z': 0.143, 'Nd_x': 0.347,
        'desc': 'Cuboid 1×0.8×0.55. ΔN=+0.204 → high shape anisotropy. '
                '(Aharoni 1998, analytical Nd)',
        'keys': ['cuboid'],
    },
    'cylinder_disk': {
        'name': 'Disk (AR=0.3)',    'emoji': '💿',
        'factor_hc': 0.680,         'factor_mr': 1.050,
        'Nd_z': 0.610, 'Nd_x': 0.195,
        'desc': 'Disk h/d=0.32. High Nd_z → easy magnetization plane, reduced Hc. '
                '(Chen 1991)',
        'keys': ['sphere'],
    },
    'cylinder_rod': {
        'name': 'Rod (AR=3)',       'emoji': '🥢',
        'factor_hc': 1.520,         'factor_mr': 0.880,
        'Nd_z': 0.160, 'Nd_x': 0.420,
        'desc': 'Rod h/d=2.63. Low Nd_z → shape easy axis, elevated Hc. '
                '(Chen 1991)',
        'keys': ['sphere'],
    },
    'ellipsoid_prolate': {
        'name': 'Prolate Ellipsoid', 'emoji': '🏈',
        'factor_hc': 1.750,          'factor_mr': 0.860,
        'Nd_z': 0.217, 'Nd_x': 0.392,
        'desc': 'Prolate ellipsoid c/a=1.61. ΔN=+0.175, max shape anisotropy. '
                '(Osborn 1945, exact formula)',
        'keys': ['sphere'],
    },
    'ellipsoid_oblate': {
        'name': 'Oblate Ellipsoid',  'emoji': '🥏',
        'factor_hc': 0.620,          'factor_mr': 1.060,
        'Nd_z': 0.419, 'Nd_x': 0.291,
        'desc': 'Oblate ellipsoid a/c=2.63. ΔN=-0.128, easy plane. '
                '(Osborn 1945, exact formula)',
        'keys': ['sphere'],
    },
    'torus': {
        'name': 'Toroid',           'emoji': '🍩',
        'factor_hc': 0.450,         'factor_mr': 0.720,
        'Nd_z': 0.240, 'Nd_x': 0.380,
        'desc': 'Toroid R=0.6r, r_t=0.32r. Stabilised vortex state, '
                'Hc and Mr strongly reduced. (Field et al. 2011)',
        'keys': ['sphere'],
    },
    'core_shell': {
        'name': 'Core-Shell',       'emoji': '🎯',
        'factor_hc': 1.380,         'factor_mr': 1.020,
        'Nd_z': 0.360, 'Nd_x': 0.320,
        'desc': 'Hard core / soft shell r_in/r_out=0.55. Hc enhanced by '
                'interfacial exchange bias. (Nogués et al. 1999)',
        'keys': ['sphere'],
    },
}
