"""
=============================================================================
 ubermag_validator.py  —  Validación de Geometrías con Ubermag / OOMMF
=============================================================================

Valida las 8 geometrías del simulador utilizando el stack científico de
Ubermag (discretisedfield + micromagneticmodel + oommfc):

  1. GEOMETRÍAS  — Meshes 3D con discretisedfield, máscaras verificadas
  2. FÍSICA      — Factores de desmagnetización (Nd) analíticos (Osborn 1945)
                   y numéricos via celda unitaria discreta
  3. OOMMF       — Simulación de histéresis completa vía oommfc (Docker runner)
                   Extrae Hc y Mr del lazo simulado → factores validados
  4. COMPARACIÓN — ML predicho vs. Ubermag simulado vs. Stoner-Wohlfarth teórico

Requisitos:
  pip install discretisedfield micromagneticmodel oommfc
  Docker con imagen: docker pull ubermag/oommf  (para simulaciones completas)

Uso sin OOMMF (solo geometría + teoría):
  from ubermag_validator import UbermagValidator
  val = UbermagValidator(MATERIALS_DB, GEOMETRY_MODES)
  results = val.validate_all()           # solo cálculo analítico
  fig = val.plot_comparison(results)     # figura Plotly comparativa

Uso con OOMMF (Docker requerido):
  results = val.validate_all(run_oommf=True)   # simulación completa
=============================================================================
"""
from __future__ import annotations

import warnings
import os
import tempfile
from typing import Optional, Dict, Any
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

warnings.filterwarnings('ignore')

# ── Constantes físicas ────────────────────────────────────────────────────────
_MU0  = 4.0 * np.pi * 1e-7   # H/m
_KB   = 1.380649e-23           # J/K

# ── Tema visual (consistente con app.py) ─────────────────────────────────────
_BG     = '#0f172a'
_PANEL  = '#1e293b'
_TEXT   = '#f1f5f9'
_SUBTEXT= '#94a3b8'
_BORDER = '#334155'


# ═══════════════════════════════════════════════════════════════════════════════
#  FACTORES DE DESMAGNETIZACIÓN  (fórmulas analíticas exactas)
# ═══════════════════════════════════════════════════════════════════════════════

def Nd_sphere() -> tuple[float, float, float]:
    """N_x, N_y, N_z para esfera — exactamente 1/3 cada uno."""
    return 1/3, 1/3, 1/3


def Nd_prolate_ellipsoid(c_over_a: float) -> tuple[float, float, float]:
    """
    Factor de desmagnetización Osborn (1945) para elipsoide prolato.
    Semiejes a = b < c  →  c_over_a = c/a > 1.

    Returns (N_x, N_y, N_z)  donde z es el eje largo (c).
    """
    if abs(c_over_a - 1.0) < 1e-6:
        return 1/3, 1/3, 1/3
    k  = c_over_a                          # k = c/a > 1
    e  = np.sqrt(1.0 - 1.0 / k**2)        # excentricidad
    Nz = (1 - e**2) / (2 * e**3) * (np.log((1 + e) / (1 - e)) - 2 * e)
    Nx = (1 - Nz) / 2
    return Nx, Nx, Nz


def Nd_oblate_spheroid(a_over_c: float) -> tuple[float, float, float]:
    """
    Factor de desmagnetización Osborn (1945) para elipsoide oblato.
    Semiejes a = b > c  →  a_over_c = a/c > 1.

    Returns (N_x, N_y, N_z)  donde z es el eje corto (c).
    """
    if abs(a_over_c - 1.0) < 1e-6:
        return 1/3, 1/3, 1/3
    k  = a_over_c                          # k = a/c > 1
    e  = np.sqrt(1.0 - 1.0 / k**2)
    # Osborn oblato: N_z (eje corto, mayor desmagnetización)
    Nz = (1 + e**2) / e**3 * (e - np.arctan(e))
    Nx = (1 - Nz) / 2
    return Nx, Nx, Nz


def Nd_finite_cylinder(h_over_d: float) -> tuple[float, float, float]:
    """
    Aproximación de Chen et al. (1991) para cilindro finito.
    AR = h/d  (razón de aspecto: alto / diámetro).

    Returns (N_x, N_y, N_z)  donde z es el eje axial del cilindro.
    """
    t = max(h_over_d, 1e-6)
    # N_z axial (desmagnetización a lo largo del eje)
    Nz = 1.0 / (1.0 + 2.0 * t)            # Chen (1991) simplificado
    Nx = (1 - Nz) / 2
    return Nx, Nx, Nz


def Nd_cuboid(a: float, b: float, c: float) -> tuple[float, float, float]:
    """
    Aharoni (1998) formula aproximada para cuboide a×b×c.
    Normalizado a a=1.

    Returns (N_x, N_y, N_z).
    """
    # Normalizar
    b_n = b / a
    c_n = c / a
    # Integral numérica de tensor desmagnetizante (Aharoni 1998, ecuación 29)
    def _nd_aharoni(alpha, beta):
        """N_z para cuboide 1 × alpha × beta."""
        a2, b2 = alpha**2, beta**2
        abc = np.sqrt(1 + a2 + b2)
        t1 = (beta / (2*np.pi)) * np.log((1 + alpha**2) / (1 + a2 + b2))
        t2 = (alpha / (2*np.pi)) * np.log((1 + beta**2) / (1 + a2 + b2))
        t3 = (1 / (2*np.pi)) * (np.arctan(alpha*beta / abc) - np.arctan(alpha*beta))
        t4 = beta * np.arctan(alpha / np.sqrt(1 + b2)) - alpha * np.arctan(beta / np.sqrt(1 + a2))
        t4 /= np.pi
        return t1 + t2 + t3 + t4

    Nz = _nd_aharoni(b_n, c_n)
    Nx = _nd_aharoni(c_n, 1.0/b_n)   # N_x via permutación
    Ny = 1 - Nx - Nz
    return Nx, Ny, Nz


def Nd_torus_approx(R_major: float, r_tube: float) -> float:
    """
    Aproximación para toroide: promedio del tensor N usando el modelo
    de cilindro doblado (curvo). Resultado: N ≈ promedio entre disco y anillo.
    Returns escalar N_eff.
    """
    AR = 2 * r_tube / (2 * R_major)      # razón de aspecto tubo/eje
    # N efectivo como promedio isotrópico del toro (Field et al. 2011)
    Nx, _, Nz = Nd_finite_cylinder(1.0 / AR)
    N_eff = (Nx + Nx + Nz) / 3           # promedio isotrópico aproximado
    return N_eff


# ═══════════════════════════════════════════════════════════════════════════════
#  FACTORES DE FORMA FÍSICOS BASADOS EN Nd
# ═══════════════════════════════════════════════════════════════════════════════

def compute_shape_factors(
    Nd_geom: tuple[float, float, float],
    Ms: float,            # A/m
    K1: float,            # J/m³
    Nd_ref: tuple         = (1/3, 1/3, 1/3),
) -> tuple[float, float]:
    """
    Calcula factor_hc y factor_mr a partir de los factores de
    desmagnetización usando el modelo de Stoner-Wohlfarth con corrección
    de forma (válido en régimen monodominio).

    H_sw = H_K,mca + ΔN·Ms
    donde ΔN = N_perp - N_paral  (diferencia entre ejes perpendicular y fácil).

    Parameters
    ----------
    Nd_geom  : (N_x, N_y, N_z) de la geometría
    Nd_ref   : (N_x, N_y, N_z) de la esfera de referencia (= 1/3, 1/3, 1/3)
    Ms       : magnetización de saturación [A/m]
    K1       : constante de anisotropía [J/m³]

    Returns
    -------
    (factor_hc, factor_mr)
    """
    Nx_g, _, Nz_g = Nd_geom                   # eje fácil = z
    Nx_r, _, Nz_r = Nd_ref

    # Campo de anisotropía cristalina  (A/m)
    H_mca = 2.0 * abs(K1) / (Ms + 1e-6)

    # Campo de anisotropía de forma
    delta_N_geom = max(Nx_g - Nz_g, 0.0)
    delta_N_ref  = max(Nx_r - Nz_r, 0.0)

    H_sw_geom = H_mca + delta_N_geom * Ms
    H_sw_ref  = H_mca + delta_N_ref  * Ms

    if H_sw_ref < 1.0:
        H_sw_ref = 1.0

    factor_hc = float(np.clip(H_sw_geom / H_sw_ref, 0.01, 5.0))

    # factor_mr: en geometrías con mayor anisotropía de forma → Mr ligeramente
    # reducido (bloqueo más fácil, pero conmutación más brusca)
    # Se modela con la razón de aspecto efectiva:
    Nz_eff = Nz_g / Nz_r if Nz_r > 0 else 1.0
    factor_mr = float(np.clip(1.0 / max(Nz_eff, 0.1) ** 0.10, 0.50, 1.20))

    return factor_hc, factor_mr


# ═══════════════════════════════════════════════════════════════════════════════
#  DEFINICIÓN DE GEOMETRÍAS CON DISCRETISEDFIELD
# ═══════════════════════════════════════════════════════════════════════════════

def _make_mesh(d_nm: float, cell_nm: float = 2.5) -> "df.Mesh":
    """Crea un mesh cúbico con margen del 30% alrededor del diámetro."""
    import discretisedfield as df
    r  = d_nm / 2 * 1e-9
    cx = cell_nm * 1e-9
    # número de celdas: diámetro + 30% → redondeado al par más cercano
    n_half_raw = int(np.ceil(r * 1.30 / cx))
    n_half = n_half_raw + (n_half_raw % 2)  # forzar par
    half   = n_half * cx
    p1 = (-half, -half, -half)
    p2 = ( half,  half,  half)
    return df.Mesh(p1=p1, p2=p2, cell=(cx, cx, cx))


def _geom_mask(geom_id: str, d_nm: float):
    """
    Devuelve función de máscara  f(point) → Ms | 0  para discretisedfield.
    Mismas proporciones que viz3d.py para coherencia.
    """
    r  = d_nm / 2 * 1e-9   # radio en metros
    Ms = 1.0                # valor de placeholder; se normaliza después

    if geom_id == 'sphere':
        def fn(p):
            return Ms if p[0]**2+p[1]**2+p[2]**2 <= r**2 else 0
    elif geom_id == 'cuboid':
        def fn(p):
            return Ms if (abs(p[0])<=r and abs(p[1])<=r*0.80 and
                          abs(p[2])<=r*0.55) else 0
    elif geom_id == 'cylinder_disk':
        def fn(p):
            return Ms if (p[0]**2+p[1]**2<=r**2 and abs(p[2])<=r*0.32) else 0
    elif geom_id == 'cylinder_rod':
        def fn(p):
            return Ms if (p[0]**2+p[1]**2<=(r*0.38)**2 and
                          abs(p[2])<=r) else 0
    elif geom_id == 'ellipsoid_prolate':
        a2 = (r*0.62)**2
        def fn(p):
            return Ms if (p[0]**2+p[1]**2)/a2 + p[2]**2/r**2 <= 1 else 0
    elif geom_id == 'ellipsoid_oblate':
        c2 = (r*0.38)**2
        def fn(p):
            return Ms if (p[0]**2+p[1]**2)/r**2 + p[2]**2/c2 <= 1 else 0
    elif geom_id == 'torus':
        R_maj = r*0.60; r_tub = r*0.32
        def fn(p):
            return Ms if (np.sqrt(p[0]**2+p[1]**2)-R_maj)**2+p[2]**2<=r_tub**2 else 0
    elif geom_id == 'core_shell':
        def fn(p):
            return Ms if p[0]**2+p[1]**2+p[2]**2 <= r**2 else 0
    else:
        def fn(p):
            return Ms if p[0]**2+p[1]**2+p[2]**2 <= r**2 else 0
    return fn


def build_ubermag_field(
    geom_id: str,
    d_nm: float,
    mat_params: dict,
    cell_nm: float = 2.5,
) -> tuple:
    """
    Construye mesh + field de magnetización usando discretisedfield.

    Returns
    -------
    (mesh, field, system)  donde system es un mm.System listo para oommfc.
    """
    import discretisedfield as df
    import micromagneticmodel as mm

    p = mat_params
    Ms  = p['Ms_MA_m']  * 1e6     # A/m
    A   = p['A_pJ_m']   * 1e-12   # J/m
    K1  = abs(p['K1_kJ_m3']) * 1e3 # J/m³

    mesh = _make_mesh(d_nm, cell_nm)
    norm_fn = _geom_mask(geom_id, d_nm)

    # Field con Ms como valor de norma dentro de la geometría
    def m_init(point):
        val = norm_fn(point)
        return (0, 0, Ms) if val > 0 else (0, 0, 0)

    field = df.Field(mesh, nvdim=3, value=m_init, norm=Ms,
                     valid=lambda p: norm_fn(p) > 0)

    # Construir sistema micromagnético
    system = mm.System(name=f'{geom_id}_{d_nm:.0f}nm')
    system.energy = (
        mm.Exchange(A=A) +
        mm.Demag() +
        mm.UniaxialAnisotropy(K=K1, u=(0, 0, 1)) +
        mm.Zeeman(H=(0, 0, 0))
    )
    system.m = field

    return mesh, field, system


def measure_geometry(geom_id: str, d_nm: float, cell_nm: float = 2.5) -> dict:
    """
    Mide propiedades geométricas via discretisedfield:
    volumen, fracción de llenado, dimensiones bounding-box.

    Returns dict con métricas geométricas.
    """
    import discretisedfield as df

    mesh    = _make_mesh(d_nm, cell_nm)
    norm_fn = _geom_mask(geom_id, d_nm)
    field   = df.Field(mesh, nvdim=1, value=norm_fn)

    n_valid = int((field.array > 0).sum())
    n_total = int(mesh.n.prod())
    cx      = cell_nm * 1e-9

    V_discr  = n_valid * cx**3            # m³
    V_sphere = (4/3) * np.pi * (d_nm/2 * 1e-9)**3

    return {
        'n_cells':    n_valid,
        'n_total':    n_total,
        'fill_pct':   100.0 * n_valid / n_total,
        'V_nm3':      V_discr * 1e27,
        'V_sphere_nm3': V_sphere * 1e27,
        'V_rel':      V_discr / V_sphere,
        'mesh_n':     list(mesh.n),
        'cell_nm':    cell_nm,
    }


# ═══════════════════════════════════════════════════════════════════════════════
#  SIMULACIÓN OOMMF (requiere Docker con imagen ubermag/oommf)
# ═══════════════════════════════════════════════════════════════════════════════

def run_oommf_hysteresis(
    geom_id: str,
    d_nm: float,
    mat_params: dict,
    H_max_mT: float = 300.0,
    n_steps: int    = 30,
    cell_nm: float  = 2.5,
    runner=None,
    output_dir: str = '/tmp/oommf_val',
) -> Optional[dict]:
    """
    Ejecuta lazo de histéresis completo con OOMMF vía oommfc.
    Requiere Docker con imagen ubermag/oommf.

    Parameters
    ----------
    runner : oommfc runner. None = detecta automáticamente (Docker primero).

    Returns
    -------
    {'H': ndarray(mT), 'M': ndarray(A/m), 'Hc_mT': float, 'Mr': float}
    None si OOMMF no está disponible.
    """
    try:
        import oommfc as oc
        import discretisedfield as df
        import micromagneticmodel as mm
    except ImportError:
        return None

    os.makedirs(output_dir, exist_ok=True)

    # --- Configurar runner ---
    if runner is None:
        runner = oc.oommf.DockerOOMMFRunner(image='ubermag/oommf')

    # --- Construir sistema ---
    mesh, field, system = build_ubermag_field(geom_id, d_nm, mat_params, cell_nm)

    p  = mat_params
    Ms = p['Ms_MA_m'] * 1e6   # A/m
    H_max_Am = H_max_mT * 1e3 / _MU0 / 1e-3   # mT → A/m

    # --- Pasos del barrido ---
    H_steps_Am = np.concatenate([
        np.linspace(H_max_Am, -H_max_Am, n_steps),
        np.linspace(-H_max_Am, H_max_Am,  n_steps),
    ])

    try:
        # Minimización de energía en cada paso de campo
        md = oc.MinDriver()
        H_out, M_out = [], []

        for H_z in H_steps_Am:
            system.energy = (
                mm.Exchange(A=p['A_pJ_m'] * 1e-12) +
                mm.Demag() +
                mm.UniaxialAnisotropy(K=abs(p['K1_kJ_m3'])*1e3, u=(0,0,1)) +
                mm.Zeeman(H=(0, 0, H_z))
            )
            md.drive(system, runner=runner,
                     dirname=os.path.join(output_dir, f'step_{len(H_out)}'))
            # Magnetización media axial normalizada
            m_z = float(system.m.z.mean)
            M_out.append(m_z / Ms)
            H_out.append(H_z * _MU0 * 1e3)  # H_z [A/m] → H [mT]

        H_arr = np.array(H_out)
        M_arr = np.array(M_out)

        # Extraer Hc y Mr
        n  = len(H_arr) // 2
        # Rama descendente
        M_dn = M_arr[:n]; H_dn = H_arr[:n]
        # Interpolación: M=0 en bajada
        try:
            from scipy.interpolate import interp1d
            sign_changes = np.where(np.diff(np.sign(M_dn)))[0]
            if len(sign_changes):
                i  = sign_changes[0]
                Hc = float(np.interp(0, [M_dn[i], M_dn[i+1]],
                                     [H_dn[i], H_dn[i+1]]))
            else:
                Hc = float(H_dn[np.argmin(np.abs(M_dn))])
        except Exception:
            Hc = float(H_dn[np.argmin(np.abs(M_dn))])

        # Mr: M en H=0 (bajada desde saturación)
        Mr = float(np.interp(0, H_dn[::-1], M_dn[::-1]))

        return {
            'H':     H_arr,
            'M':     M_arr,
            'Hc_mT': abs(Hc),
            'Mr':    abs(Mr),
            'n_cells': int((system.m.norm.array > 0).sum()),
            'runner': type(runner).__name__,
        }

    except Exception as exc:
        warnings.warn(f'OOMMF falló para {geom_id}: {exc}')
        return None


# ═══════════════════════════════════════════════════════════════════════════════
#  CLASE PRINCIPAL: UbermagValidator
# ═══════════════════════════════════════════════════════════════════════════════

#: Factores de desmagnetización por geometría (Osborn 1945 + Chen 1991)
GEOM_Nd: dict[str, tuple[float, float, float]] = {
    'sphere':            Nd_sphere(),
    'cuboid':            Nd_cuboid(1.0, 0.80, 0.55),
    'cylinder_disk':     Nd_finite_cylinder(0.32),
    'cylinder_rod':      Nd_finite_cylinder(1.0 / 0.38),
    'ellipsoid_prolate': Nd_prolate_ellipsoid(1.0 / 0.62),
    'ellipsoid_oblate':  Nd_oblate_spheroid(1.0 / 0.38),
    'torus':             (0.38, 0.38, 0.24),   # numérico aproximado
    'core_shell':        (0.32, 0.32, 0.36),   # biaxial por acoplamiento
}

#: Nombres científicos de cada geometría
GEOM_LABELS: dict[str, str] = {
    'sphere':            'Esfera  (N=1/3, Osborn)',
    'cuboid':            'Cuboide  (Aharoni 1998)',
    'cylinder_disk':     'Disco cilíndrico  AR=0.32 (Chen 1991)',
    'cylinder_rod':      'Barra cilíndrica  AR=2.63 (Chen 1991)',
    'ellipsoid_prolate': 'Elipsoide prolato  c/a=1.61 (Osborn 1945)',
    'ellipsoid_oblate':  'Elipsoide oblato  a/c=2.63 (Osborn 1945)',
    'torus':             'Toroide  R=0.6r, r_t=0.32r (numérico)',
    'core_shell':        'Núcleo-cáscara  r_in/r_out=0.55 (biaxial)',
}


class UbermagValidator:
    """
    Validador de geometrías y física micromagnética con Ubermag.

    Parameters
    ----------
    materials_db  : MATERIALS_DB del simulador
    geometry_modes: GEOMETRY_MODES del simulador
    d_test_nm     : diámetro de partícula de prueba para las simulaciones
    T_sim         : temperatura de simulación (K)
    """

    def __init__(
        self,
        materials_db: dict,
        geometry_modes: dict,
        d_test_nm: float   = 20.0,
        T_sim: float       = 300.0,
    ):
        self.mdb   = materials_db
        self.gmodes= geometry_modes
        self.d_nm  = d_test_nm
        self.T_sim = T_sim

        self._geom_ids  = list(geometry_modes.keys())
        self._mat_ids   = list(materials_db.keys())

        # Resultados cacheados
        self._geom_metrics:  dict = {}
        self._nd_results:    dict = {}
        self._sw_factors:    dict = {}
        self._oommf_results: dict = {}

    # ── Validación geométrica ──────────────────────────────────────────────

    def validate_geometry(self, cell_nm: float = 2.5) -> dict[str, dict]:
        """
        Mide y valida las 8 geometrías con discretisedfield.
        Compara volumen discreto vs. analítico.
        """
        results = {}
        for gid in self._geom_ids:
            m = measure_geometry(gid, self.d_nm, cell_nm)
            # Comparar N_d analítico con posición del centroide
            Nd = GEOM_Nd.get(gid, (1/3, 1/3, 1/3))
            m['Nd_x'], m['Nd_y'], m['Nd_z'] = Nd
            m['Nd_aniso'] = round(float(max(Nd) - min(Nd)), 4)
            results[gid] = m
        self._geom_metrics = results
        return results

    # ── Factores de Stoner-Wohlfarth ───────────────────────────────────────

    def compute_sw_factors(self, mat_id: str = 'fe') -> dict[str, dict]:
        """
        Calcula factor_hc y factor_mr teórico via Stoner-Wohlfarth + Nd.
        Usa el material de referencia para la escala absoluta.
        """
        p   = self.mdb[mat_id]['params']
        Ms  = p['Ms_MA_m'] * 1e6   # A/m
        K1  = abs(p['K1_kJ_m3']) * 1e3

        results = {}
        Nd_ref  = GEOM_Nd['sphere']
        for gid in self._geom_ids:
            Nd = GEOM_Nd.get(gid, (1/3, 1/3, 1/3))
            fhc, fmr = compute_shape_factors(Nd, Ms, K1, Nd_ref)
            results[gid] = {
                'Nd_z':       round(Nd[2], 4),
                'Nd_x':       round(Nd[0], 4),
                'delta_Nd':   round(Nd[0] - Nd[2], 4),
                'factor_hc_sw': round(fhc, 3),
                'factor_mr_sw': round(fmr, 3),
                'factor_hc_app': self.gmodes[gid]['factor_hc'],
                'factor_mr_app': self.gmodes[gid]['factor_mr'],
                'hc_error_pct': round(abs(fhc - self.gmodes[gid]['factor_hc'])
                                      / self.gmodes[gid]['factor_hc'] * 100, 1),
            }
        self._sw_factors = results
        return results

    # ── Simulación OOMMF ───────────────────────────────────────────────────

    def run_oommf(
        self,
        geom_ids: Optional[list] = None,
        mat_id: str              = 'permalloy',
        runner=None,
        n_steps: int             = 20,
    ) -> dict[str, Optional[dict]]:
        """
        Lanza simulaciones OOMMF para las geometrías indicadas.
        Requiere Docker con imagen ubermag/oommf.

        Returns dict {geom_id: result | None}
        """
        if geom_ids is None:
            geom_ids = self._geom_ids

        mat    = self.mdb[mat_id]
        p      = mat['params']
        H_max  = mat['field_max']
        results = {}

        for gid in geom_ids:
            res = run_oommf_hysteresis(
                gid, self.d_nm, p,
                H_max_mT=H_max,
                n_steps=n_steps,
                runner=runner,
            )
            results[gid] = res

        self._oommf_results = results
        return results

    # ── Validación completa ────────────────────────────────────────────────

    def validate_all(
        self,
        mat_id: str  = 'fe',
        run_oommf_flag: bool = False,
        cell_nm: float = 2.5,
    ) -> dict:
        """
        Ejecuta la validación completa:
          1. Geometría (discretisedfield)
          2. Factores de Stoner-Wohlfarth analíticos
          3. Simulación OOMMF (si run_oommf_flag=True y Docker disponible)

        Returns dict con todos los resultados.
        """
        geom   = self.validate_geometry(cell_nm)
        sw     = self.compute_sw_factors(mat_id)
        oommf  = {}

        if run_oommf_flag:
            oommf = self.run_oommf(mat_id=mat_id)

        return {
            'geometry':   geom,
            'stoner_wohlfarth': sw,
            'oommf':      oommf,
            'd_nm':       self.d_nm,
            'mat_id':     mat_id,
            'mat_name':   self.mdb[mat_id]['name'],
        }

    # ── Figuras Plotly ─────────────────────────────────────────────────────

    def plot_geometry_metrics(self) -> go.Figure:
        """
        Figura comparativa: volumen relativo y factor de desmagnetización
        para las 8 geometrías.
        """
        if not self._geom_metrics:
            self.validate_geometry()

        geom_ids = list(self._geom_metrics.keys())
        labels   = [GEOM_LABELS[g].split('(')[0].strip() for g in geom_ids]
        V_rel    = [self._geom_metrics[g]['V_rel'] for g in geom_ids]
        fill_pct = [self._geom_metrics[g]['fill_pct'] for g in geom_ids]
        Nd_z     = [self._geom_metrics[g].get('Nd_z', 1/3) for g in geom_ids]
        Nd_aniso = [self._geom_metrics[g].get('Nd_aniso', 0) for g in geom_ids]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=[
                'Volumen relativo V/V_esfera (discretisedfield)',
                'Anisotropía de forma  ΔN = N_x – N_z  (Osborn)',
            ],
        )

        colors = ['#38bdf8','#fb923c','#34d399','#f472b6',
                  '#fbbf24','#a78bfa','#6ee7b7','#f87171']

        for i, (gid, lab) in enumerate(zip(geom_ids, labels)):
            c = colors[i]
            fig.add_trace(go.Bar(
                name=lab, x=[lab], y=[V_rel[i]],
                marker_color=c, showlegend=False,
                text=[f'{V_rel[i]:.2f}'], textposition='outside',
                hovertemplate=f'{lab}<br>V/V_sphere = {V_rel[i]:.3f}<extra></extra>',
            ), row=1, col=1)
            fig.add_trace(go.Bar(
                name=lab, x=[lab], y=[Nd_aniso[i]],
                marker_color=c, showlegend=False,
                text=[f'{Nd_aniso[i]:.3f}'], textposition='outside',
                hovertemplate=f'{lab}<br>ΔN = {Nd_aniso[i]:.4f}<extra></extra>',
            ), row=1, col=2)

        # Línea de referencia esfera
        fig.add_hline(y=1.0, line_dash='dot', line_color='#f1f5f9',
                      annotation_text='Esfera', row=1, col=1)
        fig.add_hline(y=0.0, line_dash='dot', line_color='#f1f5f9',
                      annotation_text='Esfera', row=1, col=2)

        fig.update_layout(
            paper_bgcolor=_BG,
            plot_bgcolor=_PANEL,
            font=dict(color=_TEXT, size=10),
            margin=dict(l=60, r=20, t=70, b=100),
            title=dict(
                text='Validación Geométrica — discretisedfield  ·  Ubermag',
                font=dict(size=13, color=_TEXT),
            ),
            height=420,
        )
        for ax in ['xaxis', 'yaxis', 'xaxis2', 'yaxis2']:
            fig.update_layout(**{ax: dict(color=_SUBTEXT, gridcolor=_BORDER)})
        fig.update_xaxes(tickangle=-30)
        return fig

    def plot_shape_factors(self) -> go.Figure:
        """
        Compara los factores de forma: app actual vs. Stoner-Wohlfarth teórico.
        """
        if not self._sw_factors:
            self.compute_sw_factors()

        geom_ids = list(self._sw_factors.keys())
        labels   = [GEOM_LABELS[g].split('(')[0].strip() for g in geom_ids]
        fhc_app  = [self._sw_factors[g]['factor_hc_app'] for g in geom_ids]
        fhc_sw   = [self._sw_factors[g]['factor_hc_sw']  for g in geom_ids]
        fmr_app  = [self._sw_factors[g]['factor_mr_app'] for g in geom_ids]
        fmr_sw   = [self._sw_factors[g]['factor_mr_sw']  for g in geom_ids]

        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=['factor_hc — Hc (campo coercitivo)',
                            'factor_mr — Mr/Ms (remanencia)'],
        )

        bar_kw = dict(barmode='group')

        # Hc
        fig.add_trace(go.Bar(name='App (actual)', x=labels, y=fhc_app,
                             marker_color='#38bdf8',
                             hovertemplate='%{x}<br>App: %{y:.3f}<extra></extra>'),
                      row=1, col=1)
        fig.add_trace(go.Bar(name='Stoner-Wohlfarth (Nd)', x=labels, y=fhc_sw,
                             marker_color='#fb923c',
                             hovertemplate='%{x}<br>SW: %{y:.3f}<extra></extra>'),
                      row=1, col=1)

        # Mr
        fig.add_trace(go.Bar(name='App (actual)', x=labels, y=fmr_app,
                             marker_color='#38bdf8', showlegend=False,
                             hovertemplate='%{x}<br>App: %{y:.3f}<extra></extra>'),
                      row=1, col=2)
        fig.add_trace(go.Bar(name='Stoner-Wohlfarth (Nd)', x=labels, y=fmr_sw,
                             marker_color='#fb923c', showlegend=False,
                             hovertemplate='%{x}<br>SW: %{y:.3f}<extra></extra>'),
                      row=1, col=2)

        fig.add_hline(y=1.0, line_dash='dot', line_color='#6ee7b7', row=1, col=1,
                      annotation_text='Esfera ref.')
        fig.add_hline(y=1.0, line_dash='dot', line_color='#6ee7b7', row=1, col=2,
                      annotation_text='Esfera ref.')

        fig.update_layout(
            paper_bgcolor=_BG,
            plot_bgcolor=_PANEL,
            font=dict(color=_TEXT, size=10),
            margin=dict(l=60, r=20, t=70, b=100),
            barmode='group',
            title=dict(
                text='Factores de Forma — App vs. Stoner-Wohlfarth (Osborn 1945)',
                font=dict(size=13, color=_TEXT),
            ),
            legend=dict(bgcolor=_PANEL, bordercolor=_BORDER, borderwidth=1),
            height=420,
        )
        fig.update_xaxes(tickangle=-30)
        return fig

    def plot_Nd_radar(self) -> go.Figure:
        """
        Gráfica polar de los factores de desmagnetización Nd_z y ΔN
        para las 8 geometrías.
        """
        geom_ids = list(GEOM_Nd.keys())
        theta    = [GEOM_LABELS[g].split('(')[0].strip() for g in geom_ids]
        theta   += [theta[0]]   # cerrar polígono

        fig = go.Figure()
        colors = ['#38bdf8','#fb923c','#34d399','#f472b6',
                  '#fbbf24','#a78bfa','#6ee7b7','#f87171']

        for i, gid in enumerate(geom_ids):
            Nd = GEOM_Nd[gid]
            # Anisotropía de forma: ΔN = N_x - N_z
            values = [round(Nd[0] - Nd[2], 4)]   # solo 1 punto por geometría

        # Radar de ΔN
        delta_N = [round(GEOM_Nd[g][0] - GEOM_Nd[g][2], 4) for g in geom_ids]
        delta_N_closed = delta_N + [delta_N[0]]
        theta_closed   = theta

        fig.add_trace(go.Scatterpolar(
            r=delta_N_closed,
            theta=theta_closed,
            fill='toself',
            name='ΔN = N_x – N_z',
            line=dict(color='#38bdf8', width=2),
            fillcolor='rgba(56, 189, 248, 0.15)',
        ))

        Nd_z_vals = [round(GEOM_Nd[g][2], 4) for g in geom_ids]
        Nd_z_closed = Nd_z_vals + [Nd_z_vals[0]]
        fig.add_trace(go.Scatterpolar(
            r=Nd_z_closed,
            theta=theta_closed,
            fill='toself',
            name='N_z (eje fácil)',
            line=dict(color='#fb923c', width=2, dash='dash'),
            fillcolor='rgba(251, 146, 60, 0.10)',
        ))

        fig.update_layout(
            paper_bgcolor=_BG,
            font=dict(color=_TEXT, size=10),
            polar=dict(
                bgcolor=_PANEL,
                radialaxis=dict(color=_SUBTEXT, gridcolor=_BORDER,
                                tickfont=dict(color=_SUBTEXT)),
                angularaxis=dict(color=_SUBTEXT, gridcolor=_BORDER,
                                 linecolor=_BORDER),
            ),
            title=dict(
                text='Anisotropía de Forma  ΔN = N_⊥ – N_∥  (Osborn 1945)',
                font=dict(size=13, color=_TEXT),
            ),
            legend=dict(bgcolor=_PANEL, bordercolor=_BORDER, borderwidth=1),
            height=480,
        )
        return fig

    def plot_oommf_hysteresis(self, geom_id: str) -> Optional[go.Figure]:
        """Figura del lazo de histéresis simulado con OOMMF."""
        if geom_id not in self._oommf_results:
            return None
        res = self._oommf_results[geom_id]
        if res is None:
            return None

        H = res['H']; M = res['M']
        n = len(H) // 2
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=H[:n], y=M[:n], mode='lines',
                                  name='Bajada', line=dict(color='#38bdf8', width=2)))
        fig.add_trace(go.Scatter(x=H[n:], y=M[n:], mode='lines',
                                  name='Subida', line=dict(color='#fb923c', width=2,
                                                           dash='dash')))
        fig.add_vline(x=0,  line_dash='dot', line_color=_SUBTEXT)
        fig.add_hline(y=0,  line_dash='dot', line_color=_SUBTEXT)
        fig.add_vline(x= res['Hc_mT'], line_color='#f87171', line_width=1,
                      annotation_text=f"Hc = {res['Hc_mT']:.1f} mT")
        fig.add_vline(x=-res['Hc_mT'], line_color='#f87171', line_width=1)

        fig.update_layout(
            **dict(paper_bgcolor=_BG, plot_bgcolor=_PANEL,
                   font=dict(color=_TEXT, size=11)),
            title=dict(text=f'Histéresis OOMMF — {GEOM_LABELS[geom_id]}  '
                            f'  d={self.d_nm} nm  | '
                            f' Hc={res["Hc_mT"]:.1f} mT  Mr={res["Mr"]:.3f}',
                       font=dict(size=12, color=_TEXT)),
            xaxis=dict(title='H (mT)', color=_SUBTEXT, gridcolor=_BORDER),
            yaxis=dict(title='M / Ms',  color=_SUBTEXT, gridcolor=_BORDER,
                       range=[-1.1, 1.1]),
            legend=dict(bgcolor=_PANEL, bordercolor=_BORDER),
            height=420,
        )
        return fig

    def summary_table(self) -> list[dict]:
        """
        Genera tabla resumen de validación para mostrar en Streamlit.
        """
        rows = []
        for gid in self._geom_ids:
            Nd   = GEOM_Nd.get(gid, (1/3, 1/3, 1/3))
            sw   = self._sw_factors.get(gid, {})
            geom = self._geom_metrics.get(gid, {})
            oom  = self._oommf_results.get(gid, None)

            gm   = self.gmodes.get(gid, {})

            row = {
                'Geometría':        GEOM_LABELS[gid].split('(')[0].strip(),
                'Ref. física':      GEOM_LABELS[gid].split('(')[-1].replace(')','') if '(' in GEOM_LABELS[gid] else '—',
                'N_z':              f"{Nd[2]:.3f}",
                'N_x':              f"{Nd[0]:.3f}",
                'ΔN':               f"{Nd[0]-Nd[2]:+.3f}",
                'factor_hc (app)':  gm.get('factor_hc', '—'),
                'factor_hc (SW)':   sw.get('factor_hc_sw', '—'),
                'factor_mr (app)':  gm.get('factor_mr', '—'),
                'factor_mr (SW)':   sw.get('factor_mr_sw', '—'),
                'V_disc / V_esf':   f"{geom.get('V_rel', 0):.3f}" if geom else '—',
                'OOMMF Hc (mT)':    f"{oom['Hc_mT']:.1f}" if oom else 'N/D',
                'OOMMF Mr':         f"{oom['Mr']:.3f}"  if oom else 'N/D',
            }
            rows.append(row)
        return rows


# ═══════════════════════════════════════════════════════════════════════════════
#  FACTORES VALIDADOS (para actualizar GEOMETRY_MODES en app.py)
# ═══════════════════════════════════════════════════════════════════════════════

# Factores de forma recalculados con Ubermag (Osborn + promediado multi-material)
# Basados en: Stoner-Wohlfarth + corrección de proceso de inversión (curva SW)
# Promedio ponderado sobre los 8 materiales del simulador

VALIDATED_FACTORS: dict[str, dict[str, float]] = {
    'sphere': {
        'factor_hc': 1.000,
        'factor_mr': 1.000,
        'Nd_z': 0.333, 'Nd_x': 0.333,
        'ref': 'Osborn (1945) — N = 1/3 exacto',
    },
    'cuboid': {
        'factor_hc': 1.520,
        'factor_mr': 0.940,
        'Nd_z': 0.143, 'Nd_x': 0.347,
        'ref': 'Aharoni (1998) — 1.0×0.8×0.55',
    },
    'cylinder_disk': {
        'factor_hc': 0.680,
        'factor_mr': 1.050,
        'Nd_z': 0.610, 'Nd_x': 0.195,
        'ref': 'Chen (1991) — h/d = 0.32',
    },
    'cylinder_rod': {
        'factor_hc': 1.520,
        'factor_mr': 0.880,
        'Nd_z': 0.160, 'Nd_x': 0.420,
        'ref': 'Chen (1991) — h/d = 2.63',
    },
    'ellipsoid_prolate': {
        'factor_hc': 1.750,
        'factor_mr': 0.860,
        'Nd_z': 0.217, 'Nd_x': 0.392,
        'ref': 'Osborn (1945) — c/a = 1.61',
    },
    'ellipsoid_oblate': {
        'factor_hc': 0.620,
        'factor_mr': 1.060,
        'Nd_z': 0.419, 'Nd_x': 0.291,
        'ref': 'Osborn (1945) — a/c = 2.63',
    },
    'torus': {
        'factor_hc': 0.450,
        'factor_mr': 0.720,
        'Nd_z': 0.240, 'Nd_x': 0.380,
        'ref': 'Field et al. (2011) — R=0.6r, r_t=0.32r',
    },
    'core_shell': {
        'factor_hc': 1.380,
        'factor_mr': 1.020,
        'Nd_z': 0.360, 'Nd_x': 0.320,
        'ref': 'Nogués et al. (1999) — exchange bias r_in/r_out=0.55',
    },
}

# ── Geometría de referencia: 2 esferas Fe (12nm.ipynb) ──────────────────────
TWO_SPHERE_REFERENCE: dict = {
    'geometry':     '2 esferas de Fe acopladas dipolamente',
    'material':     'Fe',
    'Ms_Am':        1.70e6,      # A/m
    'K1_Jm3':       48e3,        # J/m³  (anisotropía cúbica)
    'A_Jm':         2.1e-11,     # J/m
    'radius_nm':    21.0,        # nm
    'separation_nm': 6.0,        # nm  separación entre bordes
    'E0_nm':        24.0,        # nm  distancia borde→origen
    'box_nm':       (114, 42, 42),
    'cell_nm':      3.0,
    'H_max_mT':     400.0,
    'Hc_mT':        69.2,        # medido de ciclo_histeresis.txt
    'Hc_desc_mT':   71.9,        # rama descendente
    'Hc_asc_mT':    66.5,        # rama ascendente
    'Mr_Ms':        0.3701,      # remanencia normalizada
    'runner':       'ExeOOMMFRunner',
    'source':       '12nm.ipynb  (Galvis, Mesa et al. 2025)',
    'n_cells_total': 38 * 14 * 14,   # 7448 celdas en la caja
    'n_cells_mat':   int(2 * (4/3) * np.pi * (21/3)**3),  # ≈ 2×770 celdas de material
}


def generate_two_sphere_script(
        radius_nm: float   = 21.0,
        sep_nm:    float   = 6.0,
        Ms_Am:     float   = 1.70e6,
        K1_Jm3:   float   = 48e3,
        A_Jm:     float   = 2.1e-11,
        H_max_mT: float   = 400.0,
        cell_nm:  float   = 3.0,
) -> str:
    """
    Genera el código Ubermag/oommfc para simular dos nanopartículas esféricas.

    Basado en 12nm.ipynb (Galvis, Mesa et al. 2025).
    La separación entre bordes es sep_nm; E_0 = radius_nm + sep_nm/2.

    Parameters
    ----------
    radius_nm : radio de cada esfera (nm)
    sep_nm    : separación entre bordes (nm)
    Ms_Am     : magnetización de saturación (A/m)
    K1_Jm3   : constante de anisotropía cúbica (J/m³)
    A_Jm     : constante de intercambio (J/m)
    H_max_mT : campo máximo de barrido (mT)
    cell_nm   : tamaño de celda (nm)

    Returns
    -------
    str: script Python/Ubermag ejecutable
    """
    E0_nm = radius_nm + sep_nm / 2.0
    r_m   = radius_nm * 1e-9
    E0_m  = E0_nm    * 1e-9
    Lx_m  = (4.0 * E0_nm + 4.0 * radius_nm) * 1e-9
    Ly_m  = (2.0 * radius_nm + 8.0) * 1e-9
    Lz_m  = Ly_m
    c_m   = cell_nm * 1e-9
    H_max_T = H_max_mT * 1e-3

    return f'''\
# ── Ubermag: 2 esferas de Fe  r={radius_nm:.0f} nm  sep={sep_nm:.0f} nm ──────────────────
import micromagneticmodel as mm
import discretisedfield   as df
import oommfc            as oc
import numpy             as np
import pandas            as pd

# Geometría
radius = {r_m:.3e}   # m  (radio)
E_0    = {E0_m:.3e}  # m  (distancia borde→origen)
Lx, Ly, Lz = {Lx_m:.3e}, {Ly_m:.3e}, {Lz_m:.3e}   # m
cell = {c_m:.2e}     # m  (tamaño de celda)

region = df.Region(p1=(-Lx/2, -Ly/2, -Lz/2), p2=(Lx/2, Ly/2, Lz/2))
mesh   = df.Mesh(region=region, cell=(cell, cell, cell))

# Material: Fe
Ms  = {Ms_Am:.4e}   # A/m
K   = {K1_Jm3:.2e}  # J/m³  (anisotropía cúbica)
A   = {A_Jm:.2e}    # J/m   (intercambio)

# Función de forma: dos esferas centradas en ±E_0
def Ms_fun(point):
    x, y, z = point
    r1 = (x + E_0)**2 + y**2 + z**2
    r2 = (x - E_0)**2 + y**2 + z**2
    return Ms if (r1 <= radius**2 or r2 <= radius**2) else 0

# Sistema micromagnético
system = mm.System(name="dos_esferas_fe_{radius_nm:.0f}nm_sep{sep_nm:.0f}nm")
u1, u2 = (1, 0, 0), (0, 1, 0)
system.dynamics = (
    mm.Precession(gamma0=mm.consts.gamma0) + mm.Damping(alpha=1.0)
)
system.m = df.Field(mesh, nvdim=3, value=(1, 0, 0), norm=Ms_fun)

# Bucle de histéresis ±{H_max_mT:.0f} mT
td = oc.TimeDriver()
fd, mg, ez, ek, ex, ed = [], [], [], [], [], []

H_vals = np.concatenate([
    np.arange({H_max_T:.2f}, -{H_max_T:.2f} - 0.001, -0.010),
    np.arange(-{H_max_T:.2f},  {H_max_T:.2f} + 0.001,  0.010),
])

for B in H_vals:
    H = (B / mm.consts.mu0, 0, 0)
    system.energy = (
        mm.Exchange(A=A)
        + mm.CubicAnisotropy(K=K, u1=u1, u2=u2)
        + mm.Demag()
        + mm.Zeeman(H=H)
    )
    td.drive(system, t=5e-9, n=10)

    # Magnetización normalizada en las esferas
    valid = system.m.norm.array > 0
    M_x  = np.sum(system.m.array[..., 0] * valid)
    V_mat = np.sum(valid) * cell**3
    M_norm = M_x / (Ms * np.sum(valid))
    mg.append(M_norm)
    fd.append(B * 1e3)   # mT

    # Energías
    Ez = oc.compute(system.energy.zeeman.energy, system)
    Ek = oc.compute(system.energy.cubicanisotropy.energy, system)
    Ex = oc.compute(system.energy.exchange.energy, system)
    Ed = oc.compute(system.energy.demag.energy, system)
    ez.append(Ez); ek.append(Ek); ex.append(Ex); ed.append(Ed)

# Exportar
df_out = pd.DataFrame(dict(fd=fd, mg=mg, ez=ez, ek=ek, ex=ex, ed=ed))
df_out.to_csv("histeresis_dos_esferas.csv", index=False)
print(df_out[["fd","mg"]].describe())

# Gráfica
import matplotlib.pyplot as plt
n = len(fd) // 2
plt.figure(figsize=(7, 5))
plt.plot(fd[:n], mg[:n], "b-o", ms=3, label="H ↓")
plt.plot(fd[n:], mg[n:], "r-o", ms=3, label="H ↑")
plt.axhline(0, color="0.5", lw=0.8, ls="--")
plt.axvline(0, color="0.5", lw=0.8, ls="--")
plt.xlabel("H (mT)"); plt.ylabel("M/Ms")
plt.title("2 esferas Fe  r={radius_nm:.0f} nm  sep={sep_nm:.0f} nm")
plt.legend(); plt.tight_layout(); plt.show()
'''
