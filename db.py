"""
db.py — Módulo de persistencia SQLite
Simulador Micromagnético ML · Fase 3

Almacena el historial de simulaciones entre sesiones de Streamlit.
La ruta de la base de datos se configura con la variable de entorno DB_PATH.
"""

import os
import sqlite3
from datetime import datetime
from typing import Any

# ─── Ruta de la base de datos ────────────────────────────────────────────────
DB_PATH: str = os.environ.get(
    'DB_PATH',
    os.path.join(os.path.dirname(__file__), 'data', 'simulations.db'),
)


# ─── Inicialización ──────────────────────────────────────────────────────────

def init_db() -> None:
    """Crea la base de datos y las tablas si no existen."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _connect() as conn:
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS simulations (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL,
                material      TEXT    NOT NULL,
                material_id   TEXT    NOT NULL,
                size_nm       REAL    NOT NULL,
                geometry      TEXT    NOT NULL,
                hc_sphere     REAL,
                mr_sphere     REAL,
                hc_cuboid     REAL,
                mr_cuboid     REAL,
                noise_level   REAL    DEFAULT 0.008,
                field_max     REAL,
                extrapolation INTEGER DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS reports (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp   TEXT NOT NULL,
                material_id TEXT NOT NULL,
                size_nm     REAL NOT NULL,
                filename    TEXT NOT NULL
            );
        """)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ─── Simulaciones ─────────────────────────────────────────────────────────────

def save_simulation(
    material: str, material_id: str, size_nm: float, geometry: str,
    hc_sphere: float | None, mr_sphere: float | None,
    hc_cuboid: float | None, mr_cuboid: float | None,
    noise_level: float = 0.008, field_max: float = 600,
    extrapolation: bool = False,
) -> int:
    """Inserta una simulación y retorna el id asignado."""
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO simulations
               (timestamp, material, material_id, size_nm, geometry,
                hc_sphere, mr_sphere, hc_cuboid, mr_cuboid,
                noise_level, field_max, extrapolation)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?)""",
            (datetime.now().isoformat(), material, material_id,
             size_nm, geometry, hc_sphere, mr_sphere,
             hc_cuboid, mr_cuboid, noise_level, field_max,
             int(extrapolation)),
        )
        return cur.lastrowid


def get_all_simulations() -> list[dict[str, Any]]:
    """Retorna todas las simulaciones ordenadas de más reciente a más antigua."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM simulations ORDER BY id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def get_simulations_by_material(material_id: str) -> list[dict[str, Any]]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM simulations WHERE material_id=? ORDER BY size_nm",
            (material_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_stats() -> dict[str, Any]:
    """Estadísticas globales de la base de datos."""
    with _connect() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*)                    AS total,
                COUNT(DISTINCT material_id) AS unique_materials,
                MIN(size_nm)                AS min_size,
                MAX(size_nm)                AS max_size,
                ROUND(AVG(hc_sphere), 2)    AS avg_hc_sphere,
                ROUND(AVG(mr_sphere), 3)    AS avg_mr_sphere,
                SUM(extrapolation)          AS total_extrapolations
            FROM simulations
        """).fetchone()
    return dict(row) if row else {}


def clear_simulations() -> None:
    """Borra todo el historial de simulaciones."""
    with _connect() as conn:
        conn.execute("DELETE FROM simulations")


# ─── Reportes ────────────────────────────────────────────────────────────────

def log_report(material_id: str, size_nm: float, filename: str) -> None:
    with _connect() as conn:
        conn.execute(
            "INSERT INTO reports (timestamp, material_id, size_nm, filename) VALUES (?,?,?,?)",
            (datetime.now().isoformat(), material_id, size_nm, filename),
        )


def get_report_count() -> int:
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]


# ─── Auto-init al importar ───────────────────────────────────────────────────
init_db()
