"""
db.py — SQLite persistence module
Micromagnetic ML Simulator · Phase 5

Stores the simulation history across Streamlit sessions.
The database path is configured with the DB_PATH environment variable.

Changes vs Phase 3:
  · WAL journal mode — safe for concurrent reads while Streamlit writes
  · Index on material_id — fast filtering for per-material queries
  · Row limit on get_all_simulations() — prevents loading unbounded data
  · Removed unused hc_cuboid / mr_cuboid columns (always NULL)
  · Added get_simulations_paginated() for large datasets
  · clear_simulations() now also clears the reports table
"""

import os
import sqlite3
from datetime import datetime
from typing import Any

# ─── Database path ────────────────────────────────────────────────────────────
DB_PATH: str = os.environ.get(
    'DB_PATH',
    os.path.join(os.path.dirname(__file__), 'data', 'simulations.db'),
)

# Maximum rows returned by get_all_simulations()
MAX_ROWS: int = 500


# ─── Initialization ──────────────────────────────────────────────────────────

def init_db() -> None:
    """Create the database and tables if they do not exist."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    with _connect() as conn:
        # Enable WAL mode for safe concurrent access
        conn.execute('PRAGMA journal_mode=WAL')
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS simulations (
                id            INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp     TEXT    NOT NULL,
                material      TEXT    NOT NULL,
                material_id   TEXT    NOT NULL,
                size_nm       REAL    NOT NULL,
                geometry      TEXT    NOT NULL,
                hc_val        REAL,
                mr_val        REAL,
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

            CREATE INDEX IF NOT EXISTS idx_sim_material
                ON simulations (material_id);

            CREATE INDEX IF NOT EXISTS idx_sim_timestamp
                ON simulations (timestamp DESC);
        """)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def _connect() -> sqlite3.Connection:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    return conn


# ─── Simulations ─────────────────────────────────────────────────────────────

def save_simulation(
    material: str, material_id: str, size_nm: float, geometry: str,
    hc_sphere: float | None, mr_sphere: float | None,
    hc_cuboid: float | None = None, mr_cuboid: float | None = None,
    noise_level: float = 0.008, field_max: float = 600,
    extrapolation: bool = False,
) -> int:
    """Insert a simulation record and return the assigned id.

    Note: hc_cuboid and mr_cuboid are accepted for backward compatibility
    but are not stored (the schema uses a single hc_val / mr_val pair).
    """
    with _connect() as conn:
        cur = conn.execute(
            """INSERT INTO simulations
               (timestamp, material, material_id, size_nm, geometry,
                hc_val, mr_val, noise_level, field_max, extrapolation)
               VALUES (?,?,?,?,?,?,?,?,?,?)""",
            (datetime.now().isoformat(), material, material_id,
             size_nm, geometry, hc_sphere, mr_sphere,
             noise_level, field_max, int(extrapolation)),
        )
        return cur.lastrowid


def get_all_simulations(limit: int = MAX_ROWS) -> list[dict[str, Any]]:
    """Return simulations ordered newest-first, up to ``limit`` rows."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM simulations ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
    return [dict(r) for r in rows]


def get_simulations_paginated(
    page: int = 0, page_size: int = 50
) -> list[dict[str, Any]]:
    """Return a page of simulations (newest-first) for large datasets."""
    offset = page * page_size
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM simulations ORDER BY id DESC LIMIT ? OFFSET ?",
            (page_size, offset),
        ).fetchall()
    return [dict(r) for r in rows]


def get_simulations_by_material(material_id: str) -> list[dict[str, Any]]:
    """Return all simulations for a given material, ordered by size."""
    with _connect() as conn:
        rows = conn.execute(
            "SELECT * FROM simulations WHERE material_id=? ORDER BY size_nm",
            (material_id,),
        ).fetchall()
    return [dict(r) for r in rows]


def get_stats() -> dict[str, Any]:
    """Global database statistics."""
    with _connect() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*)                    AS total,
                COUNT(DISTINCT material_id) AS unique_materials,
                MIN(size_nm)                AS min_size,
                MAX(size_nm)                AS max_size,
                ROUND(AVG(hc_val), 2)       AS avg_hc,
                ROUND(AVG(mr_val), 3)       AS avg_mr,
                SUM(extrapolation)          AS total_extrapolations
            FROM simulations
        """).fetchone()
    return dict(row) if row else {}


def clear_simulations() -> None:
    """Delete all simulation and report records."""
    with _connect() as conn:
        conn.execute("DELETE FROM simulations")
        conn.execute("DELETE FROM reports")


# ─── Reports ─────────────────────────────────────────────────────────────────

def log_report(material_id: str, size_nm: float, filename: str) -> None:
    """Log a generated PDF report."""
    with _connect() as conn:
        conn.execute(
            "INSERT INTO reports (timestamp, material_id, size_nm, filename) VALUES (?,?,?,?)",
            (datetime.now().isoformat(), material_id, size_nm, filename),
        )


def get_report_count() -> int:
    """Return the total number of generated reports."""
    with _connect() as conn:
        return conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]


# ─── Auto-init on import ─────────────────────────────────────────────────────
init_db()
