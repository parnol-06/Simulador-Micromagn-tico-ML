# ─────────────────────────────────────────────────────────────────────────────
#  Simulador Micromagnético ML — Fase 5
#  Dockerfile · Python 3.11-slim · Streamlit + Ubermag/OOMMF
# ─────────────────────────────────────────────────────────────────────────────

FROM python:3.11-slim AS base

# Metadatos
LABEL maintainer="Andres"
LABEL description="Simulador Micromagnético ML — Aplicación Streamlit Fase 5 con Ubermag"
LABEL version="5.0"

# Variables de entorno
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_THEME_BASE=dark \
    # OOMMF runner: 'docker' usa DockerOOMMFRunner (requiere Docker-in-Docker)
    # 'tcl' usa TclOOMMFRunner si OOMMF está instalado localmente
    # 'auto' intenta Docker primero y cae a Tcl
    OOMMF_RUNNER=docker \
    OOMMF_DOCKER_IMAGE=ubermag/oommf

# ── Sistema ───────────────────────────────────────────────────────────────────
# chromium-driver  → kaleido para exportar PNG desde Plotly
# docker-cli       → DockerOOMMFRunner de oommfc (Docker-in-Docker)
RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libgomp1 \
        chromium-driver \
        docker.io \
    && rm -rf /var/lib/apt/lists/*

# ── Usuario no-root (buena práctica de seguridad) ─────────────────────────────
# Añadido al grupo docker para poder lanzar contenedores OOMMF
RUN useradd -m -u 1001 appuser && \
    usermod -aG docker appuser

# ── Directorio de trabajo ─────────────────────────────────────────────────────
WORKDIR /app

# ── Dependencias Python (capa cacheada) ───────────────────────────────────────
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# ── Código fuente ─────────────────────────────────────────────────────────────
COPY app.py .
COPY ml_engine.py .
COPY micromagnetic_simulator_v2.py .
COPY db.py .
COPY viz3d.py .
COPY report.py .
COPY ubermag_validator.py .
COPY oommf_data_manager.py .
COPY oommf_reference_data.py .

# Carpeta de configuración Streamlit
COPY .streamlit/ .streamlit/

# Datos OOMMF iniciales (ciclos de histéresis, energías, notebooks)
# La carpeta puede llegar vacía en una imagen limpia; el usuario carga archivos
# desde la UI → se escriben en el volumen montado en /app/oommf_data
COPY oommf_data/ oommf_data/

# Crear carpetas de outputs, datos SQLite y oommf_data con permisos correctos
RUN mkdir -p /app/outputs /app/data /app/oommf_data && \
    chown -R appuser:appuser /app

# Variable de entorno para la ruta de SQLite
ENV DB_PATH=/app/data/simulations.db

# ── Cambiar a usuario no-root ──────────────────────────────────────────────────
USER appuser

# ── Puerto ────────────────────────────────────────────────────────────────────
EXPOSE 8501

# ── Health check ──────────────────────────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

# ── Arranque ──────────────────────────────────────────────────────────────────
ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]
