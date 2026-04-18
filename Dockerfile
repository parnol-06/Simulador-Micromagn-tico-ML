# ─────────────────────────────────────────────────────────────────────────────
#  Micromagnetic ML Simulator — Phase 5
#  Dockerfile · Python 3.11-slim · Streamlit  (OOMMF optional via volume)
#
#  Multi-stage build:
#    builder  — installs gcc / build tools, compiles C extensions
#    runtime  — lean final image, no build tools
#
#  Usage (no OOMMF):
#    docker build -t micromag-sim .
#    docker run -p 8501:8501 -v $(pwd)/data:/app/data micromag-sim
#
#  Usage (with OOMMF via Docker-in-Docker):
#    docker run -p 8501:8501 \
#      -v /var/run/docker.sock:/var/run/docker.sock \
#      -e OOMMF_RUNNER=docker \
#      micromag-sim
# ─────────────────────────────────────────────────────────────────────────────

# ── Stage 1: builder ──────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY requirements.txt .
# Install core deps into a prefix directory for easy copying to runtime
RUN pip install --upgrade pip && \
    pip install --prefix=/install -r requirements.txt


# ── Stage 2: runtime ─────────────────────────────────────────────────────────
FROM python:3.11-slim AS runtime

LABEL maintainer="Andres"
LABEL description="Micromagnetic ML Simulator — Streamlit Phase 5"
LABEL version="5.1"

# Runtime system libraries only (no gcc, no chromium)
RUN apt-get update && apt-get install -y --no-install-recommends \
        libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed Python packages from builder
COPY --from=builder /install /usr/local

# Environment
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false \
    STREAMLIT_SERVER_PORT=8501 \
    STREAMLIT_SERVER_ADDRESS=0.0.0.0 \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_THEME_BASE=dark \
    # OOMMF_RUNNER options:
    #   'none'   — disables OOMMF entirely (default; app works without it)
    #   'docker' — Docker-in-Docker; mount /var/run/docker.sock at runtime
    #   'tcl'    — local Tcl/OOMMF installation inside the container
    OOMMF_RUNNER=none \
    OOMMF_DOCKER_IMAGE=ubermag/oommf \
    DB_PATH=/app/data/simulations.db

# Non-root user (security best practice)
RUN useradd -m -u 1001 appuser

WORKDIR /app

# Application source
COPY app.py ml_engine.py micromagnetic_simulator_v2.py db.py viz3d.py \
     report.py ubermag_validator.py oommf_data_manager.py \
     oommf_reference_data.py temperature_model.py materials_db.py ./

# Optional: Streamlit config
COPY .streamlit/ .streamlit/

# Data directories — use volumes in production
RUN mkdir -p /app/outputs /app/data /app/oommf_data && \
    chown -R appuser:appuser /app

USER appuser

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=25s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8501/_stcore/health')" || exit 1

ENTRYPOINT ["streamlit", "run", "app.py", \
            "--server.port=8501", \
            "--server.address=0.0.0.0", \
            "--server.headless=true", \
            "--browser.gatherUsageStats=false"]
