#!/usr/bin/env bash
# ─────────────────────────────────────────────────────────────────────────────
#  run.sh — Script de arranque para el Simulador Micromagnético ML
#
#  Modos:
#    ./run.sh            → Docker Compose (recomendado)
#    ./run.sh local      → Python local (sin Docker)
#    ./run.sh build      → Solo construir imagen Docker
#    ./run.sh stop       → Detener contenedor
#    ./run.sh logs       → Ver logs del contenedor
# ─────────────────────────────────────────────────────────────────────────────

set -e

APP_NAME="Simulador Micromagnético ML v4.0"
PORT=8501
IMAGE="micromagnetic-simulator:4.0"

print_header() {
  echo ""
  echo "╔══════════════════════════════════════════════════════════════╗"
  echo "║  🔬  $APP_NAME"
  echo "╚══════════════════════════════════════════════════════════════╝"
  echo ""
}

MODE="${1:-docker}"

case "$MODE" in

  docker|"")
    print_header
    echo "▶  Modo: Docker Compose"
    echo "▶  Puerto: http://localhost:$PORT"
    echo ""

    if ! command -v docker &>/dev/null; then
      echo "❌  Docker no encontrado. Instálalo desde https://docker.com"
      exit 1
    fi

    docker compose up --build
    ;;

  local)
    print_header
    echo "▶  Modo: Python local"
    echo "▶  Puerto: http://localhost:$PORT"
    echo ""

    if ! command -v python &>/dev/null && ! command -v python3 &>/dev/null; then
      echo "❌  Python no encontrado."
      exit 1
    fi

    PYTHON=$(command -v python3 || command -v python)
    echo "📦  Instalando dependencias..."
    $PYTHON -m pip install -r requirements.txt -q

    echo "🚀  Iniciando Streamlit..."
    $PYTHON -m streamlit run app.py \
      --server.port=$PORT \
      --server.headless=true \
      --browser.gatherUsageStats=false
    ;;

  build)
    print_header
    echo "▶  Construyendo imagen Docker: $IMAGE"
    docker build -t "$IMAGE" .
    echo ""
    echo "✅  Imagen construida: $IMAGE"
    echo "    Para correr: docker run -p $PORT:$PORT $IMAGE"
    ;;

  stop)
    echo "⏹  Deteniendo contenedor..."
    docker compose down
    echo "✅  Contenedor detenido."
    ;;

  logs)
    docker compose logs -f
    ;;

  *)
    echo "Uso: ./run.sh [docker|local|build|stop|logs]"
    echo ""
    echo "  docker  → Docker Compose  (por defecto)"
    echo "  local   → Python local sin Docker"
    echo "  build   → Solo construir imagen"
    echo "  stop    → Detener contenedor"
    echo "  logs    → Ver logs en tiempo real"
    exit 1
    ;;
esac
