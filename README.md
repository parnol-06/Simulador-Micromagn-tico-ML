# Simulador Micromagnético con Machine Learning (Streamlit)

> Dashboard científico interactivo para explorar **nanopartículas magnéticas** con predicción ML,
> histéresis, temperatura, geometrías y visualizaciones 3D.

## Descripción

Este proyecto combina:

- **Predicción por Machine Learning (ensemble)** de propiedades del lazo de histéresis
  - **Hc** (campo coercitivo, mT)
  - **Mr/Ms** (remanencia normalizada)
- **Generación del lazo de histéresis** (modelo analítico tipo LLG simplificado) + visualizaciones 2D/3D.
- **Geometrías** con factores de forma (Nd) validados / derivados de referencias clásicas.
- **Temperatura** (entrada en **°C** o **K**) con adaptación térmica y apagado al acercarse a **Tc**.
- **Aprendizaje online (feedback)**: cada simulación agrega puntos al motor ML y es posible reentrenar.
- **Importación de energías** desde archivos `.txt` (Zeeman, intercambio, dipolar, anisotropía) para
  graficar y para extraer Hc/Mr y **entrenar** el motor ML mediante feedback.
- Persistencia de historial en **SQLite** y exportación de reportes (PDF/CSV).

---

## Vista rápida

- UI: **Streamlit**
- Backend: **Python + NumPy + scikit-learn**
- Persistencia: **SQLite**
- 3D: **Plotly**

---

## Componentes principales

- `app.py`: aplicación web (Streamlit) con dashboard, pestañas, exportación y validaciones.
- `ml_engine.py`: motor ML (ensemble) con features físicamente motivados y aprendizaje online.
- `temperature_model.py`: conversión °C/K y correcciones térmicas (tendencias físicas).
- `db.py`: persistencia SQLite (historial de simulaciones y reportes).
- `viz3d.py`: visualizaciones Plotly 2D/3D.
- `ubermag_validator.py`: validación de geometrías/factores Nd y (opcional) ejecución OOMMF.
- `report.py`: generador de reportes PDF (ReportLab).

---

## Tecnologías / librerías usadas

| Categoría | Herramientas |
|---|---|
| UI | Streamlit |
| ML | scikit-learn (GBR, RF, MLP) |
| Cálculo / datos | NumPy, Pandas |
| Gráficas 2D | Matplotlib |
| 3D interactivo | Plotly |
| Persistencia | SQLite |
| Reportes | ReportLab |
| Micromagnetismo (opcional) | Ubermag / oommfc / OOMMF |

- **Python 3.10+ / 3.11**
- **Streamlit** (UI)
- **NumPy / Pandas** (cálculo y manejo de datos)
- **scikit-learn** (ML: Gradient Boosting, Random Forest, MLP + escalado)
- **Matplotlib** (figuras 2D)
- **Plotly** (3D interactivo)
- **SQLite** (persistencia)
- **ReportLab** (PDF)
- **Ubermag / oommfc / OOMMF** (opcional) para validación y simulaciones micromagnéticas completas

---

## Funcionalidades destacadas

### 1) Predicción ML (Ensemble)
El motor ML usa un conjunto de modelos (GBR + RF + MLP) y promedia predicciones con pesos basados en
métricas internas (R²). Los features incluyen magnitudes físicas del material (Ms, K1, Tc, etc.),
además de variables térmicas reducidas.

### 2) Temperatura (°C o K)
En la barra lateral puedes seleccionar la unidad y el valor. El simulador:

- Convierte a Kelvin internamente.
- Ajusta Hc y Mr con una corrección térmica suave (tendencia física), incluyendo apagado cuando
  **T ≥ Tc**.

### 3) Importación de energías `.txt` + entrenamiento
Se pueden cargar archivos `.txt` con dos columnas (tab o espacios), con o sin encabezado, por ejemplo:

```text
fd    mg
400.0 -411539807.7401041
390.0 -416120968.52505165
...
```

Archivos soportados:
- Energía Zeeman
- Energía de intercambio
- Energía dipolar / desmagnetización
- Energía de anisotropía

Uso:
- Si activas **“Usar energías importadas…”**, se muestran en el panel de energía.
- Con el archivo **Zeeman**, el simulador estima una magnetización aproximada vía `M ∝ -dE/dH`,
  extrae **Hc** y **Mr** y permite **añadirlos como feedback** para reentrenar el motor ML.

---

## Despliegue / ejecución

### Opción recomendada (Docker Compose)

1. Construir y levantar el servicio:

```bash
docker compose up --build
```

2. Abrir en el navegador:

- http://localhost:8501

> Si necesitas ejecución en segundo plano:
>
> ```bash
> docker compose up --build -d
> ```
>
> Para detener:
>
> ```bash
> docker compose down
> ```

---

### A) Ejecutar localmente (recomendado para desarrollo)

1. Crear entorno e instalar dependencias:

```bash
python -m venv .venv
# Linux/macOS
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

2. Ejecutar Streamlit:

```bash
streamlit run app.py
```

3. Abrir en el navegador:

- http://localhost:8501

> Nota: el historial SQLite se guarda por defecto en `./data/simulations.db`.

### B) Ejecutar con Docker (sin Compose)

1. Construir imagen:

```bash
docker build -t micromagnetic-ml-sim .
```

2. Ejecutar contenedor:

```bash
docker run --rm -p 8501:8501 \
  -e DB_PATH=/app/data/simulations.db \
  -v "${PWD}/data:/app/data" \
  micromagnetic-ml-sim
```

3. Abrir:

- http://localhost:8501

#### Variables de entorno útiles
- `DB_PATH`: ruta del SQLite.
- `OOMMF_RUNNER`: `docker | tcl | auto` (según configuración).
- `OOMMF_DOCKER_IMAGE`: por defecto `ubermag/oommf`.

> Importante: ejecutar OOMMF dentro del contenedor puede requerir configuraciones extra
> (por ejemplo, Docker-in-Docker). El simulador tiene **fallback analítico** cuando
> OOMMF no está disponible.

---

## Estructura del proyecto

Estructura típica del repositorio (puede variar según el entorno):

```text
.
├── app.py                      # App principal Streamlit (UI + simulación)
├── ml_engine.py                # Motor ML (ensemble + features + feedback)
├── temperature_model.py        # Conversión °C/K + adaptación térmica
├── viz3d.py                    # Visualizaciones Plotly 2D/3D
├── ubermag_validator.py        # Validación geométrica / Nd / OOMMF (opcional)
├── report.py                   # Generador de reportes PDF
├── db.py                       # Persistencia SQLite
├── micromagnetic_simulator_v2.py   # Script CLI (versión anterior / utilidades)
├── requirements.txt            # Dependencias Python
├── Dockerfile                  # Imagen Docker para despliegue
├── docker-compose.yml          # (Opcional) Orquestación local
├── run.sh                      # (Opcional) Script de arranque
├── .streamlit/
│   └── config.toml             # Configuración Streamlit
├── data/
│   └── simulations.db          # Historial SQLite (se crea/actualiza)
├── outputs/                    # Figuras exportadas
└── docs/                       # Documentación (PRD/SRS, etc.)
```

Carpetas clave:
- `data/`: almacenamiento persistente (SQLite). Monta esta carpeta como volumen si usas Docker.
- `outputs/`: exportaciones (PNG/CSV/PDF).
- `docs/`: documentación del proyecto.

---

## Créditos y Copyright

Copyright (c) 2026 **Arnol Ferney Pérez** – **Jesus Andres Cabezas**.

---

## Referencias y posibles fuentes de inspiración

Este proyecto integra conceptos y herramientas ampliamente usadas en micromagnetismo y ML.
Algunas secciones están basadas o inspiradas en documentación pública y referencias científicas.

### Documentación / librerías
- Streamlit Docs: https://docs.streamlit.io/
- scikit-learn Docs: https://scikit-learn.org/
- NumPy / Pandas Docs: https://numpy.org/ · https://pandas.pydata.org/
- Plotly Docs: https://plotly.com/python/
- Ubermag / oommfc: https://ubermag.github.io/ · https://oommfc.readthedocs.io/
- OOMMF (NIST): https://math.nist.gov/oommf/

### Referencias físicas (Nd / anisotropía de forma)
- Osborn, J. A. “Demagnetizing factors of the general ellipsoid.” *Phys. Rev.* (1945).
- Aharoni, A. “Demagnetizing factors for rectangular ferromagnetic prisms.” *J. Appl. Phys.* (1998).
- Chen et al. (1991): fórmulas para cilindros (disco/barra) y Nd.
- Field et al. (2011): geometrías tipo toroide y tendencias de vórtice.
- Nogués et al. (1999): exchange bias / núcleo-cáscara (contexto físico).

### Datos / contexto micromagnético y materiales
- Galvis, Mesa et al. (*Results in Physics*, 2025): datasets y comportamiento de Fe.
- Galvis, Mesa, Restrepo (*Computational Materials Science*, 2024): datasets y comportamiento de Permalloy.

---

## Notas finales

- El simulador está orientado a **exploración interactiva** y enseñanza/investigación aplicada.
- Las correcciones térmicas y la extracción Hc/Mr desde energías importadas son aproximaciones
  diseñadas para conservar tendencias físicas y permitir calibración incremental mediante feedback.
- Si tienes datasets experimentales/simulados adicionales (por tamaño, temperatura, geometría),
  pueden incorporarse como feedback y/o ampliarse como datos base de entrenamiento.
