# 🔬 Simulador Micromagnético ML

> Simulador web interactivo de propiedades magnéticas de nanopartículas mediante Machine Learning combinado con física micromagnética y validación OOMMF/Ubermag.

---

## 📌 ¿Qué es el proyecto?

El **Simulador Micromagnético ML** es una aplicación web científica desarrollada como opción de grado en ingeniería, orientada a la predicción y simulación de las propiedades magnéticas de nanopartículas (campo coercitivo **Hc** y remanencia **Mr/Ms**) en función del material, tamaño y geometría.

Combina tres capas de cómputo:

1. **Machine Learning (Ensemble RF + GBR + MLP):** predicción rápida de Hc y Mr con cuantificación de incertidumbre ±1σ usando 7 características físicas motivadas por la teoría micromagnética.
2. **Física analítica (Stoner-Wohlfarth):** modelo de histéresis con factores de desmagnetización derivados de fórmulas de Osborn (1945), Aharoni (1998) y Chen (1991).
3. **Simulación micromagnética completa (OOMMF vía Ubermag):** driver completo con `oommfc` cuando Docker está disponible; respaldo analítico automático si no lo está.

### Características principales

- 8 materiales magnéticos con parámetros de literatura (Fe, Permalloy, Co, Fe₃O₄, Ni, CoFe₂O₄, BaFe₁₂O₁₉, γ-Fe₂O₃)
- 8 geometrías (esfera, cuboide, disco, barra, elipsoide prolato/oblato, toroide, núcleo-cáscara)
- Curvas de histéresis suavizadas con filtro Savitzky-Golay y bandas de incertidumbre
- Visualizaciones 3D interactivas (superficies de energía, heatmaps, stacks de histéresis, campo vectorial)
- Validación contra datos reales OOMMF del sistema de referencia (2 esferas Fe, r=21 nm)
- Carga dinámica de archivos `.txt` y `.ipynb` para entrenamiento y calibración automática
- Aprendizaje en línea: el modelo se recalibra con datos reales aportados por el usuario
- Exportación de resultados a PDF científico, PNG, CSV y **formato OriginLab (.txt)**
- Persistencia de historial de simulaciones en SQLite
- Toggle de temperatura Kelvin ↔ Celsius con escalado Bloch/Callen-Callen de Ms, K₁ y A

---

## 🛠️ ¿Qué se utilizó?

### Lenguaje y framework principal

| Tecnología | Versión | Rol |
|---|---|---|
| Python | 3.11 | Lenguaje principal |
| Streamlit | ≥ 1.32 | Framework web e interfaz de usuario |

### Machine Learning y ciencia de datos

| Librería | Versión | Uso |
|---|---|---|
| scikit-learn | ≥ 1.3 | Modelos GBR, RF, MLP; preprocesamiento; validación cruzada |
| NumPy | ≥ 1.24 | Cómputo numérico, arrays, álgebra lineal |
| SciPy | ≥ 1.11 | Filtrado Savitzky-Golay, interpolación |
| pandas | ≥ 2.0 | DataFrames, consultas SQLite, exportación CSV |

### Visualización

| Librería | Versión | Uso |
|---|---|---|
| Matplotlib | ≥ 3.7 | Figura principal de histéresis y energía (backend Agg) |
| Plotly | ≥ 5.18 | Gráficas 3D interactivas, subplots OOMMF |
| Kaleido | ≥ 0.2.1 | Exportación de figuras Plotly a PNG |

### Simulación micromagnética

| Librería | Versión | Uso |
|---|---|---|
| oommfc | ≥ 0.66.0 | Interfaz Python → OOMMF (driver Docker) |
| discretisedfield | ≥ 0.92.0 | Construcción de mallas 3D discretizadas |
| micromagneticmodel | ≥ 0.65.0 | Definición de Hamiltonianos micromagnéticos |
| ubermag/oommf | Docker image | Motor de simulación OOMMF encapsulado |

### Persistencia y reportes

| Librería / Tecnología | Uso |
|---|---|
| SQLite (stdlib) | Historial de simulaciones persistente entre sesiones |
| ReportLab ≥ 4.1 | Generación de reportes PDF científicos |

### Infraestructura

| Tecnología | Uso |
|---|---|
| Docker + Docker Compose | Contenedorización y despliegue reproducible |
| Python 3.11-slim (imagen base) | Imagen ligera de producción |

---

## 🚀 ¿Cómo se despliega?

### Requisitos previos

- [Docker Desktop](https://www.docker.com/products/docker-desktop/) (o Docker Engine en Linux)
- Docker Compose v2+
- Git

### Despliegue con Docker Compose (recomendado)

```bash
# 1. Clonar el repositorio
git clone <url-del-repositorio>
cd Simulador-Micromagn-tico-ML

# 2. Construir y levantar el contenedor
docker compose up --build

# 3. Abrir en el navegador
#    http://localhost:8501
```

Para correr en segundo plano:

```bash
docker compose up --build -d

# Ver logs en vivo
docker compose logs -f

# Detener
docker compose down
```

### Activar simulación OOMMF completa

El simulador incluye respaldo analítico (Stoner-Wohlfarth) que funciona sin configuración adicional. Para activar la simulación OOMMF completa:

```bash
# Descargar la imagen OOMMF de Ubermag
docker pull ubermag/oommf

# Levantar con soporte Docker-in-Docker
docker compose up --build
```

El contenedor detecta automáticamente si OOMMF está disponible y lo usa; de lo contrario usa el modelo analítico.

### Despliegue local (sin Docker)

```bash
# Crear entorno virtual
python -m venv .venv
source .venv/bin/activate       # Linux/Mac
# .venv\Scripts\activate        # Windows

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar
streamlit run app.py
```

### Variables de entorno

| Variable | Por defecto | Descripción |
|---|---|---|
| `DB_PATH` | `/app/data/simulations.db` | Ruta de la base de datos SQLite |
| `OOMMF_RUNNER` | `docker` | Runner OOMMF: `docker` \| `tcl` \| `auto` |
| `OOMMF_DOCKER_IMAGE` | `ubermag/oommf` | Imagen Docker para OOMMF |
| `STREAMLIT_SERVER_PORT` | `8501` | Puerto de la aplicación |

### Volúmenes persistentes

| Volumen | Ruta en contenedor | Contenido |
|---|---|---|
| `./outputs` | `/app/outputs` | Figuras PNG exportadas |
| `./data` | `/app/data` | Base de datos SQLite |
| `./oommf_data` | `/app/oommf_data` | Archivos OOMMF cargados por el usuario |

---

## 📁 Estructura del proyecto

```
Simulador-Micromagn-tico-ML/
│
├── app.py                      # Aplicación principal Streamlit (8 materiales, 8 geometrías, 7 tabs)
├── ml_engine.py                # Motor ML: ensemble GBR + RF + MLP con incertidumbre
├── db.py                       # Persistencia SQLite del historial de simulaciones
├── viz3d.py                    # Visualizaciones 3D interactivas con Plotly
├── report.py                   # Generador de reportes PDF (ReportLab)
├── ubermag_validator.py        # Validación Ubermag/OOMMF, factores Nd, Stoner-Wohlfarth
├── oommf_data_manager.py       # Gestor dinámico de datos OOMMF: escaneo, clasificación, calibración
├── oommf_reference_data.py     # API de acceso a datos de referencia OOMMF
├── micromagnetic_simulator_v2.py  # Simulador CLI independiente (uso por lotes)
│
├── oommf_data/                 # Datos OOMMF de referencia
│   ├── ciclo_histeresis.txt    # Ciclo de histéresis real (Fe, 2 esferas)
│   ├── energía Zeeman.txt      # Energía Zeeman vs. campo
│   ├── energía dipolar.txt     # Energía dipolar vs. campo
│   ├── energía de intercambio.txt
│   ├── energía de anisotropía.txt
│   ├── 12nm.ipynb              # Notebook de simulación de referencia
│   └── calibration_db.json     # Base de datos de calibración ML
│
├── requirements.txt            # Dependencias Python
├── Dockerfile                  # Imagen de producción (Python 3.11-slim)
├── docker-compose.yml          # Orquestación con volúmenes y health check
└── .streamlit/
    └── config.toml             # Tema oscuro, puerto, límite de carga (50 MB)
```

---

## 🧲 Materiales y geometrías soportados

### Materiales

| Material | Símbolo | K₁ (kJ/m³) | Ms (MA/m) | Tc (K) |
|---|---|---|---|---|
| Hierro | Fe 🔴 | 48 | 1.70 | 1044 |
| Permalloy | Ni₈₀Fe₂₀ 🟣 | 0.1 | 0.86 | 843 |
| Cobalto | Co 🟡 | 450 | 1.44 | 1388 |
| Magnetita | Fe₃O₄ 🟢 | 11 | 0.48 | 858 |
| Níquel | Ni ⚪ | -5.7 | 0.49 | 627 |
| Cobaltita | CoFe₂O₄ 🔵 | 200 | 0.38 | 793 |
| Ferrita de bario | BaFe₁₂O₁₉ 🟤 | 330 | 0.38 | 740 |
| Maghemita | γ-Fe₂O₃ 🟠 | 11 | 0.40 | 820 |

### Geometrías

Esfera · Cuboide · Disco · Barra · Elipsoide prolato · Elipsoide oblato · Toroide · Núcleo-cáscara

---

## 👥 Participantes

| Nombre | Rol |
|---|---|
| **Arnol Ferney Perez** | Investigador principal · Física micromagnética · Simulaciones OOMMF |
| **Jesus Andres Cabezas** | Investigador principal · Machine Learning · Desarrollo de la aplicación |

**Institución:** Opción de Grado — Ingeniería  
**Año:** 2026  
**© SimuGOD** — Todos los derechos reservados

---

## 📄 Licencia

Proyecto académico de opción de grado. Uso educativo y de investigación.

---

## 📚 Referencias científicas

- Osborn, J.A. (1945). *Demagnetizing Factors of the General Ellipsoid*. Physical Review, 67(11-12).
- Aharoni, A. (1998). *Demagnetizing factors for rectangular ferromagnetic prisms*. Journal of Applied Physics, 83(6).
- Stoner, E.C. & Wohlfarth, E.P. (1948). *A mechanism of magnetic hysteresis in heterogeneous alloys*. Phil. Trans. R. Soc.
- Bloch, F. (1930). *Zur Theorie des Ferromagnetismus*. Zeitschrift für Physik.
- Callen, H.B. & Callen, E. (1966). *The present status of the temperature dependence of magnetocrystalline anisotropy*. Journal of Physics and Chemistry of Solids.
- Beg, M. et al. (2017). *Ubermag: Toward More Effective Micromagnetic Workflows*. IEEE Magnetics Letters.
