# mibici-mlops-eq51

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Proyecto MiBici: propuesta y base reproducible

Este repositorio contiene el desarrollo del proyecto integrador de MLOps basado en los datos abiertos del sistema MiBici. El proyecto utiliza registros de viajes correspondientes al periodo de enero de 2024 a septiembre de 2026, equivalentes a 33 meses de información, junto con la nomenclatura de estaciones disponible para septiembre de 2026.

La ingesta de datos fue diseñada como un proceso reproducible que permite descubrir, descargar, validar, normalizar y convertir los archivos fuente a formato Parquet. Adicionalmente, se genera un manifiesto con información de los archivos y hashes SHA-256 para identificar la versión del conjunto de datos utilizada.

### Periodo de datos

Los datos se utilizarán mediante una división cronológica para el desarrollo y evaluación de los modelos:

| Conjunto | Periodo | Propósito |
| --- | --- | --- |
| Entrenamiento (Train) | Enero–diciembre de 2024 | Entrenar los modelos |
| Validación (Validation) | Enero–diciembre de 2025 | Evaluar configuraciones y seleccionar hiperparámetros |
| Prueba (Test) | Enero–septiembre de 2026 | Evaluar el desempeño final sobre datos posteriores |

Esta estrategia respeta el orden temporal de las observaciones y reduce el riesgo de fuga de información entre los conjuntos. La división temporal se implementará posteriormente durante la preparación de características y el modelado.

### Fuente y resultado real de la ingesta

Los datos provienen del [portal de datos abiertos de MiBici](https://mibici.net/es/datos-abiertos/). Los enlaces exactos de los archivos fuente utilizados están registrados en `data_manifest.json`.

La ejecución del pipeline para el periodo configurado produjo:

- **33 meses de datos:** enero de 2024 a septiembre de 2026.
- **12,540,018 registros de viaje** en los archivos CSV fuente.
- **12,539,260 viajes únicos procesados** después de eliminar 758 filas exactamente duplicadas de septiembre de 2026.
- **34 archivos fuente:** 33 archivos mensuales de viajes y 1 archivo de nomenclatura.
- **473 identificadores de estación únicos** observados en los registros de viajes.
- **484 estaciones** presentes en la nomenclatura utilizada.
- **33 archivos Parquet** de viajes y 1 archivo Parquet de nomenclatura.

De los 473 identificadores observados en los viajes, 471 tienen correspondencia en la nomenclatura. Los identificadores `405` y `407`, presentes en 3,954 viajes de junio y julio de 2026, no aparecen en `nomenclatura_2026_09.csv`.

La nomenclatura contiene cuatro estaciones con capacidad igual a cero, pero ninguna presenta actividad durante el periodo analizado.

Los archivos Parquet de viajes comparten un esquema normalizado. Durante el procesamiento se eliminan las filas exactamente duplicadas y los archivos Parquet residuales que no correspondan al periodo configurado.

## Reproducir la ingesta de datos

Los archivos originales y procesados de MiBici no se almacenan en el repositorio debido a su tamaño. Estos pueden obtenerse automáticamente mediante el pipeline de ingesta.

### Requisitos

- Python 3.10

### 1. Crear un entorno virtual

```powershell
python -m venv .venv
```

### 2. Activar el entorno

En Windows PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

### 3. Instalar las dependencias

```powershell
python -m pip install -r requirements.txt
```

### 4. Ejecutar la ingesta

```powershell
python -m mibici_mlops.dataset
```

El pipeline utiliza la configuración definida en `mibici_mlops/config.py` para descargar los archivos mensuales correspondientes a enero de 2024 a septiembre de 2026 y el archivo de nomenclatura configurado para el proyecto.

Los archivos descargados y procesados se almacenan en:

- `data/raw/`: archivos CSV originales.
- `data/processed/trips/`: registros mensuales de viajes en formato Parquet.
- `data/processed/nomenclature/`: nomenclatura de estaciones procesada.

Al finalizar también se genera `data_manifest.json`, que contiene el número de registros, tamaño y hash SHA-256 de cada archivo fuente, además de un identificador `dataset_version` calculado a partir de dichos hashes.

Para la versión actual de los datos, el identificador es:

```text
f4ac5ce85e317d512068af3b3707df52f0f6af303bc8ec9ad40180c3301ab5bc
```

Si el portal de MiBici modifica posteriormente alguno de los archivos fuente, el `dataset_version` generado puede ser diferente.

## Pregunta 3 (P3): Agrupamiento y Perfiles de Uso de Estaciones (Clustering)

* **Objetivo de Negocio:** Identificar arquetipos funcionales de estaciones y analizar su distribución espacial en el Área Metropolitana de Guadalajara (AMG) para guiar decisiones de planeación urbana, expansión de polígonos y redimensionamiento estructural de infraestructura por parte de IMEPLAN y MiBici.
* **Tipo de Tarea:** Aprendizaje no supervisado (Clustering).
* **Entidad de Análisis:** Estación consolidada (`station_id`), caracterizada mediante un vector multidimensional estático derivado del historial operativo (Enero 2024 – Septiembre 2026).
* **Horizonte:** No aplica horizonte predictivo secuencial; se analiza el comportamiento estructural agregado a nivel estación.

#### 1. Ingeniería de Características (`featuresp3.py`)
A partir de los viajes y la nomenclatura oficial, se extrajo una huella funcional de 7 variables normalizadas por estación:
* **`ratio_am`**: Razón de viajes salientes respecto a entrantes en hora pico matutina (07:00–10:00). Valores altos identifican estaciones predominantemente de origen/residenciales.
* **`ratio_pm`**: Razón de viajes salientes respecto a entrantes en hora pico vespertina (18:00–21:00). Valores bajos indican zonas de atracción laboral/comercial.
* **`pct_fin_semana`**: Proporción de viajes en sábado y domingo respecto al total, separando la movilidad utilitaria de la recreativa.
* **`pct_domingo`**: Proporción de viajes dominicales, aislando el impacto directo del corredor de la Vía RecreActiva en el AMG.
* **`rotacion_capacidad`**: Volumen total de viajes atendidos (entradas + salidas) dividido entre la capacidad instalada de anclajes, midiendo la intensidad de uso físico.
* **`latitude` y `longitude`**: Coordenadas geográficas WGS84 para contextualizar la continuidad y proximidad espacial de los grupos.

#### 2. Modelo Baseline y Métricas de Evaluación
* **Modelo Baseline:** $K$-Means ($k=4$, `n_init=10`, `random_state=42`) entrenado sobre las características estandarizadas con `StandardScaler`.
* **Métricas de Evaluación:**
  * **Silhouette Score:** Cohesión y separación entre clústeres (rango $[-1, 1]$).
  * **Davies-Bouldin Index (DBI):** Similitud promedio entre cada clúster y su más cercano (valores menores indican mejor separación).
  * **Inertia:** Suma de distancias al cuadrado de las muestras a su centroide más cercano.

#### 3. Resultados del Experimento Base (MLflow Tracking)
* **Experimento:** `mibici-equipo51`
* **Run:** `p3_baseline_kmeans`
* **Estaciones analizadas:** 471 estaciones activas.
* **Resultados Obtenidos:**
  * `test_silhouette_score`: **0.2492**
  * `test_davies_bouldin_index`: **1.3458**
  * `test_inertia`: Registrada en el run
* **Artefacto:** Mapa geoespacial de asignación de clusters registrado en `reports/figures/p3_clusters_spatial.png` y vinculado al artefacto del run en MLflow.

#### 4. Comandos de Reproducción
Para ejecutar el pipeline de extracción de variables, entrenamiento y registro en MLflow:

```bash
# Ejecutar pipeline de P3 desde la raíz del repositorio
python -m mibici_mlops.modeling.trainp3

# Visualizar resultados en MLflow UI
mlflow ui
```

## Project Organization


```
├── LICENSE            <- Open-source license if one is chosen
├── Makefile           <- Makefile with convenience commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── processed      <- Datos procesados en formato Parquet.
│   └── raw            <- Archivos CSV originales de MiBici.
│
├── docs               <- A default mkdocs project; see www.mkdocs.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── pyproject.toml     <- Project configuration file with package metadata for 
│                         mibici_mlops and configuration for tools like black
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.cfg          <- Configuration file for flake8
│
└── mibici_mlops   <- Source code for use in this project.
    │
    ├── __init__.py             <- Makes mibici_mlops a Python module
    │
    ├── config.py               <- Store useful variables and configuration
    │
    ├── dataset.py              <- Scripts to download or generate data
    │
    ├── features.py             <- Code to create features for modeling
    │
    ├── modeling                
    │   ├── __init__.py 
    │   ├── predict.py          <- Code to run model inference with trained models          
    │   └── train.py            <- Code to train models
    │
    └── plots.py                <- Code to create visualizations
```

--------

