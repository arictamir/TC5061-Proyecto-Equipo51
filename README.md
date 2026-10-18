# mibici-mlops-eq51

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

## Proyecto MiBici: propuesta y base reproducible

Este repositorio contiene el desarrollo del proyecto integrador de MLOps basado en los datos abiertos del sistema MiBici. El proyecto utiliza registros de viajes correspondientes al periodo de enero de 2024 a septiembre de 2026, equivalentes a 33 meses de información, junto con la nomenclatura de estaciones disponible para septiembre de 2026.

La ingesta de datos fue diseñada como un proceso reproducible que permite descubrir, descargar, validar, normalizar y convertir los archivos fuente a formato Parquet. Adicionalmente, se genera un manifiesto con información de los archivos y hashes SHA-256 para identificar la versión del conjunto de datos utilizada.

### Periodo de datos

Los datos se utilizan mediante una división cronológica para el desarrollo y evaluación de los modelos. Para P1, la partición temporal implementada es:

| Conjunto | Periodo | Propósito |
| --- | --- | --- |
| Entrenamiento (Train) | Enero de 2024 – diciembre de 2025 | Entrenar el modelo |
| Validación (Validation) | Enero – abril de 2026 | Evaluar configuraciones y seleccionar el modelo final |
| Prueba (Test) | Mayo – septiembre de 2026 | Evaluar una única vez el desempeño final sobre datos posteriores |

Esta estrategia respeta el orden temporal de las observaciones y reduce el riesgo de fuga de información entre los conjuntos. Las características históricas se construyen utilizando únicamente información disponible antes de la hora que se desea predecir.

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


## P1: Predicción de demanda horaria por estación

P1 aborda un problema de regresión cuyo objetivo es predecir cuántos viajes se iniciarán en cada estación durante una hora determinada. Cada observación representa una combinación estación–hora y la variable objetivo es `trips_started`.

Para construir la serie horaria se consideran como horas activas de cada estación aquellas comprendidas entre su primera y última actividad observada como origen o destino. Dentro de este periodo, las horas sin viajes iniciados se representan con demanda igual a cero.

### Características

Las características utilizadas por el modelo son:

- Variables de calendario: `hour_of_day`, `day_of_week`, `is_weekend` y `month`.
- Identificador de estación: `station_id`.
- Rezagos históricos: `lag_1`, `lag_24` y `lag_168`.
- Medias móviles: `rolling_mean_3h` y `rolling_mean_24h`.
- Historial disponible: `history_hours`.

Los rezagos y medias móviles utilizan únicamente información anterior a la hora objetivo. En particular, las medias móviles se calculan después de desplazar la serie una hora para evitar fuga de información.

### Modelos y evaluación

Se utilizó como baseline un modelo estacional ingenuo que predice la demanda utilizando el valor observado 168 horas antes (`lag_168`).

El modelo final es un `CatBoostRegressor` con 200 iteraciones, profundidad 6, `learning_rate=0.1` y semilla 42.

| Modelo | Validation MAE | Validation RMSE |
| --- | ---: | ---: |
| Seasonal Naive (`lag_168`) | 0.9979 | 2.0318 |
| CatBoostRegressor | 0.7972 | 1.4458 |

Una vez seleccionado el modelo mediante el conjunto de validación, se realizó la evaluación final sobre el conjunto de prueba:

| Métrica | Test |
| --- | ---: |
| MAE | 0.8187 |
| RMSE | 1.4499 |
| MAE en horas con demanda positiva | 1.3278 |

Los experimentos se registran en MLflow junto con sus parámetros, métricas, versión de datos, commit de Git y artefactos.

### Reproducir el entrenamiento de P1

Una vez ejecutada la ingesta y con el entorno virtual activo:

```powershell
python -m mibici_mlops.modeling.train
