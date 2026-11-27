# Model Card — MiBici Demanda Horaria (`mibici-equipo51-p1-demanda`)

Este documento sigue la estructura estándar de **Model Cards for Model Reporting** (Mitchell et al., 2019) para documentar la gobernanza, desempeño, privacidad, monitoreo y limitaciones del modelo productivo del proyecto **MiBici MLOps (Equipo 51)**.

> **Nota de gobernanza:** A diferencia del *Machine Learning Canvas* (utilizado en la fase inicial de diseño para definir propuesta de valor, fuentes y tareas), este **Model Card** documenta el comportamiento empírico, métricas segmentadas, riesgos y condiciones de operación de los modelos ya entrenados y versionados en el Model Registry.

---

## 1. Model Details (Detalles del Modelo)

* **Nombre del modelo en MLflow Registry:** `mibici-equipo51-p1-demanda`
* **Versión vigente en producción (`@champion`):** Versión 3 — `CatBoostRegressor` (`depth=6`, `learning_rate=0.10`, `iterations=200`, `random_seed=777`).
* **Versión candidata en evaluación (`@challenger`):** Versión 5 — `CatBoostRegressor` (`depth=7`, `learning_rate=0.08`, `iterations=220`, `l2_leaf_reg=4.0`).
* **Modelos complementarios en el repositorio:**
  * `mibici-equipo51-p2-riesgo` (`LogisticRegression` balanceada multiclase, `C=1.0`) para clasificación de riesgo operativo a +2 horas.
  * Segmentación no supervisada de estaciones (`K-Means`, `k=4`, `n_init=10`) para perfiles de uso de 471 cicloestaciones.
* **Desarrolladores:** Equipo 51 — Ariadna del Carmen Tapia Miranda, Gabriela Díaz Oropeza e Ingrid Hernández Carrasco (ITESM — TC5061 Operaciones de Aprendizaje Automático, noviembre de 2026).
* **Licencia y atribución:** Código bajo licencia MIT (`LICENSE`). Datos abiertos públicos de MiBici Guadalajara con atribución académica.
* **Repositorio y artefactos:** [TC5061-Proyecto-Equipo51](https://github.com/arictamir/TC5061-Proyecto-Equipo51) | API REST FastAPI (`POST /predict`, `POST /predict/batch`).

---

## 2. Intended Use (Uso Previsto y Fuera de Alcance)

* **Uso principal previsto:** Predecir el número de viajes iniciados por estación y hora (`trips_started`) en las estaciones activas de MiBici Guadalajara para apoyar la planeación de logística y rebalanceo de bicicletas y anclajes.
* **Usuarios previstos:** Equipos de análisis operativo, planeación logística y sistemas de monitoreo MLOps.
* **Usos fuera de alcance:**
  * Toma de decisiones automatizada sin supervisión humana.
  * Inferencia o perfilamiento sobre usuarios individuales.
  * Despliegue directo en otras ciudades o sistemas de movilidad sin reentrenamiento y recalibración local.

---

## 3. Factors (Factores y Subgrupos Relevantes)

El comportamiento del modelo varía según las siguientes dimensiones operativas:
* **Densidad de demanda:** El 53.40 % de las observaciones estación-hora tienen demanda cero (madrugadas y zonas periféricas), frente a horas pico con alta actividad ($y > 0$).
* **Ciclos temporales:** Hora del día (`hour_of_day`), día de la semana (`day_of_week`), fines de semana (`is_weekend`) y estacionalidad mensual (`month`).
* **Expansión de la red:** Incorporación de nuevas cicloestaciones durante 2026 (escenarios *cold-start* con menor historial acumulado en `history_hours`).

---

## 4. Metrics & Quantitative Analyses (Métricas y Evaluación)

### 4.1 Selección Inicial (Partición Temporal P1)
* **Baseline (`Seasonal Naive`, `lag_168`):** Val MAE = `0.9979`, Val RMSE = `2.0318`.
* **`@champion` v3 (`CatBoostRegressor`):** Val MAE = **`0.7958`**, Val RMSE = **`1.4425`** (**-29.00 %** vs. baseline), Test MAE = **`0.7827`**, Test RMSE = **`1.4206`**.

### 4.2 A/B Testing en Periodo Productivo (`D_curr`: mayo–septiembre 2026, 50,000 muestras pareadas)
Evaluado mediante `mibici_mlops/modeling/ab_testing.py`:

| Métrica | `@champion` (v3) | `@challenger` (v5) | Resultado |
| :--- | :---: | :---: | :--- |
| **RMSE Global** | **1.4285** | 1.4360 | Mejor `@champion` |
| **MAE Global** | **0.7889** | 0.8144 | Mejor `@champion` |
| **MAE en Demanda Positiva ($y > 0$)** | 1.3291 | **1.3090** | Mejor `@challenger` (-1.5 %) |
| **RMSE en Demanda Positiva ($y > 0$)** | 2.0336 | **2.0120** | Mejor `@challenger` (-1.1 %) |
| **Sesgo Medio ($\mu_{error}$)** | +0.1206 | **+0.0024** | Menor sesgo `@challenger` |
| **Latencia / 1,000 inferencias** | 3.21 ms | **2.37 ms** | Más rápido `@challenger` |

* **Significancia estadística (Paired t-test):** $t = -18.4070$, $p = 2.05 \times 10^{-75}$ ($p \ll 0.05$).
* **Decisión vigente:** Se mantiene **Versión 3 (`@champion`)** en el endpoint `/predict` porque minimiza el error global (RMSE y MAE) dada la alta proporción de horas sin viajes en la red.

### 4.3 Rendimiento Operativo y FinOps
* **Pruebas de carga (Locust, 100 usuarios concurrentes):** `272.86 RPS` (`569.05 inferencias/s` considerando lotes), latencia `p95 = 94 ms` (< 500 ms SLA), `0.0 %` de errores.
* **Costo estimado por inferencia:** **`$0.0313 USD` por millón de inferencias** en AWS ECS Fargate (`$0.0290 USD` en EC2 `t3.medium`).

---

## 5. Training & Evaluation Data (Datos de Entrenamiento y Evaluación)

* **Fuente:** Portal de datos abiertos de MiBici Guadalajara (enero de 2024 a septiembre de 2026, 33 archivos mensuales, trazabilidad SHA-256 en `data_manifest.json`).
* **Partición cronológica estricta (sin fuga temporal):**
  * **Entrenamiento:** Enero 2024 – diciembre 2025 (`6,401,364` filas).
  * **Validación:** Enero – abril 2026 (`1,068,480` filas).
  * **Prueba (`D_curr`):** Mayo – septiembre 2026 (`1,577,073` filas).
* **Variables de entrada (11 features):** `station_id`, `hour_of_day`, `day_of_week`, `is_weekend`, `month`, `lag_1`, `lag_24`, `lag_168`, `rolling_mean_3h`, `rolling_mean_24h`, `history_hours`. Todas las métricas rezagadas aplican desplazamiento temporal (`shift(1)`) previo para evitar fuga de datos.

---

## 6. Monitoring & Data Drift (Evidently)

Evaluado con `mibici_mlops/monitoring.py` comparando `D_ref` (2024–2025) frente a `D_curr` (mayo–septiembre 2026):
* **Deriva global:** No detectada (`4` de `12` variables con deriva = `33.3 %`, por debajo del umbral del `50.0 %`).
* **Calidad de datos:** `0` valores nulos en ambos periodos.
* **Variables con deriva localizada ($p < 0.05$):** `month`, `history_hours`, `station_id` y `rolling_mean_24h` (atribuibles a estacionalidad de lluvias, acumulación temporal y expansión de estaciones).

---

## 7. Ethical Considerations & Privacy (Ética y Privacidad)

* **Exclusión total de PII:** Las columnas `usuario_id`, `genero` y `ano_de_nacimiento` de los archivos abiertos se descartan en la ingesta; nunca ingresan al entrenamiento, al servicio FastAPI ni a MLflow.
* **Granularidad agregada:** El sistema opera únicamente a nivel estación-hora.
* **Equidad territorial:** Se monitorea el error de forma diferenciada entre estaciones céntricas de alta demanda y estaciones periféricas o recientes para evitar subestimaciones sistemáticas.

---

## 8. Caveats and Recommendations (Limitaciones y Recomendaciones)

* **Variables exógenas ausentes:** El modelo no incorpora pronósticos meteorológicos (lluvia), días festivos, eventos masivos ni cierres viales; estos factores pueden generar picos no explicados por el historial.
* **Backend de MLflow en producción:** En pruebas locales con `docker stats`, el contenedor `mibici-mlflow` con SQLite alcanzó ~1013 MiB de RAM; para producción se recomienda migrar el backend a PostgreSQL administrado (RDS) y artefactos a S3.
* **Política de actualización:** Reentrenar y reevaluar promoción mediante prueba *t* pareada ($p < 0.05$) si Evidently detecta deriva en más del 40 % de las variables o si el MAE en demanda positiva supera `1.40` viajes/hora durante 7 días continuos.
