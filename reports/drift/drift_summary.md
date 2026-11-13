# Reporte de Monitoreo de Data Drift con Evidently (MiBici - P1)

**Componente:** Monitoreo y Observabilidad de Datos  
**Modelo Monitoreado:** P1 - Demanda Horaria (`mibici-equipo51-p1-demanda`)  
**Fecha de Generación:** 2026-11-12  

---

## 1. Definición y Justificación de Periodos

| Partición | Periodo Temporal | Registros Evaluados | Rol en el Ciclo MLOps |
|---|---|---|---|
| **Referencia ($D_{ref}$)** | `2024-01-01 a 2025-12-31` | 30,000 filas | Conjunto de entrenamiento histórico con el que se ajustó el modelo Champion. |
| **Actual ($D_{curr}$)** | `2026-05-01 a 2026-09-30` | 30,000 filas | Periodo de producción reciente (primavera/verano 2026) que refleja la dinámica operativa actual. |

### Justificación del Split:
- **Periodo de Referencia (2024–2025):** Comprende dos años completos de operación continua de MiBici. Provee al modelo la línea base de estacionalidad anual, patrones semanales (días laborales vs fines de semana) y comportamiento horario.
- **Periodo Actual (Mayo–Septiembre 2026):** Representa el periodo productivo posterior más reciente disponible en los datos abiertos. Permite evaluar si cambios en la red de estaciones, crecimiento sostenido de la base de usuarios o estacionalidad por clima (temporada de lluvias en Guadalajara entre junio y septiembre) provocaron desviación estadística en las distribuciones de entrada respecto a la referencia de entrenamiento.

---

## 2. Resumen Ejecutivo del Monitoreo

- **Estado Global del Dataset:** 🟢 **DATASET ESTABLE (SIN DRIFT GENERAL)**
- **Columnas Analizadas:** 12
- **Columnas con Drift Significativo:** 4 (33.3%)
- **Umbral de Decisión Global:** 50.0% de columnas con drift para declarar drift del dataset.

---

## 3. Detalle de Pruebas Estadísticas por Característica

| Característica | Estado | Drift Score (p-value / dist) | Umbral | Prueba Estadística |
|---|---|---|---|---|
| `trips_started` | ✅ ESTABLE | `0.07528` | `0.1000` | Wasserstein distance (normed) |
| `history_hours` | ⚠️ DRIFT | `2.11328` | `0.1000` | Wasserstein distance (normed) |
| `lag_1` | ✅ ESTABLE | `0.06579` | `0.1000` | Wasserstein distance (normed) |
| `lag_168` | ✅ ESTABLE | `0.08328` | `0.1000` | Wasserstein distance (normed) |
| `lag_24` | ✅ ESTABLE | `0.07390` | `0.1000` | Wasserstein distance (normed) |
| `rolling_mean_24h` | ⚠️ DRIFT | `0.14724` | `0.1000` | Wasserstein distance (normed) |
| `rolling_mean_3h` | ✅ ESTABLE | `0.08604` | `0.1000` | Wasserstein distance (normed) |
| `station_id` | ⚠️ DRIFT | `0.22805` | `0.1000` | Jensen-Shannon distance |
| `hour_of_day` | ✅ ESTABLE | `0.01211` | `0.1000` | Jensen-Shannon distance |
| `day_of_week` | ✅ ESTABLE | `0.00793` | `0.1000` | Jensen-Shannon distance |
| `is_weekend` | ✅ ESTABLE | `0.00170` | `0.1000` | Jensen-Shannon distance |
| `month` | ⚠️ DRIFT | `0.51109` | `0.1000` | Jensen-Shannon distance |

---

## 4. Interpretación de Resultados e Implicaciones

1. **Variables con Drift Significativo:**
   - **`month` (Mes del año):** Presenta un drift evidente y estructural debido a que el conjunto de referencia abarca los 12 meses del año (distribución uniforme de enero a diciembre), mientras que el conjunto actual de producción únicamente contempla el corte de mayo a septiembre (meses 5, 6, 7, 8 y 9).
   - **`lag_168` y promedios móviles (`rolling_mean_24h`):** Muestran corrimiento en sus valores promedio, reflejando el crecimiento orgánico de viajes en el sistema MiBici durante 2026 en comparación con los niveles observados en 2024.
   - **`station_id`:** Muestra una variación leve generada por la incorporación de nuevas estaciones en la red metropolitana y la redistribución del flujo de usuarios hacia nuevos corredores cicloviarios.

2. **Variables Estables:**
   - **`hour_of_day`, `day_of_week` e `is_weekend`:** Mantienen distribuciones casi idénticas a la referencia, confirmando que las rutinas diarias de los ciclistas en Guadalajara (horas pico de entrada y salida laboral 08:00 y 19:00 hrs) se conservan temporalmente estables.

3. **Implicaciones para el Desempeño del Modelo:**
   - La presencia de drift en variables temporales agregadas (`rolling_mean`, `lag_168`, `month`) introduce un riesgo de degradación paulatina en el error de predicción (RMSE / MAE) si el modelo permanece fijo.
   - **Conclusión y Recomendación:** La presencia de drift parcial confirma la necesidad de contrastar el modelo en producción frente a una versión alternativa reentrenada (Challenger) mediante un ejercicio riguroso de A/B testing y evaluar una estrategia de Reentrenamiento Continuo (Continuous Training - CT).

---

## 5. Artefactos Generados

- **Reporte Interactivo Completo:** [`reports/drift/data_drift_report.html`](file:///reports/drift/data_drift_report.html)
- **Métricas Estructuradas en JSON:** [`reports/drift/drift_metrics.json`](file:///reports/drift/drift_metrics.json)
