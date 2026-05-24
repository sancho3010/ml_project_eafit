# Resumen ejecutivo — Predicción Saber 11

**Proyecto:** Predicción del puntaje global del examen Saber 11 (ICFES) a partir de variables socioeconómicas y de colegio.
**Equipo:** Patricia Arango, Santiago Higuita, Alexander Pelaez · Universidad EAFIT · 2026

## Pregunta de negocio

¿Qué tanto del puntaje global Saber 11 puede explicarse con variables socioeconómicas (estrato, educación de padres, acceso a tecnología) y características del entorno educativo (colegio, jornada, calendario), sin usar puntajes parciales del examen?

## Datos

Resultados Saber 11 del Portal de Datos Abiertos (recurso `kgxf-xxbe`). Después de la limpieza documentada en `notebooks/03-data-cleaning.ipynb` y `docs/datacard.md`: **3,395,834 registros**, 28 columnas, periodos 2014-2 a 2022-4.

## Modelo champion

| Atributo | Valor |
|---|---|
| Algoritmo | XGBoost |
| Hiperparámetros principales | n_estimators=949, max_depth=12, learning_rate optimizado |
| Variables de entrada | 21 columnas socioeconómicas y de colegio + 5 features derivadas + 5 agregados por municipio/depto |
| Selección | Optuna (TPE multivariado + median pruner), 40 trials por modelo, 3-fold CV |

## Resultados

### Métricas en test holdout (679,167 registros)

| Métrica | Valor |
|---|---|
| RMSE | 36.60 |
| MAE | 29.14 |
| R² | 0.460 |

El modelo explica **46% de la varianza** del puntaje global. El error medio absoluto es de **±29 puntos** sobre una escala de 0 a 500. La diferencia entre RMSE de CV (36.71) y test holdout (36.60) confirma que **no hay overfitting al CV**.

### Validaciones complementarias

| Esquema | RMSE | Lectura |
|---|---|---|
| Test holdout (random, stratified periodo) | 36.60 | Pregunta de negocio principal |
| GroupKFold por municipio (5 folds) | _por completar_ | Generalización a municipios no vistos |
| Split temporal (train ≤ 2019-4, test ≥ 2020-1) | _por completar_ | Generalización a cohortes futuras |

## Hallazgos clave

1. **Hay señal fuerte y estable** en variables socioeconómicas. Los modelos pre-baseline (regresión lineal, árbol simple) ya explicaban una parte importante; el champion la mejora marginalmente.

2. **El estrato es el predictor más fuerte**, seguido por educación de los padres y capital tecnológico (internet + computador). Los SHAP values lo confirman.

3. **El R² ≈ 0.5 representa el techo razonable** del problema con la información disponible. Para mejorarlo se necesitarían datos no presentes en el dataset público: historial académico, simulacros previos, hábitos de estudio, asistencia.

4. **Las brechas son grandes y consistentes**: estrato 6 supera a estrato 1 en ~70 puntos. La brecha urbano-rural y la brecha por acceso a internet también son significativas.

## Implicaciones

- **El modelo es útil para diagnóstico agregado** y diseño de política educativa: identificar regiones con bajo desempeño relativo, cuantificar brechas, monitorear evolución temporal.
- **No debe usarse para decisiones individuales** sobre estudiantes. Predecir un puntaje no determina el desempeño real, y usarlo así sería discriminación basada en condiciones socioeconómicas.

## Productos entregados

- Notebook final con narrativa completa (`notebooks/06-pipeline-ml.ipynb`)
- Pipeline serializado y model card (`models/champion/`)
- Aplicación Streamlit con tres páginas: predictor individual, análisis de brechas, detalle del modelo (`app/`)
- Reporte técnico completo (`reports/final_report.md`)
- Tests unitarios cubriendo transformers y serialización (`tests/`)
