# Predicción de puntaje Saber 11

**Aprendizaje de Máquina Aplicado · Universidad EAFIT · 2026**

Pipeline de ML para predecir el puntaje global del examen ICFES Saber 11 a partir de variables socioeconómicas y características del entorno educativo. Incluye notebook completo, modelo champion serializado y app de Streamlit.

## Resultados

| Métrica | Test holdout (679,167 registros) |
|---|---|
| RMSE | 36.60 |
| MAE | 29.14 |
| R² | 0.460 |

Champion: **XGBoost** (40 trials con Optuna, 3-fold CV).

Detalles completos: `reports/final_report.md` y `reports/executive_summary.md`.

## Estructura del proyecto

```
ml_project_eafit/
├── notebooks/
│   ├── 01-extract_data.ipynb       Descarga del dataset (API SODA2 del ICFES)
│   ├── 02-eda.ipynb                Análisis exploratorio
│   ├── 03-data-cleaning.ipynb      Limpieza y eliminación de leakage
│   └── 06-pipeline-ml.ipynb        Notebook final con tuning, evaluación e interpretabilidad
├── src/modeling/
│   ├── constants.py                Mapeos y listas de columnas
│   ├── transformers.py             DeterministicFeatureProcessor + GroupAggregatesTransformer (LOO)
│   ├── preprocessor.py             ColumnTransformer
│   ├── pipeline.py                 PipelineML: tuning, fit, evaluate, SHAP
│   ├── validation.py               GroupKFold y temporal holdout
│   ├── diagnostics.py              Residuales, calibración y fairness por subgrupo
│   └── champion.py                 Bundle completo del champion (joblib + metadata + model card)
├── app/
│   ├── app.py                      Página de inicio
│   ├── pages/
│   │   ├── 1_Predictor.py          Predicción individual con banda de incertidumbre
│   │   ├── 2_Brechas.py            Análisis agregado de desigualdades
│   │   └── 3_Modelo.py             Métricas, hiperparámetros, SHAP y model card
│   └── data/cleaned_dataset.parquet
├── tests/                          Tests unitarios (transformers + pipeline)
├── reports/
│   ├── final_report.md
│   ├── executive_summary.md
│   ├── reporte_entrega1.md
│   └── figures/
├── docs/
│   ├── datacard.md
│   └── reproducibility_checklist.md
├── models/champion/                Pipeline serializado + bundle
├── Makefile                        make check (lint + security + tests)
├── requirements.txt                Deploy: dependencias mínimas para Streamlit Cloud
└── requirements-dev.txt            Desarrollo: tests, lint, entrenamiento, SHAP
```

## Quick start

### Para desarrollo (entrenar, testear, ejecutar el notebook)

```bash
git clone https://github.com/AlexanderPelaezJimenez/ml_project_eafit.git
cd ml_project_eafit

python -m venv .venv
source .venv/bin/activate

pip install -r requirements-dev.txt

# Validación
make check
```

### Para correr solo la app

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Correr el experimento

```bash
jupyter notebook notebooks/06-pipeline-ml.ipynb
```

Tiempo estimado en una `c6i.16xlarge` de SageMaker: ~3.5 horas.

Configuración de paralelismo en la celda 14 según la máquina (regla: `parallel_trials × cores_per_model ≈ vCPU disponibles`).

## App de Streamlit

```bash
streamlit run app/app.py
```

Tres páginas:

- **Predictor:** formulario individual con banda de incertidumbre y perfiles preestablecidos.
- **Brechas:** KPIs y gráficos de desigualdades por estrato, educación, área urbano/rural, naturaleza del colegio y departamento. Filtros por periodo y departamento.
- **Modelo:** métricas en test holdout y validaciones complementarias, hiperparámetros, top features SHAP y model card.

Si `models/champion/pipeline.joblib` no existe, la app cae a un predictor heurístico de prueba con un banner de advertencia.

## Reproducibilidad

Procedimiento completo, configuraciones por máquina, tiempos esperados y validación de resultados: `docs/reproducibility_checklist.md`.

## Calidad de código

```bash
make format    # ruff format + ruff check --fix
make lint      # ruff format --check + ruff check
make security  # bandit
make test      # pytest
make check     # lint + security + tests
```

## Equipo

- Patricia Arango
- Santiago Higuita
- Alexander Pelaez

## Limitaciones y consideraciones éticas

El modelo es una herramienta analítica para investigación y diseño de política educativa. **No debe usarse para decisiones individuales sobre estudiantes**: predecir un puntaje a partir de condiciones socioeconómicas no determina el desempeño real, y usarlo así sería discriminación. Ver sección 14 del `notebooks/06-pipeline-ml.ipynb` y `reports/final_report.md`.

## Datos

Resultados Saber 11 del Portal de Datos Abiertos del Gobierno de Colombia (recurso `kgxf-xxbe`). 7,109,704 registros originales, 3,395,834 tras limpieza. Detalles en `docs/datacard.md`.
