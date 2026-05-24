# Reproducibility Checklist

Procedimiento para reproducir el experimento Saber 11 desde cero.

## Entorno

- **Python:** 3.10 o superior (recomendado 3.12+).
- **OS:** macOS, Linux o WSL2. Probado en macOS con Apple Silicon y Linux x86_64 (SageMaker).
- **RAM mínima:** 16 GB para notebooks de exploración. **64 GB+ recomendado** para correr el notebook 06 con el dataset completo.
- **CPU:** mientras más cores mejor. Probado con 10 (M4) y 64 (c6i.16xlarge).

## Setup paso a paso

```bash
# 1. Clonar el repo
git clone https://github.com/AlexanderPelaezJimenez/ml_project_eafit.git
cd ml_project_eafit

# 2. Crear y activar venv
python -m venv .venv
source .venv/bin/activate    # Linux/macOS
# .venv\Scripts\activate     # Windows

# 3. Instalar dependencias de desarrollo
pip install -U pip
pip install -r requirements-dev.txt
pip install -e .

# 4. Verificar el setup
make check
```

`make check` corre lint (ruff), análisis de seguridad (bandit) y tests unitarios. Debe terminar verde.

## Datos

Los datos crudos del ICFES no están en el repo (`.gitignore` los excluye). Para regenerarlos:

```bash
# Notebook 01: descarga 7M+ registros desde la API SODA2
jupyter notebook notebooks/01-extract_data.ipynb

# Notebook 03: limpia y produce data/processed/cleaned_dataset.parquet
jupyter notebook notebooks/03-data-cleaning.ipynb
```

Alternativa: el repo trae `app/data/cleaned_dataset.parquet` (versión liviana usada por la app de Streamlit). Para correr el notebook 06 sin ejecutar 01 y 03, copia ese archivo a `data/processed/`:

```bash
mkdir -p data/processed
cp app/data/cleaned_dataset.parquet data/processed/cleaned_dataset.parquet
```

Hash esperado: ver `models/champion/metadata.json` (campo `data_sha256`).

## Reproducir el experimento

```bash
jupyter notebook notebooks/06-pipeline-ml.ipynb
```

**Configuración por defecto** (celda 7):

- `SEED = 42`
- `TEST_SIZE = 0.2`
- `N_TRIALS = 40` por modelo
- `CV_FOLDS = 3`
- `MODELS_TO_TUNE = ['lightGBM', 'xgboost', 'catboost']`

**Configuración de paralelismo** (celda 14): ajustar `parallel_trials` y `cores_per_model` según la máquina:

| Máquina | `parallel_trials` | `cores_per_model` |
|---|---|---|
| Mac M4 (10 cores) | 2 | 5 |
| c6i.16xlarge (64 vCPU) | 8 | 8 |

La regla de oro: `parallel_trials × cores_per_model ≈ vCPU disponibles` para evitar sobre-suscripción.

**Tiempos esperados:**

| Máquina | Tuning (3 modelos × 40 trials) | Total experimento |
|---|---|---|
| Mac M4 | demasiado lento, no recomendado | — |
| c6i.16xlarge | ~3 horas | ~3.5 horas |

## Determinismo

- **Semillas:** `SEED = 42` propagado en `train_test_split`, `KFold`, `TPESampler`, los modelos (`random_state`) y SHAP (`np.random.RandomState`).
- **Versión de paquetes:** todas pinned en `requirements-dev.txt`. Cualquier cambio puede afectar resultados marginalmente.
- **Hardware:** los gradient boosters tienen pequeñas diferencias numéricas entre arquitecturas de CPU (Intel vs ARM). Esperar diferencias del orden de 0.001 RMSE entre máquinas distintas.

## Artefactos esperados

Tras correr el notebook 06 hasta el final:

```
models/champion/
├── pipeline.joblib              # Pipeline entrenado (XGBoost + transformers)
├── metadata.json                # Autores, fecha, semilla, hash de datos, versiones
├── metrics.json                 # test_holdout, group_kfold, temporal
├── params.json                  # Hiperparámetros del champion
├── feature_schema.json          # Columnas, dtypes y dominios categóricos
├── shap_importance.csv          # Top features por |SHAP|
└── model_card.md                # Documentación legible
```

Y figuras en `reports/figures/`:

- `pred_vs_true.png`
- `residual_distribution.png`
- `residual_vs_pred.png`
- `shap_summary.png`
- `shap_importance.png`

## Validación

Métricas esperadas (XGBoost champion):

| Métrica | Valor esperado | Tolerancia |
|---|---|---|
| RMSE (test holdout) | ~36.6 | ±0.5 |
| MAE | ~29.1 | ±0.5 |
| R² | ~0.46 | ±0.02 |

Si los números difieren significativamente (>1 RMSE) revisar:

1. Versión de Python y de las librerías (debe coincidir con `requirements-dev.txt`).
2. Que los datos sean los mismos (comparar SHA256 con `models/champion/metadata.json`).
3. Que las semillas no se hayan modificado.
4. Que la máquina no esté swapeando (verificar con `top` o `htop`).

## App de Streamlit

```bash
streamlit run app/app.py
```

Levanta en `http://localhost:8501`. Si `models/champion/pipeline.joblib` no existe, la app cae a un predictor mock con mensaje de aviso.

## Hooks de calidad

```bash
make format    # auto-fix con ruff
make lint      # validación sin auto-fix
make security  # bandit
make test      # pytest
make check     # los tres anteriores
```
