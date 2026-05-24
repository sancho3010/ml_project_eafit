# Model Card | Champion

**Modelo:** xgboost  
**Fecha:** 2026-05-24T17:27:12.896527Z  
**Semilla:** 42  
**Dataset (SHA256):** `78a77ff3ed2da3abe7207aa6ee93bbb36f7ddd3661a19e78d0b0bee6305e6644`  

## Métricas

```json
{
  "test_holdout": {
    "model": "xgboost",
    "rmse": 36.60484313964844,
    "mae": 29.13595962524414,
    "r2": 0.4602152705192566,
    "n_test": 679167
  },
  "group_kfold": {
    "scheme": "GroupKFold",
    "group_col": "estu_mcpio_reside",
    "n_splits": 5,
    "rmse_mean": 40.06408157348633,
    "rmse_std": 0.8002699745214665,
    "mae_mean": 31.775957107543945,
    "r2_mean": 0.34020218849182127,
    "folds": [
      {
        "fold": 1,
        "rmse": 41.226,
        "mae": 32.752,
        "r2": 0.325
      },
      {
        "fold": 2,
        "rmse": 39.446,
        "mae": 31.33,
        "r2": 0.341
      },
      {
        "fold": 3,
        "rmse": 39.293,
        "mae": 31.115,
        "r2": 0.363
      },
      {
        "fold": 4,
        "rmse": 40.5,
        "mae": 31.991,
        "r2": 0.336
      },
      {
        "fold": 5,
        "rmse": 39.854,
        "mae": 31.691,
        "r2": 0.337
      }
    ]
  },
  "temporal": {
    "scheme": "TemporalHoldout",
    "period_col": "periodo",
    "train_until": "20194",
    "n_train": 2812030,
    "n_test": 583804,
    "rmse": 40.405616760253906,
    "mae": 32.58104705810547,
    "r2": 0.438906729221344
  }
}
```

## Hiperparámetros

```json
{
  "n_estimators": 949,
  "max_depth": 12,
  "learning_rate": 0.030958270854986043,
  "subsample": 0.7502020712883714,
  "colsample_bytree": 0.712774834137586,
  "reg_alpha": 1.107625468407379,
  "reg_lambda": 0.06271168846797634,
  "min_child_weight": 20,
  "gamma": 3.1083360137014024e-07
}
```

## Entorno

- Python: 3.12.13
- Plataforma: Linux-6.1.170-210.320.amzn2023.x86_64-x86_64-with-glibc2.39
- Paquetes críticos:
  - `scikit-learn==1.5.2`
  - `lightgbm==4.6.0`
  - `xgboost==3.2.0`
  - `catboost==1.2.10`
  - `optuna==4.8.0`
  - `shap==0.51.0`
  - `pandas==2.2.3`
  - `polars==1.40.1`
  - `numpy==2.4.6`
  - `joblib==1.5.3`

## Top 15 features (SHAP)

- `numeric__max_educacion_padres` — mean |SHAP| = 7.0434
- `numeric__edad_estudiante_aprox` — mean |SHAP| = 7.0190
- `binary__estu_genero` — mean |SHAP| = 4.1995
- `numeric__pct_internet_mcpio` — mean |SHAP| = 4.1990
- `nominal_low__cole_jornada_COMPLETA` — mean |SHAP| = 3.9597
- `numeric__pct_oficial_mcpio` — mean |SHAP| = 2.2659
- `numeric__capital_tecnologico` — mean |SHAP| = 2.2633
- `ordinal__fami_educacionmadre` — mean |SHAP| = 2.2068
- `nominal_low__cole_jornada_SABATINA` — mean |SHAP| = 2.0149
- `numeric__pct_estrato_alto_mcpio` — mean |SHAP| = 2.0064
- `numeric__anio_periodo` — mean |SHAP| = 1.9098
- `nominal_low__cole_jornada_NOCHE` — mean |SHAP| = 1.8823
- `numeric__extra_edad` — mean |SHAP| = 1.5273
- `numeric__indice_socioeconomico` — mean |SHAP| = 1.2701
- `ordinal__fami_educacionpadre` — mean |SHAP| = 1.2538

## Uso previsto

Predicción del puntaje global Saber 11 a partir de variables socioeconómicas del alumno y sus padres, y de colegio. Es una herramienta académica para investigación y análisis educativo, **no** para decisiones individuales sobre estudiantes.

## Limitaciones

- Sesgo de selección: solo estudiantes que presentaron Saber 11.
- Concept drift: datos hasta 2022; degradación esperada con el tiempo.
- Las variables socioeconómicas son auto-reportadas y pueden tener sesgo de respuesta.

