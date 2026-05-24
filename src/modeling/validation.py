"""
Validaciones de robustez complementarias al test holdout.

* ``group_kfold_cv``: evalúa el champion con ``GroupKFold`` por municipio.
  Mide qué tan bien generaliza a municipios no vistos en el train.

* ``temporal_holdout``: split temporal (entrena con periodos antiguos y
  evalúa en los recientes). Mide concept drift y la capacidad del modelo
  para extrapolar a cohortes futuras.

Ambas funciones reciben el pipeline ya tuneado (mismos hiperparámetros del
champion) y devuelven un dict con métricas listas para serializar.
"""

from __future__ import annotations

from typing import Any

from loguru import logger
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import GroupKFold


def _metrics(y_true, y_pred) -> dict[str, float]:
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def group_kfold_cv(
    build_pipeline_fn,
    X: pd.DataFrame,
    y: np.ndarray,
    group_col: str = "estu_mcpio_reside",
    n_splits: int = 5,
) -> dict[str, Any]:
    """
    ``GroupKFold`` agrupando por la columna indicada.

    Parameters:

    build_pipeline_fn : callable
        Función sin argumentos que construye un pipeline fresco (con los
        hiperparámetros del champion). Se invoca por fold para garantizar
        un fit limpio.
    X : pandas.DataFrame
        Features (incluye la columna de grupo).
    y : numpy.ndarray
        Target.
    group_col : str
        Columna que define los grupos. Por defecto municipio de residencia.
    n_splits : int
        Número de folds.
    """
    if group_col not in X.columns:
        raise ValueError(f"'{group_col}' no está en X.")

    groups = X[group_col].fillna("__missing__").to_numpy()
    gkf = GroupKFold(n_splits=n_splits)

    fold_scores: list[dict[str, float]] = []
    for i, (tr_idx, va_idx) in enumerate(gkf.split(X, y, groups), start=1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y[tr_idx], y[va_idx]

        pipe = build_pipeline_fn()
        pipe.fit(X_tr, y_tr)

        m = _metrics(y_va, pipe.predict(X_va))
        m["fold"] = i
        m["n_train"] = int(len(tr_idx))
        m["n_val"] = int(len(va_idx))
        m["n_groups_train"] = int(pd.Series(groups[tr_idx]).nunique())
        m["n_groups_val"] = int(pd.Series(groups[va_idx]).nunique())

        fold_scores.append(m)
        logger.info(
            f"GroupKFold fold {i}: RMSE={m['rmse']:.3f} " f"MAE={m['mae']:.3f} R2={m['r2']:.3f}"
        )

    df = pd.DataFrame(fold_scores)
    summary = {
        "scheme": "GroupKFold",
        "group_col": group_col,
        "n_splits": n_splits,
        "rmse_mean": float(df["rmse"].mean()),
        "rmse_std": float(df["rmse"].std()),
        "mae_mean": float(df["mae"].mean()),
        "r2_mean": float(df["r2"].mean()),
        "folds": fold_scores,
    }

    return summary


def temporal_holdout(
    build_pipeline_fn,
    X: pd.DataFrame,
    y: np.ndarray,
    period_col: str = "periodo",
    train_until: str = "20194",
) -> dict[str, Any]:
    """
    Split temporal: ``periodo <= train_until`` para entrenar, resto para evaluar.

    El identificador del periodo es del tipo ``20142`` (año + bimestre/semestre
    según ICFES). La comparación lexicográfica funciona porque el formato es
    consistente, pero la hacemos explícita por claridad.
    """
    if period_col not in X.columns:
        raise ValueError(f"'{period_col}' no está en X.")

    periods = X[period_col].astype(str)
    mask_train = periods <= train_until
    mask_test = ~mask_train

    if mask_train.sum() == 0 or mask_test.sum() == 0:
        raise ValueError(
            f"Split temporal inválido: train={mask_train.sum()}, "
            f"test={mask_test.sum()}. Revisa 'train_until={train_until}'."
        )

    X_tr, X_te = X[mask_train], X[mask_test]
    y_tr, y_te = y[mask_train.values], y[mask_test.values]

    pipe = build_pipeline_fn()
    pipe.fit(X_tr, y_tr)
    metrics = _metrics(y_te, pipe.predict(X_te))

    summary = {
        "scheme": "TemporalHoldout",
        "period_col": period_col,
        "train_until": train_until,
        "n_train": int(mask_train.sum()),
        "n_test": int(mask_test.sum()),
        "train_periods": sorted(periods[mask_train].unique().tolist()),
        "test_periods": sorted(periods[mask_test].unique().tolist()),
        **metrics,
    }
    logger.info(
        f"TemporalHoldout: train<={train_until} -> "
        f"RMSE={metrics['rmse']:.3f} MAE={metrics['mae']:.3f} "
        f"R2={metrics['r2']:.3f}"
    )

    return summary
