"""
Diagnósticos del champion: residuales, calibración condicional y fairness.

Todas las funciones reciben ``y_true``, ``y_pred`` y opcionalmente un
``DataFrame`` con las features originales del test, para segmentar por
subgrupos sociodemográficos.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

# ----------------------------------------------------------------------
# Residuales y calibración
# ----------------------------------------------------------------------


def residual_summary(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Resumen numérico de los residuales."""
    res = y_true - y_pred
    abs_res = np.abs(res)

    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
        "residual_mean": float(res.mean()),
        "residual_std": float(res.std()),
        "abs_error_p50": float(np.percentile(abs_res, 50)),
        "abs_error_p75": float(np.percentile(abs_res, 75)),
        "abs_error_p90": float(np.percentile(abs_res, 90)),
        "abs_error_p95": float(np.percentile(abs_res, 95)),
        "abs_error_p99": float(np.percentile(abs_res, 99)),
    }


def plot_predicted_vs_true(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path | None = None,
    sample: int | None = 20_000,
    seed: int = 42,
) -> None:
    """Scatter ``y_pred`` vs ``y_true`` con línea identidad."""
    if sample and len(y_true) > sample:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(y_true), sample, replace=False)
        yt, yp = y_true[idx], y_pred[idx]

    else:
        yt, yp = y_true, y_pred

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(yt, yp, alpha=0.15, s=8)
    lo = min(yt.min(), yp.min())
    hi = max(yt.max(), yp.max())

    ax.plot([lo, hi], [lo, hi], color="red", linewidth=1.2, label="identidad")
    ax.set_xlabel("Puntaje real")
    ax.set_ylabel("Puntaje predicho")
    ax.set_title("Predicho vs real (champion, test holdout)")

    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    plt.show()


def plot_residual_distribution(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path | None = None,
) -> None:
    """Histograma + KDE de residuales."""
    res = y_true - y_pred
    fig, ax = plt.subplots(figsize=(8, 4.5))

    ax.hist(res, bins=80, density=True, alpha=0.7, color="steelblue")
    ax.axvline(0, color="red", linewidth=1)
    ax.set_xlabel("Residual (real - predicho)")
    ax.set_ylabel("Densidad")
    ax.set_title(f"Distribución de residuales (mean={res.mean():.2f}, " f"std={res.std():.2f})")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    plt.show()


def plot_residual_vs_pred(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path | None = None,
    sample: int | None = 20_000,
    seed: int = 42,
) -> None:
    """Residual vs predicho para detectar heterocedasticidad."""
    if sample and len(y_true) > sample:
        rng = np.random.RandomState(seed)
        idx = rng.choice(len(y_true), sample, replace=False)
        yt, yp = y_true[idx], y_pred[idx]

    else:
        yt, yp = y_true, y_pred

    res = yt - yp
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.scatter(yp, res, alpha=0.15, s=8)
    ax.axhline(0, color="red", linewidth=1)

    ax.set_xlabel("Puntaje predicho")
    ax.set_ylabel("Residual")
    ax.set_title("Residual vs predicho")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=120, bbox_inches="tight")

    plt.show()


def calibration_by_bucket(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_buckets: int = 10,
) -> pd.DataFrame:
    """
    Error promedio por bucket de ``y_true``.

    Útil para detectar si el modelo está sistemáticamente sesgado en colas.
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    df["bucket"] = pd.qcut(df["y_true"], q=n_buckets, duplicates="drop")

    summary = df.groupby("bucket", observed=True).agg(
        n=("y_true", "size"),
        mean_true=("y_true", "mean"),
        mean_pred=("y_pred", "mean"),
        bias=("y_true", lambda s: float((s - df.loc[s.index, "y_pred"]).mean())),
        rmse=(
            "y_true",
            lambda s: float(np.sqrt(mean_squared_error(s, df.loc[s.index, "y_pred"]))),
        ),
    )

    return summary.reset_index()


# ----------------------------------------------------------------------
# Fairness por subgrupos
# ----------------------------------------------------------------------


def metrics_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: pd.Series,
    min_n: int = 30,
) -> pd.DataFrame:
    """
    RMSE/MAE/R²/n por valor de ``groups``.

    Filtra grupos con menos de ``min_n`` filas para evitar métricas ruidosas.
    """
    df = pd.DataFrame({"y_true": y_true, "y_pred": y_pred, "group": groups.values})

    rows = []
    for g, sub in df.groupby("group", dropna=False):
        if len(sub) < min_n:
            continue

        rows.append(
            {
                "group": g,
                "n": int(len(sub)),
                "rmse": float(np.sqrt(mean_squared_error(sub["y_true"], sub["y_pred"]))),
                "mae": float(mean_absolute_error(sub["y_true"], sub["y_pred"])),
                "r2": float(r2_score(sub["y_true"], sub["y_pred"]))
                if sub["y_true"].nunique() > 1
                else float("nan"),
                "mean_true": float(sub["y_true"].mean()),
                "mean_pred": float(sub["y_pred"].mean()),
                "bias": float((sub["y_true"] - sub["y_pred"]).mean()),
            }
        )

    return pd.DataFrame(rows).sort_values("rmse").reset_index(drop=True)


def fairness_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    X_test: pd.DataFrame,
    cols: list[str] | None = None,
    min_n: int = 100,
) -> dict[str, pd.DataFrame]:
    """Reporta métricas por grupo para variables sociodemográficas clave."""
    if cols is None:
        cols = [
            "fami_estratovivienda",
            "estu_genero",
            "cole_area_ubicacion",
            "cole_naturaleza",
            "estu_depto_reside",
        ]

    return {
        c: metrics_by_group(y_true, y_pred, X_test[c], min_n=min_n)
        for c in cols
        if c in X_test.columns
    }
