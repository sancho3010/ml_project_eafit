"""
Persistencia del champion como bundle completo.

Cuando termine el experimento queremos dejar en ``models/champion/`` todo
lo necesario para que cualquiera entienda y use el modelo:

* ``pipeline.joblib``: el pipeline entrenado (deterministic + group_aggs +
  preprocessor + booster).
* ``metadata.json``: autores, fecha, semilla, hash del dataset, versiones.
* ``metrics.json``: métricas de los tres esquemas de validación.
* ``params.json``: hiperparámetros del champion.
* ``feature_schema.json``: columnas esperadas, dtypes y dominios categóricos.
* ``shap_importance.csv``: top features para consumo externo.
* ``model_card.md``: documentación legible.

Diseñamos esto pensando en la rúbrica (reproducibility) y en el deploy
posterior a Streamlit (la app carga el joblib y consulta el schema).
"""

from __future__ import annotations

import datetime as dt
import hashlib
from importlib import metadata as importlib_metadata
import json
from pathlib import Path
import platform
import sys
from typing import Any

import joblib
from loguru import logger
import pandas as pd

CRITICAL_PACKAGES = (
    "scikit-learn",
    "lightgbm",
    "xgboost",
    "catboost",
    "optuna",
    "shap",
    "pandas",
    "polars",
    "numpy",
    "joblib",
)


# ----------------------------------------------------------------------
# Utilidades
# ----------------------------------------------------------------------


def safe_version(pkg: str) -> str:
    try:
        return importlib_metadata.version(pkg)

    except importlib_metadata.PackageNotFoundError:
        return "not_installed"


def hash_file(path: str | Path, algo: str = "sha256") -> str:
    """SHA256 del contenido del archivo. Útil para fijar el dataset usado."""
    h = hashlib.new(algo)

    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)

    return h.hexdigest()


def collect_environment() -> dict[str, Any]:
    """Captura versiones de paquetes críticos y datos del runtime."""
    return {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "packages": {pkg: safe_version(pkg) for pkg in CRITICAL_PACKAGES},
    }


def feature_schema(df: pd.DataFrame, target_col: str | None = None) -> dict[str, Any]:
    """Esquema simple de columnas: dtype y dominio (para categóricas)."""
    schema = {}
    for col in df.columns:
        if target_col and col == target_col:
            continue

        s = df[col]
        info: dict[str, Any] = {"dtype": str(s.dtype)}
        if s.dtype == "object" or str(s.dtype).startswith("category"):
            uniques = s.dropna().unique().tolist()

            # Para evitar bundles enormes, recortamos dominios muy grandes.
            info["domain"] = uniques if len(uniques) <= 200 else None
            info["n_unique"] = int(len(uniques))

        schema[col] = info

    return schema


# ----------------------------------------------------------------------
# Save bundle
# ----------------------------------------------------------------------


def save_champion(
    pipeline,
    out_dir: str | Path,
    *,
    model_name: str,
    best_params: dict,
    metrics: dict[str, Any],
    shap_importance: pd.DataFrame | None,
    X_train_sample: pd.DataFrame,
    target_col: str = "punt_global",
    data_path: str | Path | None = None,
    seed: int = 42,
    extra_metadata: dict[str, Any] | None = None,
) -> Path:
    """Guarda el bundle completo en ``out_dir``.

    Parameters:

    pipeline : sklearn.pipeline.Pipeline
        El champion entrenado.
    metrics : dict
        Estructura con las métricas. Ej:
        ``{"test_holdout": {...}, "group_kfold": {...}, "temporal": {...}}``.
    shap_importance : pandas.DataFrame
        Salida de ``PipelineML.explain``. Puede ser ``None`` si SHAP
        falló o no se corrió.
    X_train_sample : pandas.DataFrame
        Muestra del DataFrame de train con las columnas originales (antes
        de las transformaciones internas del pipeline). Se usa para
        derivar el schema de inferencia.
    data_path : path-like, optional
        Si se provee, se guarda el SHA256 del dataset usado.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # 1. Pipeline serializado.
    joblib.dump(pipeline, out / "pipeline.joblib")
    logger.info(f"Pipeline guardado en {out / 'pipeline.joblib'}")

    # 2. Metadata.
    metadata = {
        "model_name": model_name,
        "created_at": dt.datetime.utcnow().isoformat() + "Z",
        "seed": seed,
        "target": target_col,
        "n_train_sample": int(len(X_train_sample)),
        "data_path": str(data_path) if data_path else None,
        "data_sha256": hash_file(data_path) if data_path else None,
        "environment": collect_environment(),
    }
    if extra_metadata:
        metadata.update(extra_metadata)
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2))

    # 3. Hiperparámetros.
    (out / "params.json").write_text(json.dumps(best_params, indent=2))

    # 4. Métricas.
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str))

    # 5. Feature schema (sirve a la app de inferencia para validar inputs).
    schema = feature_schema(X_train_sample, target_col=target_col)
    (out / "feature_schema.json").write_text(json.dumps(schema, indent=2))

    # 6. SHAP importance.
    if shap_importance is not None:
        shap_importance.to_csv(out / "shap_importance.csv", index=False)

    # 7. Model card en markdown.
    (out / "model_card.md").write_text(
        render_model_card(metadata, best_params, metrics, shap_importance)
    )

    logger.info(f"Bundle del champion guardado en {out}")
    return out


def render_model_card(
    metadata: dict,
    params: dict,
    metrics: dict,
    shap_importance: pd.DataFrame | None,
) -> str:
    lines = [
        "# Model Card | Champion",
        "",
        f"**Modelo:** {metadata['model_name']}  ",
        f"**Fecha:** {metadata['created_at']}  ",
        f"**Semilla:** {metadata['seed']}  ",
        f"**Dataset (SHA256):** `{metadata.get('data_sha256') or 'n/a'}`  ",
        "",
        "## Métricas",
        "",
        "```json",
        json.dumps(metrics, indent=2, default=str),
        "```",
        "",
        "## Hiperparámetros",
        "",
        "```json",
        json.dumps(params, indent=2),
        "```",
        "",
        "## Entorno",
        "",
        f"- Python: {metadata['environment']['python']}",
        f"- Plataforma: {metadata['environment']['platform']}",
        "- Paquetes críticos:",
    ]
    for pkg, v in metadata["environment"]["packages"].items():
        lines.append(f"  - `{pkg}=={v}`")

    if shap_importance is not None and len(shap_importance) > 0:
        lines += ["", "## Top 15 features (SHAP)", ""]

        for _, row in shap_importance.head(15).iterrows():
            lines.append(f"- `{row['feature']}` — mean |SHAP| = " f"{row['mean_abs_shap']:.4f}")

    lines += [
        "",
        "## Uso previsto",
        "",
        "Predicción del puntaje global Saber 11 a partir de variables "
        "socioeconómicas del alumno y sus padres, y de colegio. Es una herramienta académica para "
        "investigación y análisis educativo, **no** para decisiones "
        "individuales sobre estudiantes.",
        "",
        "## Limitaciones",
        "",
        "- Sesgo de selección: solo estudiantes que presentaron Saber 11.",
        "- Concept drift: datos hasta 2022; degradación esperada con el tiempo.",
        "- Las variables socioeconómicas son auto-reportadas y pueden tener "
        "sesgo de respuesta.",
        "",
    ]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------
# Load bundle
# ----------------------------------------------------------------------


def load_champion(bundle_dir: str | Path) -> dict[str, Any]:
    """Carga un bundle completo. Útil para la app de inferencia."""
    bundle = Path(bundle_dir)
    pipeline = joblib.load(bundle / "pipeline.joblib")

    metadata = json.loads((bundle / "metadata.json").read_text())
    metrics = json.loads((bundle / "metrics.json").read_text())

    params = json.loads((bundle / "params.json").read_text())
    schema = json.loads((bundle / "feature_schema.json").read_text())

    return {
        "pipeline": pipeline,
        "metadata": metadata,
        "metrics": metrics,
        "params": params,
        "schema": schema,
    }
