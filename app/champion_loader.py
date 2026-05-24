"""
Carga del champion entrenado. Si no existe aún, devuelve un stub
con métricas mock para poder iterar la UI antes de que termine el
experimento."""

from __future__ import annotations

import json
import joblib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import streamlit as st
import pandas as pd

CHAMPION_DIR = Path(__file__).resolve().parent.parent / "models" / "champion"


@dataclass
class ChampionBundle:
    pipeline: Any
    metadata: dict
    metrics: dict
    params: dict
    schema: dict
    is_mock: bool = False


@st.cache_resource
def load_champion() -> ChampionBundle:
    """Carga el bundle del champion. Si no existe, usa mock."""
    pipeline_path = CHAMPION_DIR / "pipeline.joblib"

    if not pipeline_path.exists():
        return load_mock_champion()

    pipeline = joblib.load(pipeline_path)
    metadata = json.loads((CHAMPION_DIR / "metadata.json").read_text())
    metrics = json.loads((CHAMPION_DIR / "metrics.json").read_text())
    params = json.loads((CHAMPION_DIR / "params.json").read_text())
    schema = json.loads((CHAMPION_DIR / "feature_schema.json").read_text())

    return ChampionBundle(
        pipeline,
        metadata,
        metrics,
        params,
        schema,
        is_mock=False
    )


def load_mock_champion() -> ChampionBundle:
    """Bundle de prueba con métricas plausibles. Para desarrollo de la UI."""
    return ChampionBundle(
        pipeline=None,
        metadata={
            "model_name": "lightGBM (mock)",
            "created_at": "pendiente",
            "seed": 42,
            "target": "punt_global",
            "data_sha256": "pendiente",
            "environment": {
                "python": "3.14",
                "platform": "macOS",
                "packages": {
                    "scikit-learn": "1.5.2",
                    "lightgbm": "4.6.0",
                    "shap": "0.46.0",
                },
            },
            "authors": ["Patricia Arango", "Santiago Higuita", "Alexander Pelaez"],
        },
        metrics={
            "test_holdout": {"rmse": 36.5, "mae": 28.7, "r2": 0.49, "n_test": 679167},
            "group_kfold": {
                "scheme": "GroupKFold",
                "rmse_mean": 38.1,
                "mae_mean": 30.0,
                "r2_mean": 0.45,
            },
            "temporal": {
                "scheme": "TemporalHoldout",
                "rmse": 39.2,
                "mae": 30.9,
                "r2": 0.42,
                "train_until": "20194",
            },
        },
        params={
            "n_estimators": 750,
            "max_depth": 8,
            "learning_rate": 0.045,
            "num_leaves": 64,
        },
        schema={
            "fami_estratovivienda": {
                "dtype": "object",
                "domain": [
                    "Sin Estrato",
                    "Estrato 1",
                    "Estrato 2",
                    "Estrato 3",
                    "Estrato 4",
                    "Estrato 5",
                    "Estrato 6",
                ],
            },
            "estu_genero": {"dtype": "object", "domain": ["F", "M"]},
        },
        is_mock=True,
    )


def predict_mock(features: dict) -> dict:
    """
    Heurística sencilla para que el predictor funcione antes de tener
    el champion: combinación lineal de variables socioeconómicas.

    NO es el modelo real. Solo sirve para la demo.
    """
    estrato_map = {
        "Sin Estrato": 0,
        "Estrato 1": 1,
        "Estrato 2": 2,
        "Estrato 3": 3,
        "Estrato 4": 4,
        "Estrato 5": 5,
        "Estrato 6": 6,
    }

    edu_map = {
        "Ninguno": 0,
        "Primaria incompleta": 1,
        "Primaria completa": 2,
        "Secundaria (Bachillerato) incompleta": 3,
        "Secundaria (Bachillerato) completa": 4,
        "Técnica o tecnológica incompleta": 5,
        "Técnica o tecnológica completa": 6,
        "Educación profesional incompleta": 7,
        "Educación profesional completa": 8,
        "Postgrado": 9,
    }

    estrato = estrato_map.get(features.get("fami_estratovivienda"), 2)
    edu_madre = edu_map.get(features.get("fami_educacionmadre"), 3)
    edu_padre = edu_map.get(features.get("fami_educacionpadre"), 3)

    internet = 1 if features.get("fami_tieneinternet") == "Si" else 0
    computador = 1 if features.get("fami_tienecomputador") == "Si" else 0
    bilingue = 1 if features.get("cole_bilingue") == "S" else 0

    oficial = 1 if features.get("cole_naturaleza") == "OFICIAL" else 0
    urbano = 1 if features.get("cole_area_ubicacion") == "URBANO" else 0

    base = (
        220
        + 9 * estrato
        + 4 * edu_madre
        + 3 * edu_padre
        + 8 * internet
        + 5 * computador
        + 12 * bilingue
        - 6 * oficial
        + 4 * urbano
    )
    estimate = max(0.0, min(500.0, base))

    # Banda de incertidumbre +/- 1 RMSE.
    return {
        "punt_global": estimate,
        "lower": max(0.0, estimate - 36.5),
        "upper": min(500.0, estimate + 36.5),
    }


def predict(bundle: ChampionBundle, features: dict) -> dict:
    """Punto único de predicción. Usa el modelo real si está disponible."""
    if bundle.is_mock or bundle.pipeline is None:
        return predict_mock(features)

    df = pd.DataFrame([features])
    estimate = float(bundle.pipeline.predict(df)[0])
    rmse = bundle.metrics.get("test_holdout", {}).get("rmse", 36.5)

    return {
        "punt_global": estimate,
        "lower": max(0.0, estimate - rmse),
        "upper": min(500.0, estimate + rmse),
    }
