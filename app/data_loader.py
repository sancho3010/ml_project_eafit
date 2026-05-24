"""
Carga del dataset agregado para la página de brechas. Si el parquet
no está disponible (deploy sin datos pesados), usa una versión sintética
con la misma estructura.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl
import streamlit as st


DATA_PATH = Path(__file__).resolve().parent.parent / "data" / "processed" / "cleaned_dataset.parquet"


@st.cache_data(show_spinner="Cargando datos agregados...")
def load_aggregates() -> pd.DataFrame:
    """
    Carga el parquet con un sample para que la UI sea fluida.
    En deploy sin datos, devuelve un dataframe sintético.
    """
    if not DATA_PATH.exists():
        return mock_aggregates()

    df = (
        pl.scan_parquet(DATA_PATH)
        .select(
            [
                "punt_global",
                "fami_estratovivienda",
                "fami_educacionmadre",
                "fami_tieneinternet",
                "cole_naturaleza",
                "cole_area_ubicacion",
                "estu_genero",
                "estu_depto_reside",
                "periodo",
            ]
        )
        .collect()
    )

    pdf = df.to_pandas()
    pdf["punt_global"] = pd.to_numeric(pdf["punt_global"], errors="coerce")
    pdf = pdf.dropna(subset=["punt_global"])

    return pdf


def mock_aggregates() -> pd.DataFrame:
    """
    Dataset sintético con tendencias plausibles para que las páginas
    de análisis funcionen sin el parquet.
    """
    rng = np.random.default_rng(42)
    n = 50_000

    estratos = ["Sin Estrato"] + [f"Estrato {i}" for i in range(1, 7)]
    estrato_probs = [0.05, 0.20, 0.30, 0.25, 0.12, 0.06, 0.02]
    estrato = rng.choice(estratos, size=n, p=estrato_probs)
    estrato_num = np.array(
        [0 if e == "Sin Estrato" else int(e[-1]) for e in estrato], dtype=float
    )

    base = 230 + 12 * estrato_num
    noise = rng.normal(0, 36, size=n)
    target = np.clip(base + noise, 0, 500).astype(int)

    return pd.DataFrame(
        {
            "punt_global": target,
            "fami_estratovivienda": estrato,
            "fami_educacionmadre": rng.choice(
                [
                    "Ninguno",
                    "Primaria completa",
                    "Secundaria (Bachillerato) completa",
                    "Educación profesional completa",
                    "Postgrado",
                ],
                size=n,
                p=[0.05, 0.25, 0.35, 0.25, 0.10],
            ),
            "fami_tieneinternet": rng.choice(["Si", "No"], size=n, p=[0.65, 0.35]),
            "cole_naturaleza": rng.choice(
                ["OFICIAL", "NO OFICIAL"], size=n, p=[0.78, 0.22]
            ),
            "cole_area_ubicacion": rng.choice(
                ["URBANO", "RURAL"], size=n, p=[0.82, 0.18]
            ),
            "estu_genero": rng.choice(["F", "M"], size=n, p=[0.52, 0.48]),
            "estu_depto_reside": rng.choice(
                ["BOGOTA D.C.", "ANTIOQUIA", "VALLE", "ATLANTICO", "CUNDINAMARCA"],
                size=n,
            ),
            "periodo": rng.choice(["20162", "20172", "20182", "20194", "20224"], size=n),
        }
    )
