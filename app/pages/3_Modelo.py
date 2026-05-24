"""Detalle del champion: métricas, hiperparámetros, top features y
metadata. Pensado para auditoría y transparencia."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import json

import pandas as pd
import streamlit as st

from app.champion_loader import CHAMPION_DIR, load_champion
from app.components import apply_theme, kpi_row, page_header, section_title


st.set_page_config(page_title="Modelo — Saber 11", layout="wide")
apply_theme()

bundle = load_champion()

page_header(
    "Detalle del modelo champion",
    "Métricas, validaciones de robustez, hiperparámetros e importancia de features.",
)

if bundle.is_mock:
    st.warning(
        "Mostrando métricas de ejemplo. El champion real aún no se ha entrenado."
    )

# ----------------------------------------------------------------------
# Métricas principales
# ----------------------------------------------------------------------

th = bundle.metrics.get("test_holdout", {})
gk = bundle.metrics.get("group_kfold", {})
tm = bundle.metrics.get("temporal", {})

section_title("Métricas")
kpi_row(
    [
        ("RMSE (test holdout)", f"{th.get('rmse', 0):.2f}", "puntos en escala 0–500"),
        ("MAE", f"{th.get('mae', 0):.2f}", "error absoluto medio"),
        ("R²", f"{th.get('r2', 0):.3f}", "varianza explicada"),
        ("n test", f"{int(th.get('n_test', 0)):,}", "registros"),
    ]
)

st.markdown("### Validaciones complementarias")
robustness = pd.DataFrame(
    [
        {
            "Esquema": "Test holdout (random, stratified periodo)",
            "RMSE": th.get("rmse"),
            "MAE": th.get("mae"),
            "R²": th.get("r2"),
        },
        {
            "Esquema": f"GroupKFold por municipio ({gk.get('n_splits', '-')})",
            "RMSE": gk.get("rmse_mean"),
            "MAE": gk.get("mae_mean"),
            "R²": gk.get("r2_mean"),
        },
        {
            "Esquema": f"Temporal (train ≤ {tm.get('train_until', '-')})",
            "RMSE": tm.get("rmse"),
            "MAE": tm.get("mae"),
            "R²": tm.get("r2"),
        },
    ]
)
st.dataframe(
    robustness.style.format({"RMSE": "{:.2f}", "MAE": "{:.2f}", "R²": "{:.3f}"}),
    use_container_width=True,
    hide_index=True,
)
st.caption(
    "El test holdout responde la pregunta de negocio principal (predecir en "
    "cohortes vistas). GroupKFold y temporal son validaciones complementarias "
    "que muestran cómo se comporta el modelo en municipios y periodos no vistos."
)

# ----------------------------------------------------------------------
# Hiperparámetros
# ----------------------------------------------------------------------

section_title("Hiperparámetros del champion")
st.json(bundle.params)

# ----------------------------------------------------------------------
# Top features (SHAP)
# ----------------------------------------------------------------------

section_title("Top features (SHAP)")
shap_path = CHAMPION_DIR / "shap_importance.csv"
if shap_path.exists():
    shap_df = pd.read_csv(shap_path).head(20)
    st.bar_chart(shap_df.set_index("feature")["mean_abs_shap"])
    st.dataframe(shap_df, use_container_width=True, hide_index=True)
else:
    st.info(
        "Las importancias SHAP se generan al final del notebook 06 y se "
        "cargan automáticamente en cuanto estén disponibles."
    )

# ----------------------------------------------------------------------
# Metadata
# ----------------------------------------------------------------------

section_title("Metadata del champion")
st.json(bundle.metadata)

card_path = CHAMPION_DIR / "model_card.md"
if card_path.exists():
    section_title("Model card")
    st.markdown(card_path.read_text())
