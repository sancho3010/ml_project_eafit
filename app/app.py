"""Página de inicio de la app Saber 11.

Ejecutar:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import sys
from pathlib import Path

# Asegurar que la raíz del proyecto esté en el path. Streamlit no la
# agrega por defecto cuando ejecuta archivos dentro de app/.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import streamlit as st

from app.champion_loader import load_champion
from app.components import apply_theme, page_header, section_title


st.set_page_config(
    page_title="Análisis y predicción para Saber 11",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)
apply_theme()

bundle = load_champion()

page_header(
    "Análisis y predicción para Saber 11",
    "Predicción del puntaje global del examen ICFES Saber 11 a partir "
    "de variables socioeconómicas y de colegio.",
)

if bundle.is_mock:
    st.warning(
        "El modelo entrenado aún no se ha generado. La app está corriendo "
        "con un predictor heurístico de prueba para ilustrar la interfaz. "
        "Una vez se complete el experimento del notebook 06, los resultados "
        "reales se cargarán automáticamente."
    )

col1, col2, col3 = st.columns(3)
with col1:
    section_title("Predictor")
    st.write(
        "Estima el puntaje global Saber 11 de un estudiante a partir de "
        "su contexto socioeconómico y características del colegio. "
        "Devuelve un puntaje estimado y una banda de incertidumbre."
    )
    st.page_link("pages/1_Predictor.py", label="Ir al predictor")

with col2:
    section_title("Brechas")
    st.write(
        "Análisis de las brechas educativas en Colombia: por estrato, "
        "educación de los padres, acceso a internet, área urbano/rural, "
        "naturaleza del colegio y departamento."
    )
    st.page_link("pages/2_Brechas.py", label="Ver el análisis")

with col3:
    section_title("Modelo")
    st.write(
        "Detalles del champion: métricas en test holdout y validaciones "
        "de robustez, hiperparámetros, importancia de features (SHAP) "
        "y model card completo."
    )
    st.page_link("pages/3_Modelo.py", label="Ver el modelo")

st.markdown("---")
st.markdown(
    "**Equipo:** Patricia Arango · Santiago Higuita · Alexander Pelaez  \n"
    "**Curso:** Aprendizaje de Máquina Aplicado · Universidad EAFIT · 2026  \n"
    "**Fuente de datos:** Portal de Datos Abiertos del Gobierno de Colombia "
    "(recurso `kgxf-xxbe`)."
)
