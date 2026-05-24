"""Predictor individual: formulario con las variables socioeconómicas y
del colegio, devuelve el puntaje estimado del estudiante."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import altair as alt
import pandas as pd
import streamlit as st

from app.champion_loader import load_champion, predict
from app.components import (
    apply_theme,
    disclaimer,
    kpi_row,
    page_header,
    section_title,
)


st.set_page_config(
    page_title="Predictor — Saber 11",
    layout="wide",
)
apply_theme()
bundle = load_champion()

page_header(
    "Predictor de puntaje Saber 11",
    "Ingresa el contexto del estudiante para obtener un puntaje estimado "
    "y su banda de incertidumbre.",
)


# ----------------------------------------------------------------------
# Formulario
# ----------------------------------------------------------------------

with st.form("predictor"):
    st.markdown("### Contexto del estudiante")
    col1, col2, col3 = st.columns(3)
    with col1:
        estrato = st.selectbox(
            "Estrato de la vivienda",
            [
                "Sin Estrato",
                "Estrato 1",
                "Estrato 2",
                "Estrato 3",
                "Estrato 4",
                "Estrato 5",
                "Estrato 6",
            ],
            index=2,
        )
        genero = st.selectbox("Género", ["F", "M"])
        edad = st.number_input("Edad al presentar el examen", 14, 30, 17)

    with col2:
        edu_madre = st.selectbox(
            "Educación de la madre",
            [
                "Ninguno",
                "Primaria incompleta",
                "Primaria completa",
                "Secundaria (Bachillerato) incompleta",
                "Secundaria (Bachillerato) completa",
                "Técnica o tecnológica incompleta",
                "Técnica o tecnológica completa",
                "Educación profesional incompleta",
                "Educación profesional completa",
                "Postgrado",
            ],
            index=4,
        )
        edu_padre = st.selectbox(
            "Educación del padre",
            [
                "Ninguno",
                "Primaria incompleta",
                "Primaria completa",
                "Secundaria (Bachillerato) incompleta",
                "Secundaria (Bachillerato) completa",
                "Técnica o tecnológica incompleta",
                "Técnica o tecnológica completa",
                "Educación profesional incompleta",
                "Educación profesional completa",
                "Postgrado",
            ],
            index=4,
        )
        n_personas = st.selectbox(
            "Personas en el hogar",
            ["Dos", "Tres", "Cuatro", "Cinco", "Seis", "Siete", "Ocho", "9 o más"],
            index=2,
        )
        n_cuartos = st.selectbox(
            "Cuartos en el hogar",
            ["Uno", "Dos", "Tres", "Cuatro", "Cinco", "Seis o mas"],
            index=2,
        )

    with col3:
        internet = st.radio("Tiene internet en casa", ["Si", "No"], horizontal=True)
        computador = st.radio("Tiene computador", ["Si", "No"], horizontal=True)
        automovil = st.radio("Tiene automóvil", ["Si", "No"], horizontal=True, index=1)
        lavadora = st.radio("Tiene lavadora", ["Si", "No"], horizontal=True)

    st.markdown("### Colegio")
    col4, col5, col6 = st.columns(3)
    with col4:
        naturaleza = st.selectbox("Naturaleza", ["OFICIAL", "NO OFICIAL"])
        calendario = st.selectbox("Calendario", ["A", "B", "OTRO"])
        bilingue = st.selectbox("Bilingüe", ["N", "S"])
    with col5:
        area = st.selectbox("Área de ubicación", ["URBANO", "RURAL"])
        jornada = st.selectbox(
            "Jornada", ["MAÑANA", "TARDE", "COMPLETA", "NOCHE", "SABATINA", "UNICA"]
        )
        caracter = st.selectbox(
            "Carácter", ["ACADÉMICO", "TÉCNICO", "TÉCNICO/ACADÉMICO"]
        )
    with col6:
        cole_genero = st.selectbox("Género del colegio", ["MIXTO", "FEMENINO", "MASCULINO"])
        depto = st.selectbox(
            "Departamento de residencia",
            [
                "BOGOTA D.C.",
                "ANTIOQUIA",
                "VALLE",
                "ATLANTICO",
                "CUNDINAMARCA",
                "BOLIVAR",
                "SANTANDER",
                "OTRO",
            ],
        )

    submitted = st.form_submit_button("Calcular puntaje", type="primary")


# ----------------------------------------------------------------------
# Resultado
# ----------------------------------------------------------------------

if submitted:
    features = {
        "fami_estratovivienda": estrato,
        "fami_educacionmadre": edu_madre,
        "fami_educacionpadre": edu_padre,
        "fami_personashogar": n_personas,
        "fami_cuartoshogar": n_cuartos,
        "fami_tieneinternet": internet,
        "fami_tienecomputador": computador,
        "fami_tieneautomovil": automovil,
        "fami_tienelavadora": lavadora,
        "estu_genero": genero,
        "estu_fechanacimiento": f"01/01/{2024 - int(edad)}",
        "estu_depto_reside": depto,
        "estu_mcpio_reside": depto,
        "estu_depto_presentacion": depto,
        "estu_mcpio_presentacion": depto,
        "estu_tipodocumento": "TI",
        "cole_naturaleza": naturaleza,
        "cole_area_ubicacion": area,
        "cole_calendario": calendario,
        "cole_jornada": jornada,
        "cole_genero": cole_genero,
        "cole_caracter": caracter,
        "cole_bilingue": bilingue,
        "cole_sede_principal": "S",
        "cole_depto_ubicacion": depto,
        "cole_mcpio_ubicacion": depto,
        "periodo": "20224",
    }

    result = predict(bundle, features)
    estimate = result["punt_global"]

    section_title("Resultado")
    kpi_row(
        [
            ("Puntaje estimado", f"{estimate:.0f}", "rango 0–500"),
            (
                "Banda de incertidumbre",
                f"{result['lower']:.0f} – {result['upper']:.0f}",
                "± 1 RMSE",
            ),
            (
                "vs. media nacional",
                f"{estimate - 250:+.0f}",
                "media histórica ~ 250",
            ),
        ]
    )

    # Gauge horizontal: distribución nacional de fondo + banda + punto.
    # Rango 0-500 con la media nacional marcada y el puntaje del estudiante.
    background = pd.DataFrame({"start": [0], "end": [500]})
    band = pd.DataFrame(
        {"start": [result["lower"]], "end": [result["upper"]]}
    )
    point = pd.DataFrame({"x": [estimate]})
    mean_ref = pd.DataFrame({"x": [250], "label": ["media nacional ≈ 250"]})

    base = alt.Chart(background).mark_bar(
        color="#e5e7eb", height=22
    ).encode(
        x=alt.X("start:Q", title="Puntaje", scale=alt.Scale(domain=[0, 500])),
        x2="end:Q",
    )

    band_layer = alt.Chart(band).mark_bar(color="#5b8db8", height=22).encode(
        x="start:Q", x2="end:Q",
        tooltip=[
            alt.Tooltip("start:Q", title="Banda inferior", format=".0f"),
            alt.Tooltip("end:Q", title="Banda superior", format=".0f"),
        ],
    )

    point_layer = alt.Chart(point).mark_point(
        color="#1f4e79", filled=True, size=420, shape="diamond"
    ).encode(
        x="x:Q",
        tooltip=[alt.Tooltip("x:Q", title="Puntaje estimado", format=".0f")],
    )

    mean_rule = alt.Chart(mean_ref).mark_rule(
        color="#a13a3a", strokeDash=[4, 3]
    ).encode(x="x:Q")
    mean_text = alt.Chart(mean_ref).mark_text(
        align="left", dx=6, dy=-12, color="#a13a3a", fontSize=11
    ).encode(x="x:Q", text="label:N")

    chart = (
        (base + band_layer + point_layer + mean_rule + mean_text)
        .properties(height=120)
        .configure_axisY(title=None, labels=False, ticks=False, domain=False)
        .configure_view(strokeWidth=0)
    )
    st.altair_chart(chart, use_container_width=True)

    st.caption(
        "La banda de incertidumbre representa el RMSE del modelo en test holdout. "
        "Aproximadamente el 68% de las predicciones reales caen dentro de esa banda."
    )

disclaimer()
