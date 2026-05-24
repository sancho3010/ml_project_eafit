"""Análisis agregado de brechas educativas. Vista de negocio."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import altair as alt
import pandas as pd
import streamlit as st

from app.components import apply_theme, kpi_row, page_header, section_title
from app.data_loader import load_aggregates
from app.theme import COLOR_PRIMARY, COLOR_SECONDARY


st.set_page_config(page_title="Brechas — Saber 11", layout="wide")
apply_theme()

page_header(
    "Brechas educativas",
    "Diagnóstico cuantitativo de las desigualdades en el puntaje Saber 11.",
)

df = load_aggregates()

# ----------------------------------------------------------------------
# KPIs principales
# ----------------------------------------------------------------------

mean_global = df["punt_global"].mean()
n_total = len(df)

estrato_means = (
    df.groupby("fami_estratovivienda")["punt_global"].mean().reindex(
        ["Estrato 1", "Estrato 2", "Estrato 3", "Estrato 4", "Estrato 5", "Estrato 6"]
    )
)
brecha_estrato = estrato_means.iloc[-1] - estrato_means.iloc[0]

internet_means = df.groupby("fami_tieneinternet")["punt_global"].mean()
brecha_internet = internet_means.get("Si", 0) - internet_means.get("No", 0)

area_means = df.groupby("cole_area_ubicacion")["punt_global"].mean()
brecha_urbano_rural = area_means.get("URBANO", 0) - area_means.get("RURAL", 0)

kpi_row(
    [
        ("Puntaje promedio", f"{mean_global:.0f}", f"{n_total:,} registros"),
        (
            "Brecha estrato 6 vs 1",
            f"+{brecha_estrato:.0f}",
            "puntos a favor del más alto",
        ),
        (
            "Brecha internet vs sin internet",
            f"+{brecha_internet:.0f}",
            "puntos para quienes tienen acceso",
        ),
        (
            "Brecha urbano vs rural",
            f"+{brecha_urbano_rural:.0f}",
            "puntos a favor del urbano",
        ),
    ]
)

# ----------------------------------------------------------------------
# Brecha por estrato
# ----------------------------------------------------------------------

section_title("Puntaje por estrato")
chart_estrato = (
    alt.Chart(estrato_means.reset_index())
    .mark_bar(color=COLOR_PRIMARY)
    .encode(
        x=alt.X("fami_estratovivienda:N", title="Estrato", sort=None),
        y=alt.Y("punt_global:Q", title="Puntaje promedio"),
        tooltip=["fami_estratovivienda", alt.Tooltip("punt_global:Q", format=".1f")],
    )
    .properties(height=320)
)
st.altair_chart(chart_estrato, use_container_width=True)
st.caption(
    "El gradiente por estrato es uno de los predictores más fuertes y consistentes "
    "del puntaje. La diferencia entre estrato 1 y estrato 6 supera el desempeño "
    "promedio en aproximadamente media desviación estándar."
)

# ----------------------------------------------------------------------
# Educación de la madre
# ----------------------------------------------------------------------

section_title("Puntaje por nivel educativo de la madre")
edu_order = [
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
]
edu_means = (
    df.groupby("fami_educacionmadre")["punt_global"]
    .mean()
    .reindex([e for e in edu_order if e in df["fami_educacionmadre"].unique()])
    .dropna()
    .reset_index()
)
chart_edu = (
    alt.Chart(edu_means)
    .mark_bar(color=COLOR_SECONDARY)
    .encode(
        x=alt.X("punt_global:Q", title="Puntaje promedio"),
        y=alt.Y("fami_educacionmadre:N", title="Nivel educativo de la madre", sort=None),
        tooltip=[
            "fami_educacionmadre",
            alt.Tooltip("punt_global:Q", format=".1f"),
        ],
    )
    .properties(height=380)
)
st.altair_chart(chart_edu, use_container_width=True)

# ----------------------------------------------------------------------
# Naturaleza del colegio + género del estudiante
# ----------------------------------------------------------------------

col_a, col_b = st.columns(2)
with col_a:
    section_title("Oficial vs no oficial")
    nat = (
        df.groupby(["cole_naturaleza", "cole_area_ubicacion"])["punt_global"]
        .mean()
        .reset_index()
    )
    chart_nat = (
        alt.Chart(nat)
        .mark_bar()
        .encode(
            x="cole_naturaleza:N",
            y=alt.Y("punt_global:Q", title="Puntaje promedio"),
            color=alt.Color(
                "cole_area_ubicacion:N",
                scale=alt.Scale(range=[COLOR_PRIMARY, COLOR_SECONDARY]),
                title="Área",
            ),
            xOffset="cole_area_ubicacion:N",
            tooltip=[
                "cole_naturaleza",
                "cole_area_ubicacion",
                alt.Tooltip("punt_global:Q", format=".1f"),
            ],
        )
        .properties(height=320)
    )
    st.altair_chart(chart_nat, use_container_width=True)

with col_b:
    section_title("Distribución del puntaje por género")
    chart_gen = (
        alt.Chart(df.sample(min(20_000, len(df)), random_state=42))
        .transform_density(
            "punt_global",
            groupby=["estu_genero"],
            as_=["punt_global", "density"],
        )
        .mark_area(opacity=0.5)
        .encode(
            x=alt.X("punt_global:Q", title="Puntaje"),
            y=alt.Y("density:Q", title="Densidad"),
            color=alt.Color(
                "estu_genero:N",
                scale=alt.Scale(range=[COLOR_PRIMARY, COLOR_SECONDARY]),
                title="Género",
            ),
        )
        .properties(height=320)
    )
    st.altair_chart(chart_gen, use_container_width=True)

# ----------------------------------------------------------------------
# Departamento (top y bottom)
# ----------------------------------------------------------------------

section_title("Puntaje promedio por departamento")
depto = (
    df.groupby("estu_depto_reside")
    .agg(punt_global=("punt_global", "mean"), n=("punt_global", "size"))
    .reset_index()
    .query("n >= 100")
    .sort_values("punt_global")
)
chart_depto = (
    alt.Chart(depto)
    .mark_bar(color=COLOR_PRIMARY)
    .encode(
        x=alt.X("punt_global:Q", title="Puntaje promedio"),
        y=alt.Y("estu_depto_reside:N", title="Departamento", sort="-x"),
        tooltip=[
            "estu_depto_reside",
            alt.Tooltip("punt_global:Q", format=".1f"),
            alt.Tooltip("n:Q", format=","),
        ],
    )
    .properties(height=max(300, 18 * len(depto)))
)
st.altair_chart(chart_depto, use_container_width=True)
