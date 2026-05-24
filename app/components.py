"""Componentes reutilizables: KPIs, headers, banners."""

from __future__ import annotations

import streamlit as st

from app.theme import CSS


def apply_theme():
    """Inyecta el CSS global. Llamar una vez por página."""
    st.markdown(CSS, unsafe_allow_html=True)


def page_header(title: str, subtitle: str = ""):
    """Header consistente para todas las páginas."""
    st.markdown(
        f"""
        <div style="margin-bottom: 1.5rem;">
            <h1 style="margin:0; color:#1f4e79; font-size: 2rem;">{title}</h1>
            <p class="muted" style="margin: 0.25rem 0 0 0;">{subtitle}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_title(text: str):
    st.markdown(f'<div class="section-title">{text}</div>', unsafe_allow_html=True)


def kpi(label: str, value: str, delta: str = ""):
    """Tarjeta KPI sobria. Renderizar dentro de un st.columns."""
    delta_html = f'<div class="delta">{delta}</div>' if delta else ""
    st.markdown(
        f"""
        <div class="kpi-card">
            <div class="label">{label}</div>
            <div class="value">{value}</div>
            {delta_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def kpi_row(items: list[tuple[str, str, str]]):
    """Fila de KPIs. items = [(label, value, delta), ...]"""
    cols = st.columns(len(items))
    for col, (label, value, delta) in zip(cols, items):
        with col:
            kpi(label, value, delta)


def info_banner(text: str):
    st.info(text, icon=None)


def disclaimer():
    """Aviso ético al pie de cada página de predicción."""
    st.caption(
        "Esta herramienta es analítica. No reemplaza el examen Saber 11 ni "
        "se debe usar para decisiones individuales sobre estudiantes."
    )
