"""Fixtures compartidos: mini-DataFrame con la estructura de Saber 11."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def saber11_sample() -> pd.DataFrame:
    """Mini-DataFrame de 12 filas con todas las columnas que el pipeline espera."""
    rng = np.random.default_rng(42)

    n = 12
    return pd.DataFrame(
        {
            "periodo": ["20142", "20152", "20162", "20172"] * 3,
            "estu_fechanacimiento": [
                "01/01/1998",
                "15/06/1999",
                "23/03/2000",
                "10/11/2001",
            ] * 3,
            "estu_genero": rng.choice(["F", "M"], size=n),
            "estu_tipodocumento": ["TI"] * n,
            "estu_depto_reside": ["ANTIOQUIA"] * 6 + ["BOGOTA D.C."] * 6,
            "estu_mcpio_reside": (
                ["MEDELLIN"] * 4 + ["ENVIGADO"] * 2 + ["BOGOTA D.C."] * 6
            ),
            "estu_depto_presentacion": ["ANTIOQUIA"] * 6 + ["BOGOTA D.C."] * 6,
            "estu_mcpio_presentacion": (
                ["MEDELLIN"] * 4 + ["ENVIGADO"] * 2 + ["BOGOTA D.C."] * 6
            ),
            "fami_estratovivienda": [
                "Estrato 2",
                "Estrato 3",
                "Estrato 4",
                "Estrato 1",
                "Estrato 5",
                "Estrato 2",
                "Estrato 3",
                "Estrato 6",
                "Estrato 1",
                "Estrato 4",
                "Estrato 2",
                "Estrato 3",
            ],
            "fami_educacionmadre": [
                "Primaria completa",
                "Secundaria (Bachillerato) completa",
                "Educación profesional completa",
                "Ninguno",
                "Postgrado",
                "Primaria incompleta",
                "Secundaria (Bachillerato) completa",
                "Educación profesional completa",
                "Ninguno",
                "Técnica o tecnológica completa",
                "Primaria completa",
                "Secundaria (Bachillerato) incompleta",
            ],
            "fami_educacionpadre": [
                "Primaria completa",
                "Secundaria (Bachillerato) incompleta",
                "Educación profesional incompleta",
                "Ninguno",
                "Postgrado",
                "Primaria completa",
                "Secundaria (Bachillerato) completa",
                "Educación profesional completa",
                "No sabe",
                "Técnica o tecnológica completa",
                "Primaria incompleta",
                "Secundaria (Bachillerato) completa",
            ],
            "fami_personashogar": [
                "Tres",
                "Cuatro",
                "Cinco",
                "Seis",
                "Tres",
                "Cuatro",
                "3 a 4",
                "5 a 6",
                "7 a 8",
                "1 a 2",
                "Cuatro",
                "Tres",
            ],
            "fami_cuartoshogar": [
                "Dos",
                "Tres",
                "Cuatro",
                "Dos",
                "Cinco",
                "Tres",
                "Tres",
                "Cuatro",
                "Dos",
                "Tres",
                "Dos",
                "Tres",
            ],
            "fami_tieneinternet": [
                "Si",
                "Si",
                "Si",
                "No",
                "Si",
                "No",
                "Si",
                "Si",
                "No",
                "Si",
                "No",
                "Si",
            ],
            "fami_tienecomputador": [
                "Si",
                "No",
                "Si",
                "No",
                "Si",
                "No",
                "Si",
                "Si",
                "No",
                "Si",
                "Si",
                "Si",
            ],
            "fami_tieneautomovil": [
                "No",
                "No",
                "Si",
                "No",
                "Si",
                "No",
                "No",
                "Si",
                "No",
                "Si",
                "No",
                "Si",
            ],
            "fami_tienelavadora": [
                "Si",
                "Si",
                "Si",
                "No",
                "Si",
                "Si",
                "Si",
                "Si",
                "No",
                "Si",
                "Si",
                "Si",
            ],
            "cole_naturaleza": (
                ["OFICIAL"] * 7 + ["NO OFICIAL"] * 5
            ),
            "cole_area_ubicacion": ["URBANO"] * 10 + ["RURAL"] * 2,
            "cole_calendario": ["A"] * 10 + ["B"] * 2,
            "cole_jornada": ["MAÑANA"] * 6 + ["TARDE"] * 4 + ["COMPLETA"] * 2,
            "cole_genero": ["MIXTO"] * n,
            "cole_caracter": ["ACADÉMICO"] * n,
            "cole_bilingue": ["N"] * 10 + ["S"] * 2,
            "cole_sede_principal": ["S"] * n,
            "cole_depto_ubicacion": ["ANTIOQUIA"] * 6 + ["BOGOTA D.C."] * 6,
            "cole_mcpio_ubicacion": (
                ["MEDELLIN"] * 4 + ["ENVIGADO"] * 2 + ["BOGOTA D.C."] * 6
            ),
        }
    )


@pytest.fixture
def saber11_target(saber11_sample) -> np.ndarray:
    """Target sintético correlacionado con estrato (para que el modelo aprenda algo)."""
    rng = np.random.default_rng(0)
    estrato_num = saber11_sample["fami_estratovivienda"].str.extract(
        r"(\d)"
    ).fillna("0").astype(int)[0].to_numpy()
    base = 200 + estrato_num * 15
    noise = rng.normal(0, 8, size=len(base))
    return (base + noise).astype("float32")
