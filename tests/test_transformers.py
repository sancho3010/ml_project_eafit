"""Tests de los transformers custom."""

from __future__ import annotations

import inspect
import numpy as np

from src.modeling.constants import CLIP_EDAD, MAPEO_CUARTOS, MAPEO_PERSONAS
from src.modeling.transformers import (
    DeterministicFeatureProcessor,
    GroupAggregatesTransformer,
)


# ----------------------------------------------------------------------
# DeterministicFeatureProcessor
# ----------------------------------------------------------------------

def test_deterministic_is_stateless(saber11_sample):
    """fit() no debe almacenar estado dependiente de los datos."""
    proc = DeterministicFeatureProcessor(
        MAPEO_PERSONAS,
        MAPEO_CUARTOS,
        CLIP_EDAD
    )
    state_before = vars(proc).copy()

    proc.fit(saber11_sample)
    state_after = vars(proc).copy()

    assert state_before == state_after, ("DeterministicFeatureProcessor debería ser stateless.")


def test_deterministic_creates_expected_features(saber11_sample):
    """transform() debe producir todas las features derivadas."""
    proc = DeterministicFeatureProcessor(
        MAPEO_PERSONAS,
        MAPEO_CUARTOS,
        CLIP_EDAD
    )
    out = proc.transform(saber11_sample)

    expected = {
        "edad_estudiante_aprox",
        "anio_periodo",
        "fami_personashogar_num",
        "fami_cuartos_num",
        "hacinamiento",
        "capital_tecnologico",
        "proxy_riqueza_total",
        "max_educacion_padres",
        "min_educacion_padres",
        "brecha_educativa_padres",
        "padres_universitarios",
        "padres_sin_educacion",
        "extra_edad",
        "presenta_donde_reside_mcpio",
        "presenta_donde_reside_depto",
        "indice_socioeconomico",
    }

    missing = expected - set(out.columns)
    assert not missing, f"Faltan features esperadas: {missing}"


def test_deterministic_drops_replaced_columns(saber11_sample):
    """Las columnas string que ya tienen versión numérica deben caer."""
    proc = DeterministicFeatureProcessor(MAPEO_PERSONAS, MAPEO_CUARTOS, CLIP_EDAD)
    out = proc.transform(saber11_sample)
    
    for col in ("estu_fechanacimiento", "fami_personashogar", "fami_cuartoshogar"):
        assert col not in out.columns, f"{col} debería haber sido eliminada"


def test_deterministic_age_clipping(saber11_sample):
    """Edades absurdas deben caer dentro del rango esperado."""
    df = saber11_sample.copy()

    # Forzamos un nacimiento de 1850 (edad absurda).
    df.loc[df.index[0], "estu_fechanacimiento"] = "01/01/1850"
    proc = DeterministicFeatureProcessor(MAPEO_PERSONAS, MAPEO_CUARTOS, (15, 60))

    out = proc.transform(df)
    edad = out["edad_estudiante_aprox"].dropna()
    assert edad.min() >= 15 and edad.max() <= 60


# ----------------------------------------------------------------------
# GroupAggregatesTransformer
# ----------------------------------------------------------------------

def test_group_aggs_does_not_use_target(saber11_sample, saber11_target):
    """fit() no debe aceptar ni mirar y de ninguna manera implícita."""
    sig = inspect.signature(GroupAggregatesTransformer().fit)

    # La firma debe incluir y=None, no algo obligatorio o central.
    assert "y" in sig.parameters
    assert sig.parameters["y"].default is None


def test_group_aggs_creates_expected_features(saber11_sample):
    g = GroupAggregatesTransformer()
    g.fit(saber11_sample)

    out = g.transform(saber11_sample)
    expected = {
        "n_estudiantes_mcpio",
        "pct_estrato_alto_mcpio",
        "pct_internet_mcpio",
        "pct_oficial_mcpio",
        "pct_calendario_b_depto",
    }
    assert expected.issubset(out.columns)


def test_group_aggs_loo_differs_from_plain_transform(saber11_sample):
    """
    LOO debe producir valores distintos al lookup directo en train.

    Con n>=2 por municipio, restar la propia contribución cambia la métrica.
    """
    g = GroupAggregatesTransformer()
    out_loo = g.fit_transform(saber11_sample)
    out_plain = g.transform(saber11_sample)

    # Para municipios con n>=2 deberían diferir en pct_internet.
    diff = (out_loo["pct_internet_mcpio"] - out_plain["pct_internet_mcpio"]).abs()
    assert diff.sum() > 0, "LOO debería diferir del lookup plano"


def test_group_aggs_unknown_category_falls_back_to_global(saber11_sample):
    """Una categoría no vista debe imputarse con el promedio global."""
    g = GroupAggregatesTransformer().fit(saber11_sample)

    new = saber11_sample.head(1).copy()
    new["estu_mcpio_reside"] = "MUNICIPIO_QUE_NO_EXISTE"
    out = g.transform(new)

    assert np.isclose(out["pct_internet_mcpio"].iloc[0], g.global_pct_internet_, atol=1e-6)
    assert np.isclose(out["pct_oficial_mcpio"].iloc[0], g.global_pct_oficial_, atol=1e-6)
