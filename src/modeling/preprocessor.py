"""
Construcción del ColumnTransformer.

Mantenemos esta función separada porque define el contrato entre el
feature engineering y el modelo: qué columnas son ordinales, cuáles
binarias, cuáles nominales, cuáles numéricas.
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder

from src.modeling.constants import (
    BINARY_COLS,
    NOMINAL_HIGH_COLS,
    NOMINAL_LOW_COLS,
    NUMERIC_COLS,
    ORDEN_EDUCACION,
    ORDEN_ESTRATO,
    ORDINAL_COLS,
)


def build_preprocessor() -> ColumnTransformer:
    """
    Devuelve el ColumnTransformer estándar del proyecto.

    * Ordinales con orden explícito (estrato y educación de padres).
    * Binarias y nominales de baja cardinalidad con OHE.
    * Nominales de alta cardinalidad con ordinal encoding (placeholder).
    * Numéricas con imputación por mediana.

    Todo va dentro del Pipeline, así que se ajusta por fold y nunca toca
    el test holdout durante el tuning.
    """
    return ColumnTransformer(
        [
            (
                "ordinal",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                categories=[
                                    ORDEN_ESTRATO,
                                    ORDEN_EDUCACION,
                                    ORDEN_EDUCACION,
                                ],
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                ORDINAL_COLS,
            ),
            (
                "binary",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                BINARY_COLS,
            ),
            (
                "nominal_low",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                NOMINAL_LOW_COLS,
            ),
            (
                "nominal_high",
                Pipeline(
                    [
                        ("imputer", SimpleImputer(strategy="most_frequent")),
                        (
                            "encoder",
                            OrdinalEncoder(
                                handle_unknown="use_encoded_value",
                                unknown_value=-1,
                            ),
                        ),
                    ]
                ),
                NOMINAL_HIGH_COLS,
            ),
            (
                "numeric",
                Pipeline([("imputer", SimpleImputer(strategy="median"))]),
                NUMERIC_COLS,
            ),
        ],
        remainder="drop",
    )
