"""
Constantes y mapeos del dominio Saber 11.

Centralizamos aquí los diccionarios de orden ordinal, mapeos numéricos y
listas de columnas por tipo. Esto evita que el notebook y los scripts
divergan, y deja un único punto de verdad para el feature engineering.
"""

from __future__ import annotations

# --- Mapeos de variables del hogar -------------------------------------------

# Personas en el hogar: ICFES mezcla formato unitario y por rangos según el periodo.
# Mapeamos a un entero "representativo" del rango.
MAPEO_PERSONAS: dict[str, int] = {
    "Uno": 1,
    "Dos": 2,
    "Tres": 3,
    "Cuatro": 4,
    "Cinco": 5,
    "Seis": 6,
    "Siete": 7,
    "Ocho": 8,
    "Nueve": 9,
    "Diez": 10,
    "Once": 11,
    "Doce o más": 12,
    "1 a 2": 1,
    "3 a 4": 3,
    "5 a 6": 5,
    "7 a 8": 7,
    "9 o más": 9,
}

MAPEO_CUARTOS: dict[str, int] = {
    "Uno": 1,
    "Dos": 2,
    "Tres": 3,
    "Cuatro": 4,
    "Cinco": 5,
    "Seis": 6,
    "Seis o mas": 6,
    "Siete": 7,
    "Ocho": 8,
    "Nueve": 9,
    "Diez o más": 10,
}

# Orden ordinal de estrato y educación.
ORDEN_ESTRATO: list[str] = [
    "Sin Estrato",
    "Estrato 1",
    "Estrato 2",
    "Estrato 3",
    "Estrato 4",
    "Estrato 5",
    "Estrato 6",
]

ORDEN_EDUCACION: list[str] = [
    "Ninguno",
    "No sabe",
    "No Aplica",
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

# Mapeo numérico explícito que usa el feature engineering determinista.
EDUCACION_NIVEL: dict[str, int] = {
    "Ninguno": 0,
    "No sabe": 0,
    "No Aplica": 0,
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

ESTRATO_NIVEL: dict[str, int] = {
    "Sin Estrato": 0,
    "Estrato 1": 1,
    "Estrato 2": 2,
    "Estrato 3": 3,
    "Estrato 4": 4,
    "Estrato 5": 5,
    "Estrato 6": 6,
}


# --- Clasificación de columnas para el ColumnTransformer ---------------------

ORDINAL_COLS: list[str] = [
    "fami_estratovivienda",
    "fami_educacionmadre",
    "fami_educacionpadre",
]

BINARY_COLS: list[str] = [
    "fami_tieneinternet",
    "fami_tienecomputador",
    "fami_tieneautomovil",
    "fami_tienelavadora",
    "cole_bilingue",
    "cole_sede_principal",
    "estu_genero",
]

NOMINAL_LOW_COLS: list[str] = [
    "cole_naturaleza",
    "cole_area_ubicacion",
    "cole_calendario",
    "cole_jornada",
    "cole_genero",
    "cole_caracter",
    "estu_tipodocumento",
    "estu_depto_reside",
    "estu_depto_presentacion",
    "cole_depto_ubicacion",
    "periodo",
]

# Alta cardinalidad: municipio se usa solo en el lookup de group aggregates, no entra al modelo.
NOMINAL_HIGH_COLS: list[str] = []

NUMERIC_COLS: list[str] = [
    # Generadas por DeterministicFeatureProcessor
    "edad_estudiante_aprox",
    "anio_periodo",
    "fami_personashogar_num",
    "fami_cuartos_num",
    "capital_tecnologico",
    "hacinamiento",
    "max_educacion_padres",
    "min_educacion_padres",
    "brecha_educativa_padres",
    "padres_universitarios",
    "padres_sin_educacion",
    "extra_edad",
    "presenta_donde_reside_mcpio",
    "presenta_donde_reside_depto",
    "proxy_riqueza_total",
    "indice_socioeconomico",
    # Generadas por GroupAggregatesTransformer (sin target, fit solo en train)
    "n_estudiantes_mcpio",
    "pct_estrato_alto_mcpio",
    "pct_internet_mcpio",
    "pct_oficial_mcpio",
    "pct_calendario_b_depto",
]


# --- Hiperparámetros de espacios de búsqueda (defaults) ----------------------

CLIP_EDAD: tuple[int, int] = (15, 60)
# Rango razonable para edad de un estudiante de Saber 11.
# Recortamos extremos absurdos producto de errores de captura.
