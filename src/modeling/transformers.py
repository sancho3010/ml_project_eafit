"""
Transformers custom del pipeline de Saber 11.

Dos piezas:

* ``DeterministicFeatureProcessor``: feature engineering puramente
  determinista. No aprende nada, solo deriva columnas. Stateless por diseño,
  así que no puede introducir leakage de ningún tipo.

* ``GroupAggregatesTransformer``: agregaciones por municipio y departamento.
  Aprende lookups en ``fit`` y los aplica en ``transform``. Para evitar el
  ``in-sample bias`` clásico (cada fila contribuye a su propio promedio),
  usamos una variante leave-one-out durante el ``fit_transform`` del fold
  de train.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

from src.modeling.constants import EDUCACION_NIVEL, ESTRATO_NIVEL


class DeterministicFeatureProcessor(BaseEstimator, TransformerMixin):
    """
    Feature engineering determinista (stateless, sin leakage).

    Genera variables derivadas que codifican intuiciones del dominio:
    capital tecnológico, hacinamiento, brechas educativas entre padres,
    edad aproximada, índice socioeconómico, etc. Ninguna depende del
    target ni de otras filas, por lo que es seguro aplicarlo antes o
    después del split sin afectar la validación.
    """

    def __init__(
        self,
        mapeo_personas: dict,
        mapeo_cuartos: dict,
        clip_edad: tuple[int, int] = (15, 60),
    ) -> None:
        self.mapeo_personas = mapeo_personas
        self.mapeo_cuartos = mapeo_cuartos
        self.clip_edad = clip_edad

    def fit(self, X, y=None):
        # Stateless: nada que aprender.
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        df = X.copy()

        # Edad aproximada al momento del examen.
        # estu_fechanacimiento viene como string "DD/MM/YYYY".
        year_periodo = df["periodo"].str[:4].astype(int)
        year_nacim = pd.to_numeric(df["estu_fechanacimiento"].str[6:10], errors="coerce")
        df["edad_estudiante_aprox"] = (
            (year_periodo - year_nacim).clip(*self.clip_edad).astype("Int32")
        )

        # Tendencia temporal explícita, útil para los boosters.
        df["anio_periodo"] = df["periodo"].str[:4].astype(int)

        # Tamaño del hogar y cuartos: unificamos formatos heterogéneos.
        df["fami_personashogar_num"] = (
            df["fami_personashogar"].map(self.mapeo_personas).astype("Int32")
        )
        df["fami_cuartos_num"] = df["fami_cuartoshogar"].map(self.mapeo_cuartos).astype("Int32")

        # Hacinamiento como proxy de calidad de vivienda.
        df["hacinamiento"] = (
            df["fami_personashogar_num"].astype("float32")
            / df["fami_cuartos_num"].astype("float32").replace(0, np.nan)
        ).astype("float32")

        # Capital tecnológico (0/1/2): internet + computador.
        df["capital_tecnologico"] = (df["fami_tieneinternet"] == "Si").astype(int) + (
            df["fami_tienecomputador"] == "Si"
        ).astype(int)

        # Proxy de riqueza total (0-4): tecnología + vehículo + lavadora.
        auto = (df["fami_tieneautomovil"] == "Si").astype(int)
        lav = (df["fami_tienelavadora"] == "Si").astype(int)
        df["proxy_riqueza_total"] = (df["capital_tecnologico"] + auto + lav).astype("Int32")

        # Educación de los padres en escala numérica.
        edu_madre = df["fami_educacionmadre"].map(EDUCACION_NIVEL)
        edu_padre = df["fami_educacionpadre"].map(EDUCACION_NIVEL)
        edu_concat = pd.concat([edu_madre, edu_padre], axis=1)

        df["max_educacion_padres"] = edu_concat.max(axis=1).astype("Int32")
        df["min_educacion_padres"] = edu_concat.min(axis=1).astype("Int32")
        df["brecha_educativa_padres"] = (edu_madre - edu_padre).abs().astype("Int32")
        df["padres_universitarios"] = ((edu_madre >= 7) | (edu_padre >= 7)).astype(int)
        df["padres_sin_educacion"] = ((edu_madre <= 1) & (edu_padre <= 1)).astype(int)

        # Extra-edad: estudiantes mayores a la edad esperada de Saber 11.
        df["extra_edad"] = (df["edad_estudiante_aprox"].astype("Int32") - 16).astype("Int32")

        # Movilidad geográfica del estudiante.
        df["presenta_donde_reside_mcpio"] = (
            df["estu_mcpio_reside"] == df["estu_mcpio_presentacion"]
        ).astype(int)
        df["presenta_donde_reside_depto"] = (
            df["estu_depto_reside"] == df["estu_depto_presentacion"]
        ).astype(int)

        # Índice socioeconómico compuesto, suma ponderada determinista.
        # Pesos: estrato x3, capital tecnológico x2, automóvil/lavadora x1.
        estrato_num = df["fami_estratovivienda"].map(ESTRATO_NIVEL).fillna(0).astype(int)
        df["indice_socioeconomico"] = (
            3 * estrato_num + 2 * df["capital_tecnologico"] + auto + lav
        ).astype("Int32")

        # Soltamos columnas originales que ya están reemplazadas por sus
        # versiones numéricas, para no duplicar señal.
        df = df.drop(
            columns=[
                "estu_fechanacimiento",
                "fami_personashogar",
                "fami_cuartoshogar",
            ]
        )

        return df


class GroupAggregatesTransformer(BaseEstimator, TransformerMixin):
    """
    Agregados por municipio/departamento, sin usar el target.

    Features generadas:

    * ``n_estudiantes_mcpio``: tamaño del municipio.
    * ``pct_estrato_alto_mcpio``: % con estrato >= 4.
    * ``pct_internet_mcpio``: % con internet en casa.
    * ``pct_oficial_mcpio``: % de colegios oficiales.
    * ``pct_calendario_b_depto``: % calendario B en el departamento.

    ``fit`` calcula los lookups; ``transform`` los aplica vía ``map`` con
    fallback al promedio global para categorías nuevas.

    En ``fit_transform`` (fold de train) usamos leave-one-out: cada fila
    recibe el agregado calculado sin su propia contribución, equivalente
    a ``(suma_grupo - valor_propio) / (n_grupo - 1)``. Test usa lookups
    planos vía ``transform``.
    """

    # Constantes para LOO. Si una categoría tiene n=1 en train, su LOO
    # promedio es indefinido; usamos el promedio global del lookup.
    MIN_N_LOO = 2

    def __init__(self) -> None:
        pass

    # --- API pública --------------------------------------------------------

    def fit(self, X: pd.DataFrame, y=None):
        df_temp = self.add_indicator_cols(X)

        # Conteo por municipio (lookup global, sin LOO).
        self.n_mcpio_ = df_temp.groupby("estu_mcpio_reside").size().astype("float32")

        # Sumas y conteos por municipio para los porcentajes.
        # Guardamos sumas y n para poder reconstruir el LOO sin recalcular.
        agg_mcpio = df_temp.groupby("estu_mcpio_reside")[
            ["estrato_alto", "internet_si", "oficial"]
        ].agg(["sum", "count"])

        self._sum_estrato_alto_ = agg_mcpio["estrato_alto"]["sum"].astype("float32")
        self._n_estrato_alto_ = agg_mcpio["estrato_alto"]["count"].astype("float32")
        self._sum_internet_ = agg_mcpio["internet_si"]["sum"].astype("float32")
        self._n_internet_ = agg_mcpio["internet_si"]["count"].astype("float32")
        self._sum_oficial_ = agg_mcpio["oficial"]["sum"].astype("float32")
        self._n_oficial_ = agg_mcpio["oficial"]["count"].astype("float32")

        # Promedios "globales" del municipio (lo que verá test).
        self.pct_estrato_alto_mcpio_ = (self._sum_estrato_alto_ / self._n_estrato_alto_).astype(
            "float32"
        )

        self.pct_internet_mcpio_ = (self._sum_internet_ / self._n_internet_).astype("float32")

        self.pct_oficial_mcpio_ = (self._sum_oficial_ / self._n_oficial_).astype("float32")

        # Promedio por departamento.
        self.pct_calendario_b_depto_ = (
            df_temp.groupby("estu_depto_reside")["calendario_b"].mean().astype("float32")
        )

        # Medias globales para fallback en categorías nuevas.
        self.global_n_mcpio_ = float(self.n_mcpio_.mean())
        self.global_pct_estrato_alto_ = float(self.pct_estrato_alto_mcpio_.mean())
        self.global_pct_internet_ = float(self.pct_internet_mcpio_.mean())
        self.global_pct_oficial_ = float(self.pct_oficial_mcpio_.mean())
        self.global_pct_calendario_b_ = float(self.pct_calendario_b_depto_.mean())

        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        """Aplica lookups completos. Usado para test y para inferencia."""
        df = X.copy()

        return self.apply_lookups(df, leave_one_out=False)

    def fit_transform(self, X: pd.DataFrame, y=None) -> pd.DataFrame:
        """
        Fit + transform con LOO. Usado solo en el fold de train.

        Esta sobrescritura es la que evita el in-sample bias: cada fila
        de train recibe el agregado calculado **sin su propia contribución**.
        """
        self.fit(X, y)
        df = X.copy()

        return self.apply_lookups(df, leave_one_out=True)

    # --- Helpers internos ---------------------------------------------------

    @staticmethod
    def add_indicator_cols(X: pd.DataFrame) -> pd.DataFrame:
        """Crea columnas binarias temporales para las agregaciones."""
        return X.assign(
            estrato_alto=X["fami_estratovivienda"]
            .isin(["Estrato 4", "Estrato 5", "Estrato 6"])
            .astype(float),
            internet_si=(X["fami_tieneinternet"] == "Si").astype(float),
            oficial=(X["cole_naturaleza"] == "OFICIAL").astype(float),
            calendario_b=(X["cole_calendario"] == "B").astype(float),
        )

    def loo_pct(
        self,
        keys: pd.Series,
        own_value: pd.Series,
        sum_lookup: pd.Series,
        n_lookup: pd.Series,
        global_pct: float,
    ) -> pd.Series:
        """
        Retorna el porcentaje del grupo excluyendo la fila actual.

        ``(suma_grupo - valor_propio) / (n_grupo - 1)``. Si el grupo
        tiene n<2 o la categoría es nueva, usa el promedio global.
        """
        s = keys.map(sum_lookup)
        n = keys.map(n_lookup)
        loo = (s - own_value) / (n - 1)

        # Donde n<2 o categoría nueva: fallback al global.
        mask_invalid = (n < self.MIN_N_LOO) | n.isna()
        loo = loo.where(~mask_invalid, global_pct)

        return loo.astype("float32")

    def apply_lookups(self, df: pd.DataFrame, leave_one_out: bool) -> pd.DataFrame:
        if leave_one_out:
            # Necesitamos las indicadoras temporales para restar la contribución propia.
            ind = self.add_indicator_cols(df)

            df["pct_estrato_alto_mcpio"] = self.loo_pct(
                df["estu_mcpio_reside"],
                ind["estrato_alto"],
                self._sum_estrato_alto_,
                self._n_estrato_alto_,
                self.global_pct_estrato_alto_,
            )

            df["pct_internet_mcpio"] = self.loo_pct(
                df["estu_mcpio_reside"],
                ind["internet_si"],
                self._sum_internet_,
                self._n_internet_,
                self.global_pct_internet_,
            )

            df["pct_oficial_mcpio"] = self.loo_pct(
                df["estu_mcpio_reside"],
                ind["oficial"],
                self._sum_oficial_,
                self._n_oficial_,
                self.global_pct_oficial_,
            )

        else:
            # Lookup directo (test e inferencia).
            df["pct_estrato_alto_mcpio"] = (
                df["estu_mcpio_reside"]
                .map(self.pct_estrato_alto_mcpio_)
                .fillna(self.global_pct_estrato_alto_)
                .astype("float32")
            )

            df["pct_internet_mcpio"] = (
                df["estu_mcpio_reside"]
                .map(self.pct_internet_mcpio_)
                .fillna(self.global_pct_internet_)
                .astype("float32")
            )

            df["pct_oficial_mcpio"] = (
                df["estu_mcpio_reside"]
                .map(self.pct_oficial_mcpio_)
                .fillna(self.global_pct_oficial_)
                .astype("float32")
            )

        # n_estudiantes y pct_calendario_b: lookups planos (sin LOO).
        df["n_estudiantes_mcpio"] = (
            df["estu_mcpio_reside"]
            .map(self.n_mcpio_)
            .fillna(self.global_n_mcpio_)
            .astype("float32")
        )

        df["pct_calendario_b_depto"] = (
            df["estu_depto_reside"]
            .map(self.pct_calendario_b_depto_)
            .fillna(self.global_pct_calendario_b_)
            .astype("float32")
        )

        return df
