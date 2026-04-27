"""
Utilidades para el EDA del proyecto ICFES Saber 11.

Funciones de auditoría, visualización y análisis para el notebook de EDA.
Diseñadas para trabajar con Polars y Matplotlib/Seaborn.
"""

import numpy as np
import polars as pl
import matplotlib.pyplot as plt
import seaborn as sns
from loguru import logger
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OrdinalEncoder
from scipy.stats import skew, kurtosis
from scipy.stats import chi2_contingency

# ---------------------------------------------------------------------------
# Auditoría de schema y calidad
# ---------------------------------------------------------------------------

def summarize_schema(df: pl.DataFrame, target: str = None) -> pl.DataFrame:
    """Resumen compacto del schema: tipo, nulos, unicos, rol."""
    rows = []
    for col in df.columns:
        series = df[col]
        rows.append({
            "column": col,
            "dtype": str(series.dtype),
            "non_null": series.drop_nulls().len(),
            "missing": series.null_count(),
            "missing_pct": round(series.null_count() / df.height * 100, 2),
            "n_unique": series.n_unique(),
            "role": "target" if col == target else "feature",
        })

    summary = pl.DataFrame(rows)
    return summary.sort(["role", "missing_pct"], descending=[True, True])


def missing_summary(df: pl.DataFrame) -> pl.DataFrame:
    """Resumen de nulos ordenado por porcentaje de missingness."""
    rows = []
    for col in df.columns:
        series = df[col]
        rows.append({
            "column": col,
            "missing": series.null_count(),
            "missing_pct": round(series.null_count() / df.height * 100, 2),
            "dtype": str(series.dtype),
        })

    return pl.DataFrame(rows).sort("missing", descending=True)


def low_variance_columns(df: pl.DataFrame, threshold: float = 99.0) -> pl.DataFrame:
    """Detecta columnas donde el valor mas frecuente supera el threshold (%)."""
    rows = []
    for col in df.columns:
        series = df[col].drop_nulls()

        if series.len() == 0:
            continue

        top_freq = series.value_counts().sort("count", descending=True)["count"][0]
        pct = round(top_freq / series.len() * 100, 2)
        rows.append({
            "column": col,
            "top_value": str(series.value_counts().sort("count", descending=True)[col][0]),
            "top_pct": pct,
            "drop": pct >= threshold,
        })

    return pl.DataFrame(rows).sort("top_pct", descending=True)


def detect_duplicates(df: pl.DataFrame, subset: list[str] = None) -> int:
    """Cuenta y reporta duplicados. Retorna el conteo."""
    if subset:
        n_dup = df.select(subset).is_duplicated().sum()
        logger.info(f"Duplicados por {subset}: {n_dup:,} ({n_dup / df.height * 100:.2f}%)")
    else:
        n_dup = df.is_duplicated().sum()
        logger.info(f"Duplicados exactos: {n_dup:,} ({n_dup / df.height * 100:.2f}%)")

    return n_dup


# ---------------------------------------------------------------------------
# Casteo y limpieza
# ---------------------------------------------------------------------------

def cast_numeric_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Castea columnas string a Int64, valores no parseables quedan como null."""
    return df.with_columns([
        pl.col(c).cast(pl.Int64, strict=False).alias(c) for c in columns
    ])


def drop_columns(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Elimina columnas del dataframe y reporta."""
    existing = [c for c in columns if c in df.columns]
    logger.info(f"Eliminando {len(existing)} columnas: {existing}")
    return df.drop(existing)


# ---------------------------------------------------------------------------
# Estadísticas descriptivas
# ---------------------------------------------------------------------------

def numeric_profile(df: pl.DataFrame, columns: list[str]) -> pl.DataFrame:
    """Perfil numerico robusto: media, mediana, std, skew, outliers IQR."""
    rows = []
    for col in columns:
        s = df[col].drop_nulls().to_numpy()
        if len(s) == 0:
            continue

        q1, q3 = np.percentile(s, [25, 75])
        iqr = q3 - q1
        outlier_mask = (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)

        rows.append({
            "column": col,
            "count": len(s),
            "mean": round(float(np.mean(s)), 2),
            "median": round(float(np.median(s)), 2),
            "std": round(float(np.std(s, ddof=1)), 2),
            "min": float(np.min(s)),
            "q1": float(q1),
            "q3": float(q3),
            "max": float(np.max(s)),
            "skew": round(float(skew(s)), 3),
            "kurtosis": round(float(kurtosis(s)), 3),
            "outlier_pct_iqr": round(float(outlier_mask.mean() * 100), 2),
        })

    return pl.DataFrame(rows)


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    """Calcula Cohen's d (tamano del efecto estandarizado)."""
    n_a, n_b = len(group_a), len(group_b)
    var_a, var_b = np.var(group_a, ddof=1), np.var(group_b, ddof=1)
    pooled_std = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / (n_a + n_b - 2))

    if pooled_std == 0:
        return 0.0

    return float((np.mean(group_a) - np.mean(group_b)) / pooled_std)


def equity_gap_table(
    df: pl.DataFrame,
    target: str,
    group_col: str,
    group_a_val: str,
    group_b_val: str,
) -> dict:
    """Calcula brecha entre dos grupos: diferencia de medias y Cohen's d."""
    a = df.filter(pl.col(group_col) == group_a_val)[target].drop_nulls().to_numpy()
    b = df.filter(pl.col(group_col) == group_b_val)[target].drop_nulls().to_numpy()

    if len(a) == 0 or len(b) == 0:
        logger.warning(f"Grupo vacio para {group_col}: {group_a_val} o {group_b_val}")
        return {}

    return {
        "group_col": group_col,
        "group_a": group_a_val,
        "group_b": group_b_val,
        "mean_a": round(float(np.mean(a)), 2),
        "mean_b": round(float(np.mean(b)), 2),
        "diff": round(float(np.mean(a) - np.mean(b)), 2),
        "cohens_d": round(cohens_d(a, b), 3),
        "n_a": len(a),
        "n_b": len(b),
    }


def cramers_v(df: pl.DataFrame, col_a: str, col_b: str) -> float:
    """Calcula Cramer's V entre dos columnas categoricas."""
    cross = df.select([col_a, col_b]).drop_nulls()
    ct = cross.group_by([col_a, col_b]).len().pivot(
        on=col_b, index=col_a, values="len"
    ).fill_null(0)

    matrix = ct.drop(col_a).to_numpy()
    chi2, _, _, _ = chi2_contingency(matrix)
    n = matrix.sum()
    r, k = matrix.shape
    denom = n * (min(r, k) - 1)

    if denom == 0:
        return 0.0

    return float(np.sqrt(chi2 / denom))


# ---------------------------------------------------------------------------
# Visualizaciones
# ---------------------------------------------------------------------------

def plot_missing_bars(df: pl.DataFrame, figsize: tuple = (12, 5)) -> None:
    """Barplot de porcentaje de nulos por columna (solo columnas con nulos)."""
    ms = missing_summary(df).filter(pl.col("missing") > 0)
    if ms.height == 0:
        logger.info("No hay valores faltantes en el dataset.")
        return

    fig, ax = plt.subplots(figsize=figsize)
    cols = ms["column"].to_list()
    pcts = ms["missing_pct"].to_list()

    ax.barh(cols, pcts)
    ax.set_xlabel("% faltante")
    ax.set_title("Porcentaje de valores faltantes por columna")
    ax.axvline(5, linestyle="--", linewidth=0.8, color="red", label="5%")
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_target_distribution(
    df: pl.DataFrame,
    target: str,
    bins: int = 50,
    figsize: tuple = (14, 5),
) -> None:
    """Histograma + boxplot del target."""
    values = df[target].drop_nulls().to_numpy().astype(float)
    fig, axes = plt.subplots(1, 2, figsize=figsize)

    axes[0].hist(values, bins=bins, edgecolor="black", alpha=0.8)
    axes[0].axvline(np.mean(values), ls="--", lw=1, label=f"Media: {np.mean(values):.1f}")
    axes[0].axvline(np.median(values), ls=":", lw=1, label=f"Mediana: {np.median(values):.1f}")
    axes[0].set_title(f"Distribucion de {target}")
    axes[0].set_xlabel(target)
    axes[0].set_ylabel("Frecuencia")
    axes[0].legend()

    axes[1].boxplot(values, vert=True, patch_artist=True)
    axes[1].set_title(f"Boxplot de {target}")
    axes[1].set_ylabel(target)

    plt.tight_layout()
    plt.show()
    logger.info(
        f"{target} | n={len(values):,} | media={np.mean(values):.1f} "
        f"| mediana={np.median(values):.1f} | std={np.std(values):.1f}"
    )


def plot_categorical_vs_target(
    df: pl.DataFrame,
    cat_col: str,
    target: str,
    order: list[str] = None,
    max_label_len: int = 20,
    figsize: tuple = (14, 5),
) -> None:
    """Boxplot del target por categoria + conteo de frecuencias."""
    sub = df.select([cat_col, target]).drop_nulls()
    pdf = sub.to_pandas()

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Frecuencias
    counts = pdf[cat_col].value_counts()
    if order:
        counts = counts.reindex(order).dropna()
    labels_full = counts.index.astype(str).tolist()
    labels_short = [
        l[:max_label_len] + "..." if len(l) > max_label_len else l
        for l in labels_full
    ]

    axes[0].barh(labels_short, counts.values)
    axes[0].set_title(f"Frecuencia: {cat_col}")
    axes[0].set_xlabel("Conteo")

    # Boxplot
    sns.boxplot(data=pdf, x=cat_col, y=target, order=order or labels_full, ax=axes[1])
    axes[1].set_title(f"{target} por {cat_col}")
    axes[1].set_xticklabels(
        [l[:max_label_len] + "..." if len(l) > max_label_len else l
         for l in (order or labels_full)],
        rotation=45, ha="right", fontsize=8,
    )

    plt.tight_layout()
    plt.show()


def plot_correlation_heatmap(
    df: pl.DataFrame,
    columns: list[str],
    figsize: tuple = (10, 8),
) -> None:
    """Heatmap de correlacion con valores anotados."""
    sub = df.select(columns).drop_nulls()
    corr = sub.to_pandas().corr()

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(corr.values, aspect="auto", vmin=-1, vmax=1, cmap="RdBu_r")

    for i in range(len(corr)):
        for j in range(len(corr)):
            val = corr.values[i, j]
            color = "white" if abs(val) > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=9, color=color)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.index)))
    ax.set_yticklabels(corr.index)
    ax.set_title("Matriz de correlacion")

    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def plot_kde_by_group(
    df: pl.DataFrame,
    target: str,
    group_col: str,
    groups: list[str],
    figsize: tuple = (12, 5),
) -> None:
    """KDE plots superpuestos del target para distintos grupos."""
    fig, ax = plt.subplots(figsize=figsize)
    for g in groups:
        values = df.filter(pl.col(group_col) == g)[target].drop_nulls().to_numpy().astype(float)

        if len(values) > 0:
            ax.hist(values, bins=80, density=True, alpha=0.3, label=f"{g} (n={len(values):,})")

    ax.set_title(f"Distribucion de {target} por {group_col}")
    ax.set_xlabel(target)
    ax.set_ylabel("Densidad")
    ax.legend()
    plt.tight_layout()
    plt.show()


def plot_temporal_trend(
    df: pl.DataFrame,
    target: str,
    period_col: str = "periodo",
    figsize: tuple = (14, 5),
) -> None:
    """Linea temporal de promedio del target y conteo por periodo."""
    agg = (
        df.filter(pl.col(target).is_not_null())
        .group_by(period_col)
        .agg([
            pl.col(target).cast(pl.Float64).mean().alias("mean_target"),
            pl.len().alias("count"),
        ])
        .sort(period_col)
    )

    fig, ax1 = plt.subplots(figsize=figsize)
    periods = agg[period_col].to_list()
    means = agg["mean_target"].to_list()
    counts = agg["count"].to_list()

    ax1.plot(periods, means, marker="o", color="tab:blue", label=f"Media {target}")
    ax1.set_xlabel("Periodo")
    ax1.set_ylabel(f"Media {target}", color="tab:blue")
    ax1.tick_params(axis="x", rotation=45)

    ax2 = ax1.twinx()
    ax2.bar(periods, counts, alpha=0.2, color="tab:gray", label="Registros")
    ax2.set_ylabel("Registros", color="tab:gray")

    ax1.set_title(f"Tendencia temporal de {target}")
    fig.legend(loc="upper left", bbox_to_anchor=(0.12, 0.88))
    plt.tight_layout()
    plt.show()


def plot_radar_by_group(
    df: pl.DataFrame,
    score_cols: list[str],
    group_col: str,
    groups: list[str],
    figsize: tuple = (8, 8),
) -> None:
    """Radar chart de puntajes promedio por grupo."""
    angles = np.linspace(0, 2 * np.pi, len(score_cols), endpoint=False).tolist()
    angles += angles[:1]

    fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True))
    for g in groups:
        sub = df.filter(pl.col(group_col) == g)
        means = [sub[c].drop_nulls().cast(pl.Float64).mean() for c in score_cols]
        means += means[:1]

        ax.plot(angles, means, marker="o", label=g)
        ax.fill(angles, means, alpha=0.1)

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(score_cols, size=9)
    ax.set_title(f"Perfil de puntajes por {group_col}")
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Analisis jerarquico y valor agregado
# ---------------------------------------------------------------------------

def compute_icc(df: pl.DataFrame, target: str, group_col: str) -> float:
    """
    Calcula el ICC (Intraclass Correlation Coefficient).

    ICC = var_between / (var_between + var_within)
    """
    target_values = df.select([group_col, target]).drop_nulls()
    grand_mean = target_values[target].cast(pl.Float64).mean()

    group_stats = (
        target_values
        .group_by(group_col)
        .agg([
            pl.col(target).cast(pl.Float64).mean().alias("group_mean"),
            pl.col(target).cast(pl.Float64).var().alias("group_var"),
            pl.len().alias("n"),
        ])
    )

    means = group_stats["group_mean"].to_numpy()
    vars_ = group_stats["group_var"].to_numpy()
    ns = group_stats["n"].to_numpy()

    # Filtrar colegios con varianza null (1 solo estudiante)
    valid = ~np.isnan(vars_)
    means = means[valid]
    vars_ = vars_[valid]
    ns = ns[valid]

    var_between = float(np.average((means - grand_mean) ** 2, weights=ns))
    var_within = float(np.average(vars_, weights=ns))

    if var_between + var_within == 0:
        return 0.0

    icc = var_between / (var_between + var_within)
    logger.info(
        f"ICC({target} | {group_col}) = {icc:.4f} | "
        f"var_between={var_between:.2f} | var_within={var_within:.2f}"
    )
    return round(icc, 4)


def school_value_added(
    df: pl.DataFrame,
    target: str,
    socioeconomic_cols: list[str],
    school_col: str = "cole_codigo_icfes",
) -> pl.DataFrame:
    """
    Calcula el valor agregado por colegio.

    Ajusta un modelo lineal simple target ~ socioeconomic_cols,
    calcula residuos y promedia por colegio.
    """
    sub = df.select([school_col, target] + socioeconomic_cols).drop_nulls()
    pdf = sub.to_pandas()

    X = pdf[socioeconomic_cols]
    y = pdf[target].astype(float)

    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)
    X_enc = encoder.fit_transform(X)

    model = LinearRegression()
    model.fit(X_enc, y)
    pdf["predicted"] = model.predict(X_enc)
    pdf["residual"] = pdf[target].astype(float) - pdf["predicted"]

    va = (
        pl.from_pandas(pdf)
        .group_by(school_col)
        .agg([
            pl.col("residual").mean().alias("value_added"),
            pl.col("residual").std().alias("va_std"),
            pl.col(target).cast(pl.Float64).mean().alias("mean_score"),
            pl.len().alias("n_students"),
        ])
        .sort("value_added", descending=True)
    )

    logger.info(
        f"Valor agregado calculado para {va.height} colegios | "
        f"R2 del modelo socioeconomico: {model.score(X_enc, y):.4f}"
    )
    return va


# ---------------------------------------------------------------------------
# Gini educativo
# ---------------------------------------------------------------------------

def gini_coefficient(values: np.ndarray) -> float:
    """Calcula el coeficiente de Gini para un array de valores."""
    values = np.sort(values)
    n = len(values)
    if n == 0 or values.sum() == 0:
        return 0.

    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * values) - (n + 1) * np.sum(values)) / (n * np.sum(values)))


def gini_by_group(df: pl.DataFrame, target: str, group_col: str) -> pl.DataFrame:
    """Calcula Gini del target por cada grupo."""
    rows = []
    for group_val in df[group_col].drop_nulls().unique().sort().to_list():
        values = (
            df.filter(pl.col(group_col) == group_val)[target]
            .drop_nulls()
            .cast(pl.Float64)
            .to_numpy()
        )

        if len(values) > 0:
            rows.append({
                group_col: group_val,
                "gini": round(gini_coefficient(values), 4),
                "n": len(values),
            })

    return pl.DataFrame(rows).sort("gini", descending=True)


# ---------------------------------------------------------------------------
# Utilidades generales
# ---------------------------------------------------------------------------

def print_section(title: str, width: int = 88) -> None:
    """Imprime un delimitador visual de seccion."""
    print("\n" + "=" * width)
    print(title.center(width))
    print("=" * width)


def plot_nulls_by_period(
    df: pl.DataFrame,
    target: str,
    period_col: str = "periodo",
    figsize: tuple = (14, 6),
) -> None:
    """Barras agrupadas: total de registros vs nulos del target por periodo."""
    agg = (
        df.group_by(period_col)
        .agg([
            pl.col(target).is_null().sum().alias("nulos_target"),
            pl.len().alias("total"),
        ])
        .sort(period_col)
    )

    periodos = agg[period_col].to_list()
    totales = agg["total"].to_list()
    nulos = agg["nulos_target"].to_list()

    x = np.arange(len(periodos))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, totales, width, label="Total registros", color="tab:blue", alpha=0.8)
    ax.bar(x + width / 2, nulos, width, label=f"Nulos en {target}", color="tab:red", alpha=0.8)

    ax.set_xticks(x)
    ax.set_xticklabels(periodos, rotation=45, ha="right")
    ax.set_xlabel("Periodo")
    ax.set_ylabel("Cantidad de registros")
    ax.set_title(f"Registros totales vs nulos en {target} por periodo")
    ax.legend()

    plt.tight_layout()
    plt.show()


def plot_cramers_v_matrix(
    df: pl.DataFrame,
    columns: list[str],
    figsize: tuple = (10, 8),
) -> None:
    """Heatmap de Cramers V entre columnas categoricas."""
    n = len(columns)
    matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i, n):
            if i == j:
                matrix[i][j] = 1.0
            else:
                v = cramers_v(df, columns[i], columns[j])
                matrix[i][j] = v
                matrix[j][i] = v

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(matrix, aspect="auto", vmin=0, vmax=1, cmap="YlOrRd")

    for i in range(n):
        for j in range(n):
            color = "white" if matrix[i][j] > 0.6 else "black"
            ax.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center", fontsize=8, color=color)

    ax.set_xticks(range(n))
    ax.set_xticklabels(columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(columns, fontsize=8)
    ax.set_title("Matriz de asociacion (Cramers V)")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.show()


def plot_equity_gap_trend(
    df: pl.DataFrame,
    target: str,
    group_col: str,
    group_a: str,
    group_b: str,
    period_col: str = "periodo",
    figsize: tuple = (14, 6),
) -> None:
    """Linea temporal de la brecha (diff) entre dos grupos por periodo."""
    periodos = df[period_col].unique().sort().to_list()
    diffs = []
    periods_valid = []

    for p in periodos:
        sub = df.filter(pl.col(period_col) == p)
        gap = equity_gap_table(sub, target, group_col, group_a, group_b)
        if gap and gap.get("diff", 0) != 0:
            diffs.append(gap["diff"])
            periods_valid.append(p)

    if not diffs:
        logger.warning("No hay datos suficientes para graficar la brecha temporal.")
        return

    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(periods_valid, diffs, marker="o", color="tab:red", lw=2)
    ax.fill_between(periods_valid, diffs, alpha=0.15, color="tab:red")
    ax.axhline(0, ls="--", lw=0.8, color="gray")
    ax.set_xlabel("Periodo")
    ax.set_ylabel(f"Diferencia en {target} ({group_a} - {group_b})")
    ax.set_title(f"Brecha {group_a} vs {group_b} por periodo")
    ax.tick_params(axis="x", rotation=45)

    plt.tight_layout()
    plt.show()

    logger.info(
        f"Brecha promedio: {np.mean(diffs):.1f} | "
        f"min: {np.min(diffs):.1f} ({periods_valid[np.argmin(diffs)]}) | "
        f"max: {np.max(diffs):.1f} ({periods_valid[np.argmax(diffs)]})"
    )


def plot_nulls_by_period(
    df: pl.DataFrame,
    target: str,
    period_col: str = "periodo",
    figsize: tuple = (14, 6),
) -> None:
    """Barras agrupadas: total de registros vs nulos del target por periodo."""
    agg = (
        df.group_by(period_col)
        .agg([
            pl.col(target).is_null().sum().alias("nulos_target"),
            pl.len().alias("total"),
        ])
        .sort(period_col)
    )

    periodos = agg[period_col].to_list()
    totales = agg["total"].to_list()
    nulos = agg["nulos_target"].to_list()

    x = np.arange(len(periodos))
    width = 0.35

    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(x - width / 2, totales, width, label="Total registros", color="tab:blue", alpha=0.8)
    ax.bar(x + width / 2, nulos, width, label=f"Nulos en {target}", color="tab:red", alpha=0.8)

    for i, (t, n) in enumerate(zip(totales, nulos)):
        pct = n / t * 100 if t > 0 else 0
        if pct > 0:
            ax.text(i, max(t, n) * 1.02, f"{pct:.0f}%", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(periodos, rotation=45, ha="right")
    ax.set_xlabel("Periodo")
    ax.set_ylabel("Cantidad de registros")
    ax.set_title(f"Registros totales vs nulos en {target} por periodo")
    ax.legend()

    plt.tight_layout()
    plt.show()


def correlation_ratio(df: pl.DataFrame, cat_col: str, num_col: str) -> float:
    """Correlation ratio (eta) entre una categorica y una numerica.

    Equivale al R2 de un ANOVA de una via.
    Rango: 0 (sin asociacion) a 1 (asociacion perfecta).
    """
    sub = df.select([cat_col, num_col]).drop_nulls()
    values = sub[num_col].cast(pl.Float64).to_numpy()
    grand_mean = float(np.mean(values))
    n_total = len(values)

    ss_between = 0.0
    groups = sub[cat_col].unique().to_list()
    for g in groups:
        group_vals = sub.filter(pl.col(cat_col) == g)[num_col].cast(pl.Float64).to_numpy()
        n_g = len(group_vals)

        if n_g == 0:
            continue

        ss_between += n_g * (float(np.mean(group_vals)) - grand_mean) ** 2

    ss_total = float(np.sum((values - grand_mean) ** 2))
    if ss_total == 0:
        return 0.0

    return round(float(ss_between / ss_total), 4)


def feature_target_association(
    df: pl.DataFrame,
    feature_cols: list[str],
    target: str,
) -> pl.DataFrame:
    """Calcula correlation ratio (eta) de cada feature categorica con el target."""
    rows = []
    for col in feature_cols:
        eta = correlation_ratio(df, col, target)
        rows.append({"feature": col, "eta": eta})

    result = pl.DataFrame(rows).sort("eta", descending=True)
    return result


def plot_feature_target_association(
    df: pl.DataFrame,
    feature_cols: list[str],
    target: str,
    figsize: tuple = (10, 6),
) -> None:
    """Barplot horizontal de correlation ratio (eta) de features con el target."""
    assoc = feature_target_association(df, feature_cols, target)
    features = assoc["feature"].to_list()
    etas = assoc["eta"].to_list()

    fig, ax = plt.subplots(figsize=figsize)
    colors = ["tab:red" if e > 0.1 else "tab:blue" if e > 0.02 else "tab:gray" for e in etas]
    ax.barh(features[::-1], etas[::-1], color=colors[::-1])
    ax.set_xlabel("Correlation Ratio (eta)")
    ax.set_title(f"Asociacion de features con {target}")
    ax.axvline(0.02, ls="--", lw=0.8, color="gray", alpha=0.5)
    ax.axvline(0.1, ls="--", lw=0.8, color="gray", alpha=0.5)

    plt.tight_layout()
    plt.show()

    for f, e in zip(features[:5], etas[:5]):
        logger.info(f"  {f}: eta={e:.4f}")
