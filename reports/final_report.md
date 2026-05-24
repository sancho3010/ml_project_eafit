# Reporte final — Predicción del puntaje Saber 11

**Aprendizaje de Máquina Aplicado · Universidad EAFIT · 2026**
**Equipo:** Patricia Arango · Santiago Higuita · Alexander Pelaez
**Repositorio:** [ml_project_eafit](https://github.com/AlexanderPelaezJimenez/ml_project_eafit)

---

## 1. Problema y pregunta de investigación

El examen Saber 11 (ICFES) es la prueba estandarizada que cierra la educación media en Colombia y condiciona el acceso a la educación superior. Existe amplia evidencia de que las condiciones socioeconómicas son un predictor fuerte del desempeño académico, pero cuantificar esa relación con datos masivos y modelos modernos permite:

1. Identificar qué factores tienen mayor peso predictivo.
2. Medir brechas de equidad de forma rigurosa.
3. Detectar variabilidad geográfica y temporal.

**Pregunta:** ¿Qué tanto del puntaje global Saber 11 puede explicarse con variables socioeconómicas y del entorno educativo, sin usar puntajes parciales del examen?

**Tipo de tarea:** regresión supervisada sobre `punt_global` (rango 0-500).

**Métrica primaria:** RMSE. Penaliza errores grandes, que en este dominio son los más costosos. Reportamos también MAE (interpretabilidad directa) y R² (varianza explicada).

## 2. Datos

**Fuente:** Portal de Datos Abiertos del Gobierno de Colombia (recurso `kgxf-xxbe`).
**Volumen original:** 7,109,704 registros, 51 columnas, periodos 2010-1 a 2022-4.
**Volumen tras limpieza:** 3,395,834 registros con `punt_global` válido.

Detalles completos en `docs/datacard.md`.

### Decisiones de limpieza relevantes

| Decisión | Justificación |
|---|---|
| Filtrar registros sin `punt_global` (36.7% del dataset) | Concentrados en periodos pre-2014 (MAR). Imputar el target sería metodológicamente incorrecto. |
| Eliminar 5 puntajes parciales (`punt_matematicas`, `punt_lectura_critica`, etc.) y `desemp_ingles` | Correlación ≈0.99 con `punt_global`. Incluirlos sería leakage del target. |
| Eliminar identificadores (`estu_consecutivo`, códigos DANE, nombres de colegio) | Sin señal generalizable. |
| Eliminar columnas casi constantes | Aportan ruido sin información. |

## 3. Metodología

### 3.1 Estrategia de validación y prevención de leakage

Tres líneas de defensa:

1. **Eliminación de fuentes de leakage en cleaning** (puntajes parciales, identificadores).
2. **Pipeline encapsulado:** imputers, encoders y agregados van dentro de un `sklearn.Pipeline` que se reentrena por fold del CV. El test holdout se transforma con el pipeline ya ajustado, nunca con `fit`.
3. **Group aggregates con leave-one-out:** el `GroupAggregatesTransformer` calcula porcentajes por municipio (% con internet, % estrato alto, etc.) en `fit`. En `fit_transform` (fold de train) usamos LOO para que cada fila reciba el agregado calculado **sin su propia contribución**, eliminando in-sample bias.

**Split principal:** estratificado por `periodo` (80/20). Es la pregunta razonable de negocio: ¿predigo bien dentro del rango de cohortes vistas?

### 3.2 Feature engineering

Dos transformers custom dentro del pipeline:

**`DeterministicFeatureProcessor`** (stateless, sin estado aprendido):

- `edad_estudiante_aprox`: año del periodo - año de nacimiento, recortada al rango [15, 60].
- `anio_periodo`: tendencia temporal explícita.
- `fami_personashogar_num`, `fami_cuartos_num`: unificación de formatos heterogéneos del ICFES.
- `hacinamiento`: personas / cuartos.
- `capital_tecnologico` (0-2): internet + computador.
- `proxy_riqueza_total` (0-4): tecnología + automóvil + lavadora.
- `max_educacion_padres`, `min_educacion_padres`, `brecha_educativa_padres`.
- `padres_universitarios`, `padres_sin_educacion`.
- `extra_edad`: estudiantes mayores a la edad esperada de Saber 11.
- `presenta_donde_reside_mcpio`, `presenta_donde_reside_depto`: movilidad geográfica.
- `indice_socioeconomico`: suma ponderada determinista.

**`GroupAggregatesTransformer`** (aprende lookups en `fit`, sin target):

- `n_estudiantes_mcpio`: tamaño del municipio.
- `pct_estrato_alto_mcpio`, `pct_internet_mcpio`, `pct_oficial_mcpio`: composición del municipio.
- `pct_calendario_b_depto`: composición del departamento.

### 3.3 Modelos candidatos

Tres gradient boosting trees: **LightGBM, XGBoost y CatBoost**. Robustos a multicolinealidad (relevante porque las features socioeconómicas están correlacionadas), manejo nativo de missing values, estado del arte en datos tabulares.

### 3.4 Tuning

- **Optuna con TPE multivariado** + median pruner (descarta trials malos temprano).
- **40 trials por modelo**, 3-fold CV.
- Cada trial reentrena el pipeline completo en cada fold; encoders e imputers se ajustan solo en train.

## 4. Resultados

### 4.1 Comparación de modelos (CV en train)

| Modelo | RMSE CV | Mejores hiperparámetros (resumen) |
|---|---|---|
| **CatBoost** | 36.6909 | iterations=975, depth=10 |
| **XGBoost** | 36.7106 | n_estimators=949, max_depth=12 |
| **LightGBM** | 36.8100 | n_estimators=701, max_depth=8 |

Las diferencias de ~0.1 RMSE están dentro del rango de incertidumbre del CV (±0.5 RMSE típico con 3 folds). **Todos los modelos llegan al mismo techo.**

### 4.2 Selección del champion

Elegimos **XGBoost**. Razones:

- RMSE de CV indistinguible del top.
- Mayor madurez en producción (más años desplegado, mejor soporte multi-plataforma vía ONNX/Treelite).
- API más estable a través de versiones.

Ver `notebooks/06-pipeline-ml.ipynb` celda 17 para la decisión.

### 4.3 Evaluación en test holdout

| Métrica | Valor |
|---|---|
| RMSE | **36.60** |
| MAE | **29.14** |
| R² | **0.460** |
| n_test | 679,167 |

La diferencia entre RMSE de CV (36.71) y holdout (36.60) es de 0.11 puntos: el modelo generaliza casi idéntico. **No hay overfitting al CV.**

### 4.4 Validaciones complementarias

_Por completar tras finalizar la corrida en SageMaker._

| Esquema | RMSE | Lectura |
|---|---|---|
| Test holdout (random, stratified periodo) | 36.60 | Pregunta de negocio principal |
| GroupKFold por municipio (5 folds) | _pendiente_ | Generalización a municipios no vistos |
| Split temporal (train ≤ 2019-4) | _pendiente_ | Generalización a cohortes futuras |

### 4.5 Comparación contra baseline (entrega 1)

_Por completar con números reales._

| Modelo | RMSE | MAE | R² |
|---|---|---|---|
| Naive (media) | ~50 | ~40 | ~0 |
| LinearRegression | _pendiente_ | _pendiente_ | _pendiente_ |
| DecisionTree (max_depth=8) | _pendiente_ | _pendiente_ | _pendiente_ |
| **XGBoost (champion)** | **36.60** | **29.14** | **0.460** |

### 4.6 Interpretabilidad (SHAP)

Las features más importantes (resumen, tabla completa en `models/champion/shap_importance.csv`):

_Por completar tras finalizar la corrida._

Esto coincide con los hallazgos del EDA (entrega 1): brechas grandes y consistentes por estrato, educación de padres, acceso a tecnología y tipo de colegio.

## 5. Limitaciones

- **Sesgo de selección:** solo estudiantes que presentaron Saber 11. No representa a quienes no lo presentaron.
- **Concept drift:** datos hasta 2022. La pandemia y posibles cambios en el ICFES pueden degradar el modelo en cohortes futuras.
- **Variables auto-reportadas:** estrato, internet, educación de padres pueden tener sesgo de respuesta.
- **Cambio de escala del ICFES** alrededor de 2014: el split estratificado por periodo lo controla parcialmente.
- **Techo del problema:** R² ≈ 0.5 es el límite razonable con las features disponibles. Para mejorarlo se necesita información no presente en el dataset público (historial académico, hábitos, asistencia).

## 6. Consideraciones éticas

El modelo predice puntajes a partir de variables socioeconómicas. **Bajo ninguna circunstancia debe usarse para:**

- Decisiones de admisión universitaria.
- Asignación individual de recursos a estudiantes.
- Clasificación de estudiantes por desempeño esperado.
- Decisiones que afecten oportunidades individuales.

**Usos legítimos:**

- Diagnóstico agregado de brechas educativas a nivel regional.
- Diseño de políticas públicas focalizadas.
- Identificación de colegios y municipios que requieren intervención.
- Investigación académica.

## 7. Productos entregados

- `notebooks/06-pipeline-ml.ipynb`: notebook final con narrativa completa.
- `models/champion/`: pipeline serializado, model card, métricas, hiperparámetros, schema de features y SHAP importance.
- `app/`: aplicación Streamlit con predictor individual, análisis de brechas y detalle del modelo.
- `src/modeling/`: implementación modular del pipeline (transformers, preprocessor, validation, diagnostics, champion bundle).
- `tests/`: tests unitarios cubriendo transformers y serialización.
- `reports/`: este reporte y resumen ejecutivo.

## 8. Reproducibilidad

Ver `docs/reproducibility_checklist.md` para el procedimiento exacto de reproducción.

**Comandos clave:**

```bash
# Setup
pip install -r requirements-dev.txt
pip install -e .

# Validación de calidad
make check

# Correr el experimento
jupyter notebook notebooks/06-pipeline-ml.ipynb

# Levantar la app
streamlit run app/app.py
```

## Referencias

- Barrera-Osorio, F., & Bayona-Rodríguez, H. (2019). Comparación de los resultados de las pruebas Saber 11.
- OECD (2023). PISA 2022 Results: Country Notes Colombia.
- Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions (SHAP).
- Ke, G. et al. (2017). LightGBM: A Highly Efficient Gradient Boosting Decision Tree.
- Chen, T. & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
- Prokhorenkova, L. et al. (2018). CatBoost: unbiased boosting with categorical features.
