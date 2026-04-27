# Data Card — Resultados Saber 11 (ICFES)

## Información General

| Campo | Valor |
|-------|-------|
| **Nombre del dataset** | Resultados únicos Saber 11 |
| **Fuente** | ICFES — Portal de Datos Abiertos del Gobierno de Colombia |
| **URL** | https://www.datos.gov.co/resource/kgxf-xxbe |
| **Formato de descarga** | JSON (API SODA2) → almacenado en Parquet |
| **Fecha de descarga** | Abril 2026 |
| **Licencia** | Datos abiertos del Gobierno de Colombia (uso libre con atribución) |

## Dimensiones

| Métrica | Valor |
|---------|-------|
| **Filas totales** | 7,109,704 |
| **Columnas** | 51 |
| **Filas con target válido (`punt_global`)** | ~4,500,181 (63.3%) |
| **Periodos cubiertos** | 2010-1 a 2022-4 |
| **Cobertura geográfica** | 33 departamentos de Colombia |

## Variable Objetivo

| Campo | Valor |
|-------|-------|
| **Nombre** | `punt_global` |
| **Tipo** | Numérica continua (Int64) |
| **Rango** | 0 – 495 |
| **Media** | ~253 |
| **Desviación estándar** | ~50 |
| **Nulos** | 36.7% (concentrados en periodos pre-2014) |

## Tipo de Tarea

Regresión supervisada: predecir el puntaje global del examen Saber 11 a partir de variables socioeconómicas del estudiante y características del colegio.

## Variables de Entrada (Features)

### Familia (hipótesis central)
| Variable | Tipo | Ejemplo de valores |
|----------|------|--------------------|
| `fami_estratovivienda` | Categórica ordinal | Estrato 1, Estrato 2, ..., Estrato 6 |
| `fami_educacionmadre` | Categórica ordinal | Ninguno, Primaria incompleta, ..., Postgrado |
| `fami_educacionpadre` | Categórica ordinal | Ninguno, Primaria incompleta, ..., Postgrado |
| `fami_tieneinternet` | Categórica binaria | Si, No |
| `fami_tienecomputador` | Categórica binaria | Si, No |
| `fami_tieneautomovil` | Categórica binaria | Si, No |
| `fami_tienelavadora` | Categórica binaria | Si, No |
| `fami_personashogar` | Categórica ordinal | 1 a 2, Tres, Cuatro, ..., 9 o más |
| `fami_cuartoshogar` | Categórica ordinal | Uno, Dos, Tres, ..., Seis o más |

### Colegio
| Variable | Tipo | Ejemplo de valores |
|----------|------|--------------------|
| `cole_naturaleza` | Categórica nominal | OFICIAL, NO OFICIAL |
| `cole_area_ubicacion` | Categórica nominal | URBANO, RURAL |
| `cole_calendario` | Categórica nominal | A, B, OTRO |
| `cole_jornada` | Categórica nominal | MAÑANA, TARDE, COMPLETA, SABATINA |
| `cole_bilingue` | Categórica binaria | S, N |
| `cole_genero` | Categórica nominal | MIXTO, FEMENINO, MASCULINO |
| `cole_caracter` | Categórica nominal | ACADÉMICO, TÉCNICO, TÉCNICO/ACADÉMICO |

### Estudiante
| Variable | Tipo | Ejemplo de valores |
|----------|------|--------------------|
| `estu_genero` | Categórica binaria | F, M |

### Features derivadas
| Variable | Tipo | Descripción |
|----------|------|-------------|
| `capital_tecnologico` | Numérica (0, 1, 2) | Suma de tiene internet + tiene computador |
| `edad_aprox` | Numérica | Año del periodo - año de nacimiento |

## Variables Excluidas

| Variable(s) | Razón de exclusión |
|-------------|-------------------|
| `punt_ingles`, `punt_matematicas`, `punt_lectura_critica`, `punt_c_naturales`, `punt_sociales_ciudadanas` | Componentes directos del target (leakage). Correlación ~0.99 con la suma |
| `desemp_ingles` | Derivada del puntaje de inglés (posible leakage) |
| `estu_consecutivo`, `cole_cod_dane_*`, `cole_codigo_icfes`, `cole_nombre_*` | Identificadores sin señal generalizable |
| `estu_estudiante`, `estu_pais_reside`, `estu_estadoinvestigacion`, `estu_privado_libertad` | Casi constantes (>99% un solo valor) |
| `estu_cod_depto_*`, `estu_cod_mcpio_*`, `estu_cod_reside_*` | Redundantes con nombres de ubicación |

## Calidad de Datos

| Problema | Detalle | Decisión |
|----------|---------|----------|
| Nulos en target | 36.7% de registros sin `punt_global`, concentrados en periodos pre-2014 (MAR) | Filtrar registros sin target |
| Duplicados | Duplicados exactos detectados por `estu_consecutivo` | Eliminar duplicados, conservar primero |
| Tipos incorrectos | Puntajes vienen como string desde la API JSON | Castear a Int64 |
| Columnas constantes | 4 columnas con >99% un solo valor | Eliminadas |

## Limitaciones y Riesgos

| Riesgo | Descripción |
|--------|-------------|
| **Sesgo de selección** | Solo incluye estudiantes que presentaron el examen Saber 11. No representa a quienes no lo presentaron |
| **Cambio de escala** | El ICFES cambió la escala de calificación alrededor de 2014. Mezclar periodos puede ser problemático |
| **Sesgo geográfico** | Departamentos con pocos registros tienen estimaciones inestables |
| **Sesgo socioeconómico** | Las variables familiares son auto-reportadas y pueden tener sesgo de respuesta |
| **Temporalidad** | Las condiciones socioeconómicas cambian con el tiempo (ej: acceso a internet ha crecido). Concept drift potencial |
| **Uso ético** | Un modelo predictivo de puntajes no debe usarse para discriminar o negar oportunidades a estudiantes |
