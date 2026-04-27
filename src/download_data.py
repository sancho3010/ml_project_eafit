"""
Pipeline de descarga de datos. Resultados Saber 11 (ICFES).

Descarga por lotes desde la API SODA2 de datos.gov.co,
con reintentos por lote y soporte para retomar descargas interrumpidas.
Almacena cada lote en formato Parquet.
"""

import io
import time
from pathlib import Path

import polars as pl
import requests
from loguru import logger


URL_BASE = "https://www.datos.gov.co/resource/kgxf-xxbe.json"
LIMIT = 50_000
MAX_ATTEMPTS = 3
OUTPUT_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"


def get_data_batch(
    batch_number: int,
    url_base: str,
    limit: int,
    offset: int,
    max_attempts: int = MAX_ATTEMPTS,
) -> pl.DataFrame:
    """
    Descarga un lote de datos con reintentos y backoff exponencial.

    Args:
        batch_number: Numero de lote (para logging).
        url_base: URL base de la API sin parametros.
        limit: Cantidad de registros por lote.
        offset: Desplazamiento desde el inicio del dataset.
        max_attempts: Numero maximo de reintentos por lote.

    Returns:
        DataFrame con los datos del lote, o DataFrame vacio si no hay mas datos
        o si todos los intentos fallaron.
    """
    logger.info(f"Se definieron {max_attempts} intentos para este lote...")

    for attempt in range(max_attempts):
        try:
            url = f"{url_base}?$limit={limit}&$offset={offset}"
            response = requests.get(url, timeout=60)
            response.raise_for_status()

            batch = pl.read_json(io.StringIO(response.text))

            if batch.is_empty():
                logger.info("Descarga completa")
                return pl.DataFrame()

            logger.info(
                f"Lote {batch_number:04d} | offset {offset:>8} | {batch.shape} dimensiones"
            )
            return batch

        except Exception as e:
            logger.info(f"Intento {attempt + 1} fallo. Error: {str(e)}")
            time.sleep(2 ** attempt)

    logger.warning(f"Lote fallo tras {max_attempts} intentos. Abortando...")
    return pl.DataFrame()


def download_all(
    url_base: str = URL_BASE,
    limit: int = LIMIT,
    output_dir: Path = OUTPUT_DIR,
    max_attempts: int = MAX_ATTEMPTS,
) -> None:
    """
    Descarga el dataset completo por lotes y los guarda en Parquet.

    Soporta retomar descargas interrumpidas: detecta los lotes ya descargados
    y continua desde el ultimo. Verifica que el ultimo lote no este corrupto.

    Args:
        url_base: URL base de la API.
        limit: Registros por lote.
        output_dir: Directorio donde se guardan los parquet.
        max_attempts: Reintentos por lote.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    lotes_existentes = sorted(output_dir.glob("lote_*.parquet"))
    if lotes_existentes:
        lote = int(lotes_existentes[-1].stem.split("_")[1]) + 1
        offset = lote * limit

        try:
            pl.read_parquet(lotes_existentes[-1])
            logger.info(f"Retomando desde lote {lote:04d} | offset {offset}")
        except Exception:
            logger.warning("Ultimo lote corrupto, re-descargando desde ese punto")
            lote -= 1
            offset = lote * limit
    else:
        lote = 0
        offset = 0
        logger.info("Iniciando descarga desde cero")

    while True:
        batch = get_data_batch(
            batch_number=lote,
            url_base=url_base,
            limit=limit,
            offset=offset,
            max_attempts=max_attempts,
        )

        if batch.is_empty():
            break

        batch.write_parquet(output_dir / f"lote_{lote:04d}.parquet")
        offset += limit
        lote += 1

    logger.info(f"Descarga finalizada. {lote} lotes guardados en {output_dir}")


if __name__ == "__main__":
    download_all()
