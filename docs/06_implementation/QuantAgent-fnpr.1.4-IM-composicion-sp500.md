# QuantAgent-fnpr.1.4: Composición histórica del S&P 500 y sector GICS por fecha

Implementación de `quantagent/data/sp500.py` y tablas versionadas en `quantagent/data/tablas/` para consultar la composición punto en el tiempo del S&P 500 y el sector GICS vigente de cada miembro en cualquier fecha (lote L22, `lote/m5-datos`).

## Fuente y snapshot congelado

- **Repositorio:** `fja05680/sp500` (MIT), fijado en el commit `a2430f2af0c79ddf0748e91de11bdeb1616ab5a7`.
- **Archivos descargados:**
  - `S&P 500 Historical Components & Changes (Updated).csv` -> guardado como `composicion.parquet` (2.720 filas, desde `1996-01-02` hasta `2026-08-18`).
  - `sp500.csv` -> guardado como `sp500.parquet` (503 miembros actuales con su `GICS Sector`).
- **Snapshot:** `sp500-2026-10` en `$QUANTAGENT_SNAPSHOT_DIR/sp500-2026-10/` con manifiesto versionado en `tests/fixtures/snapshots/sp500-2026-10.json`.

## Reglas de consulta

1. **`miembros(nombre, fecha)`:** busca en `composicion.parquet` la última fila con `date <= fecha` (sin mirar ninguna fila posterior a `fecha`) y devuelve la lista de símbolos de esa fila. Si `fecha` es anterior a la primera fecha del snapshot (`1996-01-02`), lanza `SnapshotError`.
2. **`sector(nombre, ticker, fecha)`:**
   - Parte del sector base: `GICS Sector` en `sp500.parquet` si el ticker es miembro actual, o `sector` en `quantagent/data/tablas/gics_salidos.csv` si salió del índice (con `fuente != 'desconocido'`). Si no figura en ninguna tabla, devuelve `None`.
   - Recorre `quantagent/data/tablas/gics_reclasificaciones.csv` para ese ticker ordenando por `desde` de más nueva a más vieja (`2023-03-17`, `2018-09-24`, `2016-09-01`). Si `fecha < desde`, reemplaza el sector por `sector_anterior`.

## Cobertura y tickers sin sector

- **Período 2007-01-01 a 2018-12-31 (universo in-sample):** 780 tickers únicos fueron miembros del índice (382 siguen en `sp500.csv` y 398 salieron). Los 398 miembros salidos están cubiertos en `gics_salidos.csv` con `fuente='manual'` (**0 desconocidos** en los 398).
- **Fuera de 2007-2018:** sobre los 1.209 tickers únicos de toda la serie `1996-2026`, hay 308 tickers sin sector en las tablas (277 que salieron antes de `2007-01-01` y 31 que entraron y salieron entre `2019` y `2026`, como `BBWI`, `CTLT`, `ENPH`, `ETSY`, `FRC`, `LUMN`, `PARA`, `PEAK`, `SEDG`, `SBNY`, `VIAC`), para los cuales `sector()` devuelve `None`.

## Limitaciones de la fuente y de la indexación por ticker

1. **Nombres que cambian de ticker y sufijos de liquidación (`Q`):** la fuente `fja05680/sp500` aplica ajustes retroactivos desparejos en algunos símbolos históricos. Once empresas que quebraron figuran en la serie histórica con su ticker OTC de liquidación en vez del símbolo que tenían en bolsa mientras integraban el índice (por ejemplo Lehman Brothers aparece como `LEHMQ` hasta `2008-09-16` en lugar de `LEH`, Washington Mutual como `WAMUQ` en lugar de `WM`, General Motors pre-2009 como `MTLQQ`, Circuit City como `CCTYQ`, Kodak como `EKDKQ`, RadioShack como `RSHCQ`, Ambac como `ABKFQ`, CIT Group como `CITGQ`, Peabody como `BTUUQ`, Alpha Natural Resources como `ANRZQ` y MEMC/SunEdison como `SUNEQ`). Otras empresas aparecen con un ticker renombrado posterior (`AABA` por Yahoo! `YHOO`, `KATE` por Liz Claiborne `LIZ`, `VIAV` por JDS Uniphase `JDSU`, `ATGE` por DeVry `DV`), mientras que en otros cambios coexisten ambos símbolos según el tramo (`FB` hasta `2022-06-08` y `META` desde `2022-06-09`; `KORS` y `CPRI`; `HCP`, `PEAK` y `DOC`; `CBS`/`VIAB` y `VIAC`/`PARA`).
2. **Tickers reutilizados por empresas distintas:** la tabla indexa por `ticker` (sin CUSIP/CIK por fecha). Símbolos de una o dos letras fueron usados en distintas décadas por compañías diferentes: por ejemplo, `H` en `2006-2007` correspondía a Realogy Corp (`Financials`) y luego fue asignado a Hyatt Hotels; `SE` entre `2007` y `2017` era Spectra Energy (`Energy`) y hoy corresponde a Sea Limited; `C` antes de `1998` fue Chrysler y luego Citigroup. Como `gics_salidos.csv` registra el sector al salir del índice en la ventana `2007-2018`, el sector refleja a la empresa que integró el S&P 500 en ese período.
