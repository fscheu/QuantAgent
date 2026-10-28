# QuantAgent-fnpr.1.6: foto diaria de consenso de EPS desde Yahoo Finance

## Qué se guarda

Se guarda una foto diaria en formato Parquet (`consenso-eps-yahoo/<fecha>.parquet`) con las estimaciones de tendencia y revisiones de EPS para los miembros del S&P 500 obtenidos mediante `yfinance`. Cada fila representa un período analizado (`0q`, `+1q`, `0y`, `+1y`) uniendo las propiedades `Ticker.eps_trend` y `Ticker.eps_revisions` en un esquema con columnas en snake_case:

- `symbol`: símbolo del activo (preservando formato original con punto, ej. `BRK.B`).
- `period`: período de consenso (`0q`, `+1q`, `0y`, `+1y`).
- `fecha_foto`: fecha de la captura (`YYYY-MM-DD`).
- Tendencia (`eps_trend`): `current`, `d7`, `d30`, `d60`, `d90`, `currency`.
- Revisiones (`eps_revisions`): `up_7d`, `up_30d`, `down_7d`, `down_30d`.

## Por qué Yahoo y no FMP

El plan gratuito de Financial Modeling Prep (FMP) limita las estimaciones a un subconjunto restringido de megacaps (retornando error 402 en empresas como LLY, CAT, DE, PANW, AOS, ALB), imposibilitando la captura de un universo completo como el S&P 500. Yahoo Finance a través de `yfinance` permite acceder a la tendencia y revisiones recientes de EPS para la gran mayoría del índice sin costo ni clave de API. El cliente de FMP se conserva en el repositorio pero la fuente por defecto del CLI pasa a ser `yahoo`.

## Límites

- **Fuente no oficial:** los endpoints de Yahoo Finance utilizados por `yfinance` no tienen SLA formal y su estructura puede variar sin previo aviso.
- **Uso personal e investigación:** destinado exclusivamente a pruebas y generación de historia propia de consenso para investigación cuantitativa.

## Corrida real

- **Fecha:** 2026-10-28
- **Comando:** `data consenso foto --fuente yahoo --archivo universos/sp500-actual.txt`
- **Destino:** `local-data/consenso-eps-yahoo/2026-10-28.parquet`
- **Total símbolos en universo:** 503
- **Símbolos con foto guardada:** 501 (2.004 filas, 4 períodos por símbolo)
- **Símbolos sin datos (2):** `FOX` y `L`
- **Reejecución con misma fecha:** falla con `FileExistsError` sin pisar los datos existentes.
