# M5-35: precios de acciones 2019–hoy

Snapshot `acciones-sp500-2026-10c` en `$QUANTAGENT_SNAPSHOT_DIR` (VM: `/home/baiscf/repos/QuantAgent/local-data`), generado el 2026-10-31 para 2019-01-01..2026-10-31; última sesión cerrada: **2026-10-30**. Tiene **611 símbolos, 1.155.273 filas**. [Manifiesto con hashes](../../tests/fixtures/snapshots/acciones-sp500-2026-10c.json).

Reutilicé `miembros`, los alias de `acciones.py`, `download_daily` y `write_snapshot` del [mecanismo de 10b](QuantAgent-fnpr.1.8-IM-reuso-tickers-acciones.md). El creador 2007–2018 tiene fechas y WIKI fijos: se usó un script ad-hoc, sin modificarlo. Yahoo/yfinance 1.7.0, `auto_adjust=False`, índice diario sin zona, columnas `open/high/low/close/adj_close/volume`; sin WIKI ni empalmes. Tres intentos por ticker; IPv4 solo en la sesión ad-hoc. El manifiesto registra rango solicitado, 671 tickers, alias y 60 sin descarga. Los Parquet quedan fuera de Git.

Universo: unión de miembros en cada sesión NYSE de `sp500-2026-10`. **La composición termina el 2026-08-18**: desde ahí `miembros` mantiene la última lista. No se certifican altas/bajas posteriores; ese límite debe revisarse antes de usar etapa 3 o la reserva H4. Este ticket descarga precios y no ejecuta evaluaciones.

Cobertura: un miembro tiene precio si al menos una sesión de pertenencia del año/sector tiene `close` y `adj_close` no nulos. El sector es punto en el tiempo, incluidas las reclasificaciones de 2023; un ticker puede contar en dos sectores ese año, por eso el total anual se deduplica. Tener una fila no implica cobertura completa.

| Sector (con/sin precio) | 2019 | 2020 | 2021 | 2022 | 2023 | 2024 | 2025 | 2026 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Communication Services | 19/10 | 19/8 | 19/8 | 21/9 | 20/5 | 19/3 | 22/4 | 26/1 |
| Consumer Discretionary | 62/5 | 64/4 | 62/3 | 61/0 | 58/0 | 54/0 | 53/0 | 48/0 |
| Consumer Staples | 31/2 | 31/2 | 30/2 | 31/2 | 36/2 | 36/2 | 36/2 | 37/0 |
| Energy | 22/9 | 21/7 | 20/6 | 19/4 | 19/4 | 20/4 | 21/2 | 21/1 |
| Financials | 64/8 | 59/7 | 61/7 | 62/7 | 71/7 | 73/3 | 77/2 | 78/0 |
| Health Care | 55/8 | 58/8 | 61/6 | 62/4 | 65/2 | 63/2 | 60/1 | 60/1 |
| Industrials | 68/6 | 72/5 | 72/4 | 72/3 | 79/1 | 81/2 | 80/1 | 84/1 |
| Information Technology | 64/10 | 66/8 | 72/8 | 72/6 | 74/3 | 67/3 | 70/2 | 74/0 |
| Materials | 26/2 | 25/2 | 25/2 | 27/2 | 26/2 | 27/1 | 29/0 | 25/0 |
| Real Estate | 29/4 | 27/4 | 26/4 | 29/4 | 29/3 | 29/3 | 29/2 | 32/0 |
| Utilities | 29/0 | 28/0 | 28/0 | 30/0 | 30/0 | 31/0 | 31/0 | 31/0 |
| Total único | 469/64 | 470/55 | 476/50 | 486/41 | 493/29 | 500/23 | 508/16 | 516/4 |
| % miembro-días con precio | 88.41 | 90.22 | 91.46 | 93.19 | 95.02 | 96.15 | 97.47 | 98.95 |

En todo el tramo: **603 de 671 miembros con precio; 68 sin precio durante su pertenencia**. Se cubren **929.980/992.112 miembro-días (93,74 %)**; 595 tickers tienen todas sus sesiones, 8 cobertura parcial. Detalle reproducible por año/sector/ticker en `$QUANTAGENT_SNAPSHOT_DIR/acciones-sp500-2026-10c/cobertura.csv` (`dias`, `precios`).

Los 68 sin precio, agrupados por sector:

- **Communication Services**: ATVI, CBS, DISCA, DISCK, DISH, IPG, SATS, TWTR, VIAB, VIAC.
- **Consumer Discretionary**: FL, HBI, JWN, LEG, TIF.
- **Consumer Staples**: K, WBA.
- **Energy**: APC, COG, CTRA, CXO, HES, MRO, NBL, NFX, PXD, XEC.
- **Financials**: CMA, DFS, ETFC, FI, FRC, PBCT, RE, SBNY, SIVB, STI.
- **Health Care**: ABMD, AGN, ALXN, CELG, CERN, CTLT, HOLX, VAR, WCG.
- **Industrials**: ARNC, CDAY, DAY, INFO, KSU, LLL, NLSN, RTN.
- **Information Technology**: ANSS, CTXS, FLIR, JNPR, MXIM, NLOK, QRVO, RHT, TSS, XLNX.
- **Materials**: SEE, WRK.
- **Real Estate**: DRE, PEAK.

De esos 68, ocho sí tienen archivo pero ninguna fila durante la pertenencia: **APC, INFO, LEG, NFX, QRVO, SATS, SBNY, STI**. Yahoo devuelve series posteriores o demasiado cortas; no se cuentan como recuperados. De los ocho parciales, AVB, EA, EQR y WBD tienen una sola sesión válida; PSKY 15/310, FOX 1.921/1.969, FOXA 1.922/1.969 y VLTO 772/774. No se rellenan faltantes ni se valida aquí la identidad legal de cada serie/alias. La cobertura menor en los primeros años deja sesgo de supervivencia.

Fuente propuesta (sin implementar): [Sharadar Stock Prices/SEP](https://sharadar.com/docs/stocks), que documenta OHLCV de acciones activas y deslistadas; cruzar sus identificadores permanentes y [acciones corporativas](https://sharadar.com/docs/actions) para resolver renombres y tickers reutilizados. Confirmar los 68 y los parciales contra esa fuente antes de usar resultados para paper; no se contrató ni descargó Sharadar.

Validación sin red (entorno del ticket):
```bash
python scripts/m5/cobertura_precios.py
python -m quantagent.cli data snapshot verify --name acciones-sp500-2026-10c
python -m pytest -q tests/test_cobertura_precios.py
python -m ruff check --no-fix scripts/m5/cobertura_precios.py tests/test_cobertura_precios.py
scripts/loop/smoke.sh
```
Resultado: hashes OK (611 símbolos, 1.155.273 filas); 1 test passed; lint OK; smoke exit 0 (9 trades). El test cubre precios fuera de pertenencia, valores nulos, ausencia de archivo y cambio de sector.
