# QuantAgent-fnpr.1.3 — Snapshot de acciones del S&P 500 (2007-2018)

Fuentes, reglas y cobertura real de `acciones-sp500-2026-10`. Contrato: ticket `QuantAgent-fnpr.1.3` (decisión D1 de Fede: fuente gratis; si una hipótesis de L28 pasa la etapa 2 se confirma con Sharadar antes de paper).

## Fuentes y reglas

Código: [`quantagent/data/acciones.py`](../../quantagent/data/acciones.py). Comandos: `quantagent data acciones create --name acciones-sp500-2026-10` y `quantagent data acciones cobertura --name acciones-sp500-2026-10` (guarda `cobertura.json` y `cobertura.csv` en la carpeta del snapshot).

- Universo: unión de `miembros('sp500-2026-10', fecha)` sobre todas las sesiones NYSE entre 2007-01-01 y 2018-12-31 (780 tickers).
- Por ticker, sin empalmar fuentes: si Yahoo (`auto_adjust=False`, desde 2006-01-01) cubre el período de pertenencia (tolerancia de 7 días en cada punta) se usa Yahoo. Si no, Quandl WIKI (`WIKI_PRICES.csv`, Kaggle `marketneutral/quandl-wiki-prices-us-equites`). Si WIKI tampoco lo tiene pero Yahoo trae algo parcial, se usa ese parcial.
- Yahoo se recorta a 2018-12-31. WIKI se mapea a las columnas de `read_snapshot`: `open/high/low/close = adj_*`, `adj_close = close` (factor 1: ya ajustado por splits y dividendos), `volume = adj_volume`.
- El manifiesto guarda `fuentes` (ticker → `yahoo` | `wiki`). `write_snapshot` recibe `fuentes` como parámetro opcional; sin él el manifiesto no cambia.
- Alias en [`alias_tickers.csv`](../../quantagent/data/tablas/alias_tickers.csv): `BF.B`/`BRK.B` (Yahoo los escribe con guion) y `SUNEQ`→`SUNE` en WIKI.
- Manifiesto versionado: `tests/fixtures/snapshots/acciones-sp500-2026-10.json`. `verify_snapshot` OK: 691 símbolos, 2.688.776 filas (478 Yahoo, 213 WIKI).

## Cobertura real

Miembro-días 2007-2018 con precio: **93,84 %** (1.416.479 de 1.509.486).

| 2007 | 2008 | 2009 | 2010 | 2011 | 2012 | 2013 | 2014 | 2015 | 2016 | 2017 | 2018 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 82,77 | 87,29 | 90,89 | 93,47 | 94,88 | 96,49 | 98,14 | 99,04 | 99,20 | 99,21 | 99,41 | 85,19 |

**Sin ningún precio (89 de 780):** AABA, ABI, ABKFQ, ACS, ADCT, ANRZQ, APCC, ASN, ATGE, AV, AW, AYE, BDK, BJS, BMC, BMET, BNI, BOL, BRL, BSC, BTUUQ, CBE, CBH, CBSS, CCTYQ, CEG, CEPH, CFC, CITGQ, CMVT, CMX, CTX, CVH, DJ, EDS, EKDKQ, EOP, FDC, GR, HCR, HET, HMA, HNZ, HPC, IAC, JAVA, KATE, KSE, LEHMQ, MEDI, MEE, MEL, MER, MFE, MHS, MOLX, MTLQQ, NCC, NOVL, NVLS, NYX, OMX, PBG, PD, PGL, PGN, PTV, ROH, RSHCQ, RX, SAF, SBL, SGP, SLR, SOV, TEK, TIE, TIN, TLAB, TRB, TSG, TXU, UVN, WAMUQ, WFT, WWY, WYE, WYND, XTO.

**Terminan antes de su salida del índice (105 tickers):** casi todos son empresas que siguieron en el índice hasta 2018-12 pero sin Yahoo (adquiridas después): WIKI se congeló el 2018-03-27 y les faltan 192 sesiones (p. ej. ABC, ATVI, FB, WBA, XLNX). El resto son salidas posteriores a esa fecha (p. ej. MXIM 19, MON 49, TWX 55). Detalle por ticker en `cobertura.json`.

**Saltos de cierre ajustado >50 % de un día a otro dentro de 2007-2018:** 166 en 47 tickers, para revisar a mano (no se corrigen). Los más cargados: CPWR (50), EP (41), FMCC (6), GNW (6), FNMA (5). No se investigó la causa de cada uno. Lista completa en `cobertura.json`.

## Chequeo de deslistadas

Un símbolo que dejó de cotizar conserva sus precios hasta su último día: WFM (WIKI) fue miembro hasta 2017-08-25 y su precio llega a 2017-08-28; SUNEQ (alias `SUNE`) cotiza hasta 2016-04-21. Cubierto por `tests/test_acciones.py` con LEH/LEHMQ (precios hasta 2008-09-15).

## Limitaciones

- WIKI está congelado en 2018-03-27: las salidas posteriores quedan cortas si Yahoo no las trae.
- LEH/LEHMQ, BSC, WAMUQ, MER, CFC y otras quiebras o fusiones de 2008 no están en WIKI ni en Yahoo: la lista "sin precio" tiene sesgo de supervivencia residual (cobertura 82,8 % en 2007 contra 99 % desde 2014; el 2018 baja a 85 % por el corte de WIKI).
- Yahoo es para uso personal; sus tickers reasignados a otra empresa podrían contaminar un símbolo si su historia cubre todo el período (no se detectó, pero no hay control específico).
- Los precios de Yahoo salen de la descarga del día de creación; el snapshot se congela y no se vuelve a bajar.
