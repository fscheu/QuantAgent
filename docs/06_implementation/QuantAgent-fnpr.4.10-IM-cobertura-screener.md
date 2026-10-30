# QuantAgent-fnpr.4.10 · CIK históricos y cobertura del screener (2010-03 a 2018-12)

Reemplaza la medición de `QuantAgent-fnpr.4.8-IM-cobertura-screener` (no se reescribe, es histórica). Reproducir: `python scripts/m5/cobertura_screener.py --csv docs/06_implementation/QuantAgent-fnpr.4.10-IM-cobertura-screener.csv` (~25 min con este snapshot; escribe el CSV de esta carpeta). Mismo método que 4.8.

## Qué cambió

- `quantagent/data/cik_historicos.csv` (ticker, cik, nombre, fuente): CIK de 266 de los 287 tickers que `company_tickers.json` no trae (renombrados o deslistados). `crear_snapshot_fundamentales` lo usa como segundo mapa, después del fallback punto→guion. `descargar_facts_cik` reintenta 3 veces (espera 2 s y 4 s) ante `URLError` o timeout; un `HTTPError` (404) no se reintenta.
- Snapshot `fundamentales-sp500-2026-10` bajado de nuevo (incluye el arreglo de dedup de M5-32e): 877 símbolos (antes 637), `sin_cik` 21 (antes 287), `fallidos` 26 (antes 0), todos `HTTPError` (empresas anteriores a XBRL sin companyfacts): ACS, AKS, APCC, ASN, AT, AW, BDK, BRL, CBSS, CCTYQ, CFC, CMX, DJ, EOP, HPC, JAVA, KSE, LO, MIL, NCC, ROH, SAF, SBL, SBNY, SLR, TEK. El anterior quedó en `local-data/fundamentales-sp500-2026-10.old-pre-4.10`.

## Cómo se armó el CSV de CIK

1. Nombre de la empresa: la composición de `sp500-2026-10` no trae nombres (solo `sp500.parquet`, con los miembros vigentes). Se usó la tabla "Removed / Added" de Wikipedia (*Historical components of the S&P 500*, wikitext) y, para los tickers que no figuran ahí, nombres escritos a mano (conocimiento propio, sin fuente pública; los valida el paso 3).
2. Candidatos: `www.sec.gov/Archives/edgar/cik-lookup-data.txt`, por nombre normalizado (sin sufijos Inc/Corp/Co, sin `/DE/`).
3. Confirmación contra `data.sec.gov/submissions/CIK##########.json`: el `name` o un `formerNames` tiene que coincidir con el nombre **y** haber estado vigente durante la pertenencia al índice (ventana de la composición ±1 año). Esto descartó, por ejemplo, el CIK actual de "Bed Bath & Beyond" (Overstock tomó el nombre en 2023) a favor del 886158.
4. Con más de un CIK confirmado: el que tenga hechos 10-K/10-Q en `companyfacts` dentro del período, si el segundo tiene menos de una décima parte. Si no, desempate manual para diez tickers (ARNC, BHGE, BRCM, CCE, DTV, EVHC, KRFT, PSKY, VIAB, WRK), con `fuente` marcada "(desempate manual)": se eligió el registrante que cotizó con ese ticker en el período. Revisar CCE (el ticker lo usaron dos empresas distintas: la anterior a 2010 no tiene XBRL) y WRK (WRKCo Inc. 1636023 fue el registrante hasta 2018; se dejó WestRock Co 1732845).
5. Los seis casos del contrato coinciden: ANTM 1156039, ABC 1140859, ATVI 718877, AET 1122304, ALXN 899866, BK 1390777.

**Sin resolver (21, fuera del CSV, siguen en `sin_cik`):** AGN (Allergan Inc. y Allergan plc usaron el mismo ticker; cada uno con datos), y por varios CIK con el mismo nombre sin forma de desempatar: ABI, ACAS, BOL, BSC, CBH, CTX, EDS, FRC, HCR, HOT, KATE, LEHMQ, MEDI, MEL, PGL, TRB (sin nombre encontrado), UST, UVN, WCG, WWY.

## Cobertura anual (% promedio de las 4 fechas del año), antes (4.8) → después

| sector                 | 2010    | 2011    | 2012    | 2013    | 2014    | 2015    | 2016    | 2017    | 2018    |
|:-----------------------|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|:-------:|
| Communication Services | -       | -       | -       | -       | -       | -       | -       | -       | 29 → 46 |
| Consumer Discretionary | 15 → 27 | 35 → 62 | 43 → 66 | 44 → 69 | 47 → 71 | 50 → 74 | 52 → 75 | 36 → 76 | 24 → 69 |
| Consumer Staples       | 15 → 20 | 27 → 38 | 29 → 38 | 26 → 40 | 28 → 46 | 36 → 49 | 39 → 46 | 28 → 48 | 16 → 45 |
| Energy                 | 19 → 33 | 24 → 48 | 26 → 47 | 31 → 54 | 31 → 57 | 28 → 59 | 33 → 64 | 28 → 62 | 24 → 50 |
| Financials             | 22 → 28 | 37 → 49 | 46 → 55 | 50 → 60 | 52 → 64 | 60 → 70 | 61 → 71 | 62 → 70 | 61 → 68 |
| Health Care            | 28 → 45 | 37 → 71 | 43 → 72 | 44 → 70 | 46 → 77 | 47 → 75 | 53 → 79 | 44 → 81 | 28 → 73 |
| Industrials            | 34 → 45 | 47 → 69 | 59 → 77 | 63 → 83 | 65 → 83 | 68 → 87 | 68 → 85 | 48 → 84 | 34 → 80 |
| Information Technology | 24 → 36 | 41 → 66 | 54 → 80 | 55 → 81 | 58 → 85 | 55 → 82 | 58 → 85 | 42 → 87 | 30 → 75 |
| Materials              | 14 → 27 | 19 → 52 | 28 → 64 | 42 → 76 | 43 → 78 | 47 → 77 | 55 → 77 | 39 → 80 | 19 → 75 |
| Real Estate            | -       | -       | -       | -       | -       | -       | 56 → 74 | 48 → 69 | 43 → 62 |
| Utilities              | 26 → 29 | 45 → 53 | 59 → 68 | 60 → 82 | 62 → 77 | 68 → 76 | 75 → 79 | 67 → 80 | 62 → 78 |

El "después" mezcla dos cambios: los CIK nuevos (este ticket) y el arreglo de dedup de M5-32e (mergeado antes; el snapshot "antes" era anterior). El de 2017-2018 viene sobre todo de M5-32e.

**Sector-años bajo 60%: 28 de 85** (antes 71). Quedan: Communication Services 2018 (46); Consumer Discretionary 2010 (27); Consumer Staples 2010-2018 (20, 38, 38, 40, 46, 49, 46, 48, 45); Energy 2010-2015 y 2018 (33, 48, 47, 54, 57, 59, 50); Financials 2010-2012 (28, 49, 55); Health Care 2010 (45); Industrials 2010 (45); Information Technology 2010 (36); Materials 2010-2011 (27, 52); Utilities 2010-2011 (29, 53). Detalle por fecha en el CSV.

## Causas

Sobre todos los miembros-fecha (17.830): elegible 65% (antes 43%), faltan_metricas 31% (32%), sin_cik 0,9% (25%), sin_precio 3,2% (0,6%). Por año (% de miembros):

| año | elegible | faltan_metricas | sin_cik | sin_precio |
|---:|---:|---:|---:|---:|
| 2010 | 33 | 60 | 2 | 6 |
| 2011 | 58 | 36 | 1 | 5 |
| 2012 | 64 | 32 | 1 | 3 |
| 2013 | 69 | 29 | 1 | 1 |
| 2014 | 71 | 27 | 1 | 1 |
| 2015 | 73 | 25 | 1 | 1 |
| 2016 | 75 | 24 | 1 | 0 |
| 2017 | 76 | 24 | 0 | 0 |
| 2018 | 69 | 20 | 0 | 11 |

Causas que quedan:

- **`faltan_metricas` es la dominante en todos los sectores.** Las que más faltan son `margen_op_yoy` / `margen_pretax_yoy` y `aceleracion_ingresos` (necesitan la serie `ingresos` completa de 6 a 8 trimestres) y después `roic_4t` o `crecimiento_eps`. En Consumer Staples afecta al 56% de sus miembros-fecha (764 de 1.362); en Financials son 1.122 y en Consumer Discretionary 876. En 2010 el 60% de los miembros falla por eso: los facts XBRL de las grandes empresas empiezan en 2009-2010, así que a 2010-03 no hay 6 trimestres (AMZN, AAPL, MSFT, NVDA, BAC, JPM, BA siguen excluidas a 2010).
- **`sin_precio` subió** (de 0,6% a 3,2%): los tickers recuperados, que ya salieron del índice, ahora cuentan como miembros con fundamentales pero varios no tienen precio en `acciones-sp500-2026-10b` (no se diagnosticó ticker por ticker). Pesa 11% en 2018, sobre todo Energy (116), IT (103), Consumer Discretionary (84), Health Care (78) y Communication Services (22 de 52: XLC arranca en 2018-06 y falta rs6).
- **`sin_cik` residual** (155 miembros-fecha: Health Care 40, Energy 30, Consumer Discretionary 26, Utilities 24, Consumer Staples 22, Materials 7, Financials 6) son los 21 sin resolver más los fallidos pre-XBRL.
- Consumer Staples y Energy no llegan a 60% en ningún año (salvo Energy 2016-2017): sus métricas de margen y crecimiento faltan más que en el resto; no se diagnosticó el tag concreto.

Tres métricas que más faltan por sector (entre los excluidos por `faltan_metricas`):

| sector | 1ª | 2ª | 3ª |
|:--|:--|:--|:--|
| Communication Services | crecimiento_eps (6) | rs6 (5) | roic_4t (2) |
| Consumer Discretionary | margen_op_yoy (870) | aceleracion_ingresos (800) | roic_4t (501) |
| Consumer Staples | margen_op_yoy (760) | aceleracion_ingresos (725) | crecimiento_eps (402) |
| Energy | margen_op_yoy (532) | aceleracion_ingresos (433) | roic_4t (326) |
| Financials | margen_pretax_yoy (1116) | aceleracion_ingresos (1026) | crecimiento_eps (539) |
| Health Care | margen_op_yoy (459) | aceleracion_ingresos (440) | crecimiento_eps (233) |
| Industrials | margen_op_yoy (464) | aceleracion_ingresos (428) | crecimiento_eps (311) |
| Information Technology | margen_op_yoy (517) | aceleracion_ingresos (502) | crecimiento_eps (378) |
| Materials | margen_op_yoy (300) | aceleracion_ingresos (274) | roic_4t (148) |
| Real Estate | margen_op_yoy (94) | aceleracion_ingresos (84) | roic_4t (45) |
| Utilities | margen_op_yoy (285) | aceleracion_ingresos (280) | crecimiento_eps (151) |
