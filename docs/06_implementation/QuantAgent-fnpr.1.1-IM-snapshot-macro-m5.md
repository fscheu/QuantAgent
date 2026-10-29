# QuantAgent-fnpr.1.1: snapshot macro macro-m5-2026-10

Snapshot macro de ALFRED para M5 (lote L22, `lote/m5-datos`). Incluye las 10 series de `macro-2026-10` más 7 series nuevas (`ICSA`, `T10YIE`, `T5YIE`, `DFII10`, `VIXCLS`, `DTWEXBGS` y el sustituto de ISM New Orders `NEWORDER` para D4).

Bajado el 2026-10-27 con `data macro create --name macro-m5-2026-10`: 190.432 filas en total, `data macro verify macro-m5-2026-10` pasa (OK 17 series). Los datos viven en `local-data/macro-m5-2026-10/`; el manifiesto versionado está en `tests/fixtures/snapshots/macro-m5-2026-10.json`.

| Serie | Frecuencia | Primera fecha | Filas (versiones) | Observaciones con más de una versión | Máx. versiones por observación | Demora típica de publicación (días, mediana, obs. desde 2010) |
|---|---|---|---|---|---|---|
| BAA10Y | diaria | 1986-01-02 | 10662 | 14/10648 | 2 | 1 |
| CPIAUCSL | mensual | 1947-01-01 | 3363 | 658/957 | 7 | 43 |
| DFII10 | diaria | 2003-01-02 | 6214 | 1/6213 | 2 | 1 |
| DGS10 | diaria | 1962-01-02 | 16917 | 6/16910 | 3 | 1 |
| DGS2 | diaria | 1976-06-01 | 13158 | 7/13150 | 3 | 1 |
| DGS3MO | diaria | 1981-09-01 | 11786 | 5/11780 | 3 | 1 |
| DTWEXBGS | diaria | 2006-01-02 | 22107 | 5159/5430 | 10 | 250 |
| FEDFUNDS | mensual | 1954-07-01 | 921 | 19/867 | 4 | 31 |
| ICSA | semanal | 1967-01-07 | 7803 | 1114/3120 | 10 | 5 |
| INDPRO | mensual | 1919-01-01 | 39368 | 1292/1293 | 44 | 45 |
| NEWORDER | mensual | 1968-02-01 | 7029 | 703/704 | 22 | 55 |
| PAYEMS | mensual | 1939-01-01 | 13691 | 1052/1053 | 24 | 34 |
| T10Y2Y | diaria | 1976-06-01 | 13155 | 3/13151 | 3 | 1 |
| T10YIE | diaria | 2003-01-02 | 6216 | 2/6214 | 2 | 1 |
| T5YIE | diaria | 2003-01-02 | 6216 | 2/6214 | 2 | 1 |
| UNRATE | mensual | 1948-01-01 | 2199 | 703/945 | 6 | 34 |
| VIXCLS | diaria | 1990-01-02 | 9627 | 14/9605 | 3 | 0 |

## Sustituto de ISM New Orders (D4) y notas en manifiesto

El manifiesto incluye la clave `notas` documentando la elección de `NEWORDER` como sustituto de ISM New Orders:
- Serie: `NEWORDER` (Manufacturers' New Orders: Nondefense Capital Goods Excluding Aircraft, Census), transformación propuesta `yoy`.
- Razón: es la única serie de nuevos pedidos con historial de vintages en ALFRED desde 1997. Las encuestas regionales de la Fed (Philly Fed `NOCDFSA066MSFRBPHI` y Empire State `NOCDISA066MSFRBNY`) sólo registran vintages a partir de 2014-2015, por lo que no cubren el período in-sample 2007-2018 punto en el tiempo.

## Lectura de las nuevas series

- `NEWORDER` e `ICSA` tienen revisiones sistemáticas: prácticamente todas las observaciones de `NEWORDER` tienen múltiples versiones (hasta 22 versiones por mes), y en `ICSA` más de un tercio de las semanas se corrigen en publicaciones posteriores. El punto en el tiempo (`valor_conocido`) es indispensable para no incurrir en lookahead bias.
- Las series de tasas reales y expectativas (`DFII10`, `T10YIE`, `T5YIE`) y el `VIXCLS` son series financieras de mercado con publicación casi inmediata (demora 0-1 día) y revisiones casi nulas (apenas 1 a 14 observaciones con corrección aislada en toda su historia).
- `DTWEXBGS` presenta múltiples revisiones por recálculo de ponderaciones de comercio exterior, con una demora de publicación inicial corta pero revisiones periódicas retrospectivas.
