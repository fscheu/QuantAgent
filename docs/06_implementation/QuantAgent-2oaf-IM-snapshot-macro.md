# QuantAgent-2oaf: snapshot macro macro-2026-10

Primer snapshot macro de ALFRED para L14 (carril B, lote LB1). Bajado el 2026-10-17 con `data macro create --name macro-2026-10` sobre las 10 series del ticket: 125185 filas en total, `data macro verify macro-2026-10` pasa (OK 10 series). Los datos viven en `local-data/macro-2026-10/`; el manifiesto versionado está en `tests/fixtures/snapshots/macro-2026-10.json`. Solo agregados, sin valores.

| Serie | Frecuencia | Primera fecha | Filas (versiones) | Observaciones con más de una versión | Máx. versiones por observación | Demora típica de publicación (días, mediana, obs. desde 2010) |
|---|---|---|---|---|---|---|
| BAA10Y | diaria | 1986-01-02 | 10655 | 14/10641 | 2 | 1 |
| CPIAUCSL | mensual | 1947-01-01 | 3363 | 658/957 | 7 | 43 |
| DGS10 | diaria | 1962-01-02 | 16910 | 6/16903 | 3 | 1 |
| DGS2 | diaria | 1976-06-01 | 13151 | 7/13143 | 3 | 1 |
| DGS3MO | diaria | 1981-09-01 | 11779 | 5/11773 | 3 | 1 |
| FEDFUNDS | mensual | 1954-07-01 | 921 | 19/867 | 4 | 31 |
| INDPRO | mensual | 1919-01-01 | 39368 | 1292/1293 | 44 | 45 |
| PAYEMS | mensual | 1939-01-01 | 13691 | 1052/1053 | 24 | 34 |
| T10Y2Y | diaria | 1976-06-01 | 13148 | 3/13144 | 3 | 1 |
| UNRATE | mensual | 1948-01-01 | 2199 | 703/945 | 6 | 34 |

## Lectura

- CPIAUCSL, UNRATE, PAYEMS e INDPRO se revisan: casi todas sus observaciones tienen más de una versión (INDPRO hasta 44). Para ellas el punto en el tiempo (`valor_conocido`) importa.
- Las series de mercado (DGS3MO, DGS2, DGS10, T10Y2Y, BAA10Y) casi no se revisan: la mayoría tiene una sola versión por observación. DGS10 no cumple la letra del criterio del ticket ("una versión"): tiene 6 observaciones con 2 o 3 versiones de 16903. Son correcciones aisladas de FRED, sin efecto práctico.
- La demora típica de las series de mercado es 1 día; la de actividad e inflación va de 31 a 45 días (CPIAUCSL 43, INDPRO 45, PAYEMS y UNRATE 34, FEDFUNDS 31). La mediana se calcula sobre observaciones desde 2010 porque antes del inicio del registro de vintages ALFRED completa la primera versión con fecha posterior.
- El cliente tuvo que partir el pedido en ventanas de vintage (#266): las series diarias superan el tope de 2000 fechas de vintage de la API.
