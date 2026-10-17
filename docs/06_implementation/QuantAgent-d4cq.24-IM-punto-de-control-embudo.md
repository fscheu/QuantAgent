# QuantAgent-d4cq.24: punto de control del embudo L10

Corridas del PM del 2026-10-16 sobre `lote/embudo @13a7285a`, snapshot
`etf-1d-2026-10`, registros reales en `/home/azureuser/repos/projects/QuantAgent/local-data`.
Este documento reemplaza el bloqueo de escritura según el comentario del ticket de las 23:18.
Se leyeron los artefactos existentes; no se volvió a ejecutar `hipotesis correr`.

## Resultado

`rsi-reversion-corta`: PASA etapas 0, 1 y 2; CORTA en etapa 3.
En etapa 2 pasan V30 XLE y XLU y el control coincide en 12 de 12 activos.
En la prueba, ningún activo cumple simultáneamente V30 PASA y DSR ≥ 0,95.
Todos fallan DSR; los ocho con V30 NO PASA también tienen PnL ≤ 0 y Sharpe tasa 0 < 0,10.
Última etapa, transcripción de `/tmp/qa-embudo/rsi-reversion-corta.log`:

| Activo | V30 | DSR |
|---|---|---:|
| SPY | PASA | 0.00124286 |
| IWM | NO PASA | 0.0000360211 |
| EFA | NO PASA | 0.000126497 |
| EEM | NO PASA | 0.0000263758 |
| TLT | NO PASA | 0.0000249926 |
| SHY | PASA | 0.00471553 |
| GLD | NO PASA | 0.00014277 |
| DBC | NO PASA | 0.000000222634 |
| XLK | PASA | 0.00703334 |
| XLE | PASA | 0.00182111 |
| XLF | NO PASA | 0.000159353 |
| XLU | NO PASA | 0.000103329 |

`sma-tendencia`: PASA etapa 0; CORTA en etapa 1: 12 de 12 bajo 0,10
o con menos de 30 trades. Todos tienen menos de 30. No abrió la prueba.
Última etapa, filas de `intentos.csv` (Sharpe redondeado):

| Activo | Trades | Sharpe |
|---|---:|---:|
| SPY | 3 | 0.371159 |
| IWM | 4 | 0.597319 |
| EFA | 7 | -0.235299 |
| EEM | 7 | -0.204756 |
| TLT | 10 | 0.454131 |
| SHY | 7 | 0.491952 |
| GLD | 10 | -0.402143 |
| DBC | 6 | -0.124985 |
| XLK | 4 | 0.683665 |
| XLE | 4 | 0.007093 |
| XLF | 5 | 0.479563 |
| XLU | 7 | 0.480644 |

## Tiempos y registros

`/tmp/qa-embudo/tiempos.txt`: RSI 614 s, SMA 153 s; ambos exit 1 (CORTA).
Diferencias entre `fecha_utc` consecutivas de `hipotesis.csv`, en segundos:

| Hipótesis | Etapa 1 | Etapa 2 | Etapa 3 |
|---|---:|---:|---:|
| rsi-reversion-corta | 127.765087 | 342.506098 | 113.579736 |
| sma-tendencia | 124.372562 | no alcanzada | no alcanzada |

Etapa 0 sin duración individual registrada; el total incluye arranque y cierre
y no equivale a la suma de esos intervalos.
`antes.txt` registra 15 líneas de intentos (14 datos); ahora son 63 (62 datos).
Nuevos: RSI 36 intentos (12 por etapa 1/2/3), 4 estados y 1 candado;
SMA 12 intentos, 2 estados y 0 candados. Total: +48, +6, +1.
`hipotesis.csv` y `hipotesis-candado.csv` no existían antes según el PM;
ahora tienen 6 datos más encabezado y 1 fila sin encabezado, respectivamente.
El único candado es RSI/prueba, abierto a las 23:13:24.601827 UTC.

## Corte temporal y reproducción de la lectura

Las doce tablas RSI de `etapa3/*.csv` declaran 2019-01-02 a 2022-12-30.
Se revisaron también las fechas de entradas y salidas en los CSV de trades:
ninguna alcanza 2023. Las salidas de etapa 2 terminan en 2018 y los intentos
nuevos de SMA declaran 2007–2018. Esto concuerda con los límites del motor
en `quantagent/fabrica/validacion.py` (etapa 2: 2007–2018; etapa 3: 2019–2022).
No se inspeccionaron precios ni se reproducen filas de trades en este informe.

Lectura reproducible en la VM, sin abrir otra vez la prueba:

```bash
python -c 'import csv,pathlib; p=pathlib.Path("/home/azureuser/repos/projects/QuantAgent/local-data"); r=list(csv.DictReader((p/"intentos.csv").open()))[14:]; assert len(r)==48 and all(x["hasta"]<"2023" for x in r); assert len(list(csv.DictReader((p/"hipotesis.csv").open())))==6; assert len((p/"hipotesis-candado.csv").read_text().splitlines())==1; fs=list(pathlib.Path("/tmp/qa-embudo/rsi-reversion-corta").rglob("*.csv")); ds=[v for f in fs for row in csv.DictReader(f.open()) for k,v in row.items() if k in ("desde","hasta","entry_time","exit_time") and v]; assert ds and max(ds)<"2023"; print("48 intentos nuevos; 6 estados; 1 candado; fechas < 2023")'
```
