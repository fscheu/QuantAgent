# QuantAgent-d4cq.34: punto de control de tanda L11

Corrida del PM del 2026-10-17 sobre `lote/informe-y-tanda @836bcd36`, snapshot
`etf-1d-2026-10`, registros reales en `/home/azureuser/repos/projects/QuantAgent/local-data`.
Se leyeron los artefactos generados en `/tmp/qa-tanda` y los registros; no se volvió a ejecutar la tanda.

## Resultado de la tanda

Cuatro hipótesis evaluadas en `hipotesis/`:
- `maximo-52-semanas`: PASA etapa 0 (admisión aceptada); CORTA en etapa 1 (12 de 12 bajo 0,10 o con menos de 30 trades).
- `momentum-12m`: PASA etapa 0 (admisión aceptada); CORTA en etapa 1 (12 de 12 bajo 0,10 o con menos de 30 trades).
- `rsi-reversion-corta`: salteada (ya evaluada: etapa 3 CORTA).
- `sma-tendencia`: salteada (ya evaluada: etapa 1 CORTA).

Cada hipótesis cuenta con su subdirectorio en `/tmp/qa-tanda/` con `corrida.log` e `informe.html`.
La tanda generó `/tmp/qa-tanda/resumen.md` informando 2 hipótesis con CORTA en etapa 1 y 24 intentos totales.

## Observación sobre el filtro (etapa 1)

`maximo-52-semanas` y `momentum-12m` cortan ambas por la regla de mínimo de 30 trades (E01), a pesar de que la mayoría de los activos presenta Sharpe fuera de muestra ≥ 0,10:
- En `maximo-52-semanas`, 8 de 12 activos tienen Sharpe ≥ 0,10 (hasta 0,45 en SHY y 0,44 en XLF), pero registran entre 4 y 14 trades en 12 años.
- En `momentum-12m`, 6 de 12 activos tienen Sharpe ≥ 0,10 (hasta 0,76 en XLK y 0,60 en SPY/XLF), pero registran entre 5 y 10 trades en 12 años.

Al tener menos de 30 trades por activo fuera de muestra, todos cuentan como bajo 0,10 y resultan en 12 de 12 bajo el umbral (superando el límite de 2/3).

## Registros en local-data y tiempos

- `intentos.csv`: 24 filas nuevas (12 de `maximo-52-semanas` y 12 de `momentum-12m`, ambas en etapa 1). Pasa de 62 a 86 filas de datos (87 líneas con encabezado). `rsi` y `sma` sumaron 0 intentos nuevos.
- `hipotesis.csv`: 4 filas nuevas (etapas 0 PASA y 1 CORTA para las dos hipótesis nuevas). Pasa de 6 a 10 filas de datos (11 líneas con encabezado). `rsi` y `sma` no sumaron estados nuevos.
- `hipotesis-candado.csv`: sin cambios (1 fila previa correspondiente a la prueba de `rsi-reversion-corta`). Ninguna hipótesis avanzó a etapa 3.
- Rango de fechas: todos los intentos nuevos declaran `desde: 2007-01-01` y `hasta: 2018-12-31`. Ninguna corrida leyó velas de 2023 en adelante.
- Tiempo total de la tanda: ~706 s (~11m 46s). `maximo-52-semanas` demandó ~576 s, `momentum-12m` ~109 s, y `rsi`/`sma` se saltearon de inmediato.

## Verificación en la VM

```bash
python3 -c 'import csv, pathlib; p=pathlib.Path("/home/azureuser/repos/projects/QuantAgent/local-data"); i=list(csv.DictReader((p/"intentos.csv").open()))[62:]; h=list(csv.DictReader((p/"hipotesis.csv").open()))[6:]; c=(p/"hipotesis-candado.csv").read_text().splitlines(); assert len(i)==24 and all(x["hasta"]<"2023" for x in i); assert {x["hipotesis"] for x in i}=={"maximo-52-semanas","momentum-12m"}; assert len(h)==4; assert len(c)==1; t=pathlib.Path("/tmp/qa-tanda"); assert (t/"resumen.md").exists(); assert all((t/x/"informe.html").exists() and (t/x/"corrida.log").exists() for x in ["rsi-reversion-corta","sma-tendencia","maximo-52-semanas","momentum-12m"]); print("Tanda OK: 24 intentos, 4 estados, 0 candados nuevos, 4 informes, 1 resumen, fechas < 2023")'
```
