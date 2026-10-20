# Cambio de mes: corte en etapa 1

Corrida real del 2026-10-20, ticket QuantAgent-dtbv (L18). Sin cambios al YAML ni al motor.

## Ejecución

`NUMBA_CACHE_DIR=/tmp/numba-dtbv MPLCONFIGDIR=/tmp/matplotlib-dtbv timeout 30m python -m quantagent.cli backtest hipotesis correr hipotesis/cambio-de-mes.yaml --snapshot etf-1d-2026-10-amplio --carpeta /home/azureuser/repos/projects/QuantAgent/local-data/cambio-de-mes-dtbv --hasta-etapa 2`

Salida real (exit 1 por corte del embudo, no por error):

```text
Etapa 0: PASA (admisión aceptada)
Etapa 1: CORTA (Sharpe de cartera bajo 0,10)
Llegó a la etapa 1: CORTA (Sharpe de cartera bajo 0,10)
```

| Etapa | Veredicto | Sharpe tasa 0 | Trades | Benchmark | DSR |
|---|---|---:|---:|---:|---|
| 0 | PASA: admisión aceptada | No aplica | No aplica | No aplica | No aplica |
| 1 | CORTA: Sharpe < 0,10 | -0,464643114668135 | 736 | 0,3152005130375183 | No calculado por el filtro |
| 2 | No ejecutada: corte temprano | No disponible | No disponible | No disponible | No disponible |

Benchmark descriptivo: mismos ocho ETFs, pesos iguales y rebalanceo mensual, tasa 0,
calculado con `sharpe_benchmark` sobre los cierres del snapshot recortados a 2007–2018.
No interviene en el corte de etapa 1. No se calculó DSR ni se ejecutó el motor después del corte.

## Registro y protección del tramo de prueba

`QUANTAGENT_SNAPSHOT_DIR=/home/azureuser/repos/projects/QuantAgent/local-data`.
`intentos.csv` pasó de 111 a 112 filas de datos. Se agregó una sola fila:

```csv
fecha_utc,estrategia,activo,snapshot,desde,hasta,costos,tamano,params,trades,pnl,sharpe,hipotesis,etapa,sharpe_base
2026-10-20T23:42:40.914206+00:00,turn-of-the-month,cartera,etf-1d-2026-10-amplio,2007-01-01,2018-12-31,etf,1.0,{},736,,-0.464643114668135,cambio-de-mes,1,tasa_0
```

Antes no había intentos de cambio-de-mes. `hipotesis.csv` registró etapa 0 PASA
a las 23:42:06.509860 UTC y etapa 1 CORTA a las 23:42:40.914574 UTC.
Ningún intento de esta hipótesis tiene `hasta >= 2019-01-01`.
`hipotesis-candado.csv` conserva sus cinco entradas previas, ninguna de cambio-de-mes.
SHA-256 antes y después: `8cc215d8a57c73991d2c36513b0743e09752d4383b0f8304f21f99ae8910f7a5`.
No hubo etapa 3 ni invocación con `--abrir-prueba`.

Recomendación: cerrar esta evaluación en el corte de etapa 1. No pedir el OK para
2019–2022; la condición de pase de etapa 2 no se alcanzó. No ajustar la ventana para rescatar el resultado.

## Verificación

El primer arranque falló al importar vectorbt por caché Numba fuera del área escribible;
no registró intentos ni estados. La ejecución indicada arriba usó caché en `/tmp`.
Los registros reales permanecen en la VM, ignorados por Git; la fila se transcribe arriba para revisión.
Sin módulos ni tests modificados: no hay tests dirigidos por importación aplicables.
`NUMBA_CACHE_DIR=/tmp/numba-dtbv MPLCONFIGDIR=/tmp/matplotlib-dtbv timeout 5m scripts/loop/smoke.sh`
terminó con exit 0: 9 trades, PnL 561,18.
El comando de Probar del PR lee registros sin agregar intentos; repetir la corrida sí los agregaría.
