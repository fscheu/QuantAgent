# QuantAgent-e4jc.6 — F2 hasta etapa 2

La aprobación de F2 quedó registrada el 2026-10-26 por Fede. La decisión 8 de
`docs/04_decisions/QuantAgent-R01-DC-decisiones-investigacion-edge.md` la habilita
con C0, sin diversificador, y cuenta como variante de tendencia.

## Ejecución

Snapshot: `etf-1d-2026-10-amplio`, cargado desde
`/home/azureuser/repos/projects/QuantAgent/local-data`. Se usó
`NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/mpl-cache`; no se deshabilitó
JIT ni se abrieron ventanas 2019–2022 o la reserva.

La aprobación movió el YAML a `hipotesis/tendencia-larga-corta.yaml` y agregó
`autor: fede`. La corrida se intentó con `--carpeta
/home/azureuser/repos/projects/QuantAgent/local-data/intentos/f2 --hasta-etapa 2`
y terminó antes del backtest:

```
Error: Estrategia desconocida 'tsmom-largo-corto'. Opciones: curva-pendiente, fifty-two-week-high, halloween, momentum-12m, rebalanceo-fin-de-mes, rsi, rsi-regimen, senal-externa, sma-cross, triple-screen, turn-of-the-month
```

## Resultado

Etapas 0, 1 y 2: **no evaluadas**. El motor de esta rama no registra
`tsmom-largo-corto`; no se generó curva F2. Por eso no hay correlación de F2 con
1/N ni veredicto que habilite pedir la reserva a Fede.

`intentos.csv` conserva N=112 (SHA-256
`b95c7a7375af267527dce3706fd4282f345910a156ec87e72f077615a234bae9`); no se
fabricó una fila para una ejecución que no llegó al backtest. El candado también
quedó intacto (SHA-256
`8cc215d8a57c73991d2c36513b0743e09752d4383b0f8304f21f99ae8910f7a5`).

Para continuar, registrar `tsmom-largo-corto` en las estrategias admitidas por
el CLI antes de repetir la corrida.
