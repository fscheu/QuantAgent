# QuantAgent-e4jc.6 — F2 hasta etapa 2

La propuesta `tendencia-larga-corta` sigue en `catalogo/propuestas/` sin aprobar.
La decisión 8 de `docs/04_decisions/QuantAgent-R01-DC-decisiones-investigacion-edge.md`
condiciona su aprobación a que exista la clase `tsmom-largo-corto` (ticket
QuantAgent-t2qp). El intento registrado abajo no llegó al backtest.

## Ejecución

Snapshot: `etf-1d-2026-10-amplio`, cargado desde
`/home/azureuser/repos/projects/QuantAgent/local-data`. Se usó
`NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/mpl-cache`; no se deshabilitó
JIT ni se abrieron ventanas 2019–2022 o la reserva.

La corrida se intentó con `--carpeta
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

## Reintento tras registrar F2

Con el OK de Fede del 2026-10-26, la propuesta se aprobó y quedó en
`hipotesis/tendencia-larga-corta.yaml`. Se usaron los snapshots reales de
`/home/azureuser/repos/projects/QuantAgent/local-data`; no se abrió 2019–2022 ni
la reserva. Comando:

```bash
NUMBA_CACHE_DIR=/tmp/numba-cache MPLCONFIGDIR=/tmp/mpl-cache QUANTAGENT_SNAPSHOT_DIR=/home/azureuser/repos/projects/QuantAgent/local-data python -m quantagent.cli backtest hipotesis correr hipotesis/tendencia-larga-corta.yaml --snapshot etf-1d-2026-10-amplio --carpeta local-data/intentos/f2 --hasta-etapa 2
```

Resultado: etapa 0 **PASA** (admisión aceptada); etapa 1 **PASA** (Sharpe de
cartera 0,54); etapa 2 **CORTA**. La etapa 2 reporta operaciones menores a dos,
P&L no positivo y drawdown mayor a 0,35 en varias celdas, además de errores del
adaptador al validar órdenes cortas. El corte de etapa 2 no habilita la reserva.

Con las curvas guardadas y la misma lógica de pesos por volatilidad del motor,
la correlación diaria de F2 con el benchmark 1/N en 2007–2018 es **0,8562**
(29 ETFs, 3.018 retornos); Sharpe de la curva combinada 0,6958 frente a 0,4368
de 1/N. La curva de F2 termina en USD 149.988,90 desde USD 100.000. La
correlación alta confirma que la cartera no diversifica 1/N en este tramo.

La ejecución generó dos registros de etapa 1 y dos tandas de celdas de etapa 2
en `intentos.csv` (N incluye ambos intentos). No se agregó ninguna fila a
`hipotesis-candado.csv` (SHA-256 sin cambios:
`8cc215d8a57c73991d2c36513b0743e09752d4383b0f8304f21f99ae8910f7a5`). F2 no
superó etapa 2; no se pide el OK de Fede para la reserva.

## Nota del PM

Entrega recuperada por el PM del log de la corrida de Codex (la rama `loop/QuantAgent-e4jc.6` ya existía por #515 y el wrapper no pudo publicar). Las cifras están copiadas del IM que escribió Codex; no las recalculé. Salvedad: la etapa 2 reporta "errores del adaptador al validar órdenes cortas"; antes de leer el CORTA como falta de edge hay que revisar si esas celdas fallaron por el adaptador (ver ticket de seguimiento).
