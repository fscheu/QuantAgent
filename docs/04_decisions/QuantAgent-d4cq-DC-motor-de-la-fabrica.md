# QuantAgent-d4cq: motor de la fábrica de estrategias

- **Tipo:** decisión de arquitectura previa al plan M3 (fábrica de estrategias)
- **Estado:** propuesta del PM para decisión de Fede
- **Fecha:** 2026-10-14
- **Entrada:** spike de LEAN (`QuantAgent-ryaw`, rama `spike/lean`, `docs/04_decisions/QuantAgent-ryaw-DC-spike-lean.md`), punto de control de L7 (PR #132) y una medición de tiempos en la VM el mismo día.

## La pregunta

La fábrica tiene que evaluar cientos de hipótesis. Cada una se corre sobre 12 activos y, si tiene parámetros, con walk-forward. Hay que decidir qué motor corre cada etapa del embudo:

- el filtro rápido sobre 2007–2018;
- el control;
- la prueba 2019–2022;
- la reserva.

El motor de paper trading es otra decisión y queda para más adelante.

## Lo que medimos

| Dato | Valor | Fuente |
|---|---:|---|
| Motor propio, rsi/SPY 2007–2018 (3.020 velas), una celda | 14 s | VM, 2026-10-14, `/usr/bin/time` |
| Motor propio, 4 estrategias sobre SPY 2007–2018 | 364 s (unos 115 s por celda en las otras 3) | ídem |
| LEAN, rsi/SPY 2007–2018 | 6 s de algoritmo, 24 s de pared con arranque | spike ryaw |
| Punto de control L7, 48 celdas × 4 años, `compare` + control + veredicto | unos 8 min | PR #132 |
| VM | 2 núcleos, 15 GB | `nproc`, `free` |

**Cuenta de capacidad.** 100 hipótesis × 12 activos son 1.200 celdas. A 60 s por celda, en serie, son 20 horas de VM, y eso sin walk-forward. Con walk-forward de 8 ventanas y una grilla de 20 combinaciones, cada celda pasa a ser 160 corridas. Con el motor propio en la VM, el filtro rápido no escala: hay que hacerlo más rápido o correrlo en otra parte.

Hay un dato que conviene revisar antes de decidir. Alguna de las tres estrategias que no son rsi tarda unas 8 veces más que rsi con las mismas velas. Probablemente recalcula sus ventanas en cada vela, y eso se puede corregir dentro del motor propio.

## Opciones

**A. Motor propio en todas las etapas.** Se acelera perfilando la estrategia lenta y corriendo en paralelo, en la VM y en la nube.
- A favor: no agrega dependencias. El código está auditado: oráculos de M1, control contra `backtesting.py` y recálculo independiente.
- En contra: Python con una vela por vez. Aun acelerado, cientos de hipótesis con walk-forward cuestan horas de cómputo por noche.

**B. Filtro vectorizado con `vectorbt` y motor propio para validar (híbrido).** La etapa 1 corre en `vectorbt`, que es Python, open source y muy usado en la comunidad, y procesa miles de combinaciones en segundos con numpy. Solo las hipótesis que pasan el filtro se corren en el motor propio, con control, prueba y reserva.
- A favor: la etapa cara se hace para pocas hipótesis, y el validador que construimos en M2 se mantiene igual. Cumple tu preferencia de usar una biblioteca establecida en lugar de construir.
- En contra: `vectorbt` simplifica la ejecución: llenado, stops dentro de la vela y tamaño. Sus números no van a coincidir con los nuestros, y no tienen que coincidir. Lo que hay que exigirle es que **ordene igual**: que las hipótesis que filtra como malas también sean malas en el motor propio. Eso se mide antes de confiar en el filtro.
- Riesgo de licencia: la versión libre de `vectorbt` tiene licencia Apache 2.0 con Commons Clause. Sirve para uso propio, no para revender. La versión PRO es paga y no hace falta.

**C. Pivot a LEAN.** LEAN corre todas las etapas y nuestra capa de validación (candado, intentos, veredicto, Sharpe deflactado) se monta encima.
- A favor: optimizador, análisis automáticos y camino directo a paper y live.
- En contra: 756 mil líneas de C#, puente con Python frágil (el proceso aborta al cerrar), diagnóstico en un lenguaje que no es el nuestro y una brecha sin cerrar contra nuestros controles (rsi 140 de 187). Además, 24 s por corrida con el arranque no es más rápido que el motor propio para el filtro.

**D. LEAN como segundo control.** Es tu propuesta del 2026-10-13: un control más en la etapa 3, para las finalistas. No resuelve la capacidad, pero se puede combinar con A o con B.

## Recomendación del PM

**B**, con dos condiciones medibles antes de construir la fábrica encima:

1. `vectorbt` corre las 4 estrategias de M2 sobre los 12 ETFs de 2007–2018 en menos de 1 minuto en total en la VM.
2. El orden por Sharpe de esas 48 celdas en `vectorbt` y en el motor propio tiene una correlación de Spearman de 0,8 o más.

Si alguna de las dos falla, se pasa a A: se perfila la estrategia lenta y se corre en paralelo en la nube.

La decisión de LEAN como motor de paper (C) queda para cuando haya una finalista que justifique ir a paper. Ahí es donde LEAN aporta, y para entonces habrá que haber verificado fracciones de acción y el plugin de Alpaca. Mientras tanto, D se puede sumar como control de las finalistas si el costo de mantener el adaptador lo justifica.

## Qué decide Fede

- **Motor del filtro (etapa 1):** A | B (recomendado) | C.
- **LEAN:** se posterga hasta tener una finalista (recomendado) | se decide ahora.
