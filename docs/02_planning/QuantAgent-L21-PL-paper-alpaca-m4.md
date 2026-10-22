# QuantAgent-L21: plan del lote de paper trading con Alpaca (M4)

**Estado:** propuesta, pendiente del OK de Fede (2026-10-22). No se lanza ningún ticket de código antes.
**Rama:** `lote/paper-alpaca`, desde `origin/main` (no depende de S07 ni de S09 de L19).
**Marco (Fede, DC R01):** paper es prueba del sistema, no evaluación de estrategias. Paper liviano: pesos iguales a la sombra salvo redondeo, ninguna orden rechazada sin explicar, dividendos bien tratados; si algo no coincide, se frena y se investiga. Ningún resultado de paper promueve una estrategia a dinero real.

## Qué hay hoy (verificado en origin/main, 2026-10-22)

- `quantagent/trading/paper_broker.py` ya tiene la clase abstracta `Broker` (`place_order`, `cancel_order`, `get_balance`, `get_positions`) y el `PaperBroker` simulado. **No existe `AlpacaBroker`**: `quantagent/data/sources/alpaca.py` es solo de datos.
- Cerrados y en main: `say`/`ev7d` (la salida solo cierra), `oc0h` (reversión: plana al 100 %), `dfa4` (salida intravela compartida), `7nle` (diseño de velas para operar), `kw84` (gate por lote).
- Claves de Alpaca paper en `~/secrets/alpaca.env`.
- **Hueco sin ticket:** nada traduce una cartera (pesos objetivo) a órdenes de rebalanceo en paper. El scheduler opera estrategia por símbolo. Las dos carteras de M4 (cartera-transversal y núcleo+curva) necesitan esa pieza.
- Nota: núcleo+curva dio CORTA el criterio de paper (IM de `ro3w.1`: Sharpe 0,38 contra 0,61 de 1/N en 2019–2022). Entra a M4 igual, por el marco de arriba: es prueba de sistema.

## Decisiones que necesito de Fede

1. **¿Entra el sizing por riesgo (`0gh7`) al lote?** Las carteras usan pesos, no tamaño por stop; el sizing por riesgo sirve para las estrategias sueltas con stop. Recomiendo **sí, solo `0gh7.1`–`.4`** (el stop por defecto de 2 % y el tamaño llegan a `execute_decision`), así queda probado en paper también el camino de una estrategia suelta. `0gh7.5` (crossval) queda en L20. Si preferís paper solo con carteras, se saca el bloque B.
2. **Regla de stop simple para las carteras:** propongo que las carteras no tengan stop propio en M4 (se rebalancean por pesos) y que el stop de ±2 % rija solo para estrategias sueltas. Confirmar.

## Orden de tickets

Cada bloque es una sola cola; un ticket por vez. Los tickets nuevos los crea el PM en bd tras el OK, con padre `QuantAgent-L21` (a crear).

**A. Salida y riesgo sin broker (todo sobre main)**
1. `QuantAgent-beo`: cerrar lo que falte de la unificación de salida (dfa4 cubrió 4 de 5 entregas; verificar si queda una). Bloquea a `0gh7`.
2. `QuantAgent-brwj` (X01): guard "vela posterior a la entrada".

**B. Sizing por riesgo (si Fede dice sí)**
3. `0gh7.1` stop por defecto → `0gh7.2` regla `riesgo` → `0gh7.3` stop hasta `execute_decision` → `0gh7.4` configuración y flags.

**C. Broker (puede ir en paralelo con B)**
4. `QuantAgent-13a` (reducido): la ABC ya existe; falta cerrar el contrato con lo que Alpaca necesita (fills parciales, rechazos, nocional), apoyado en la medición de `8659`.
5. `QuantAgent-8659`: medir en Alpaca paper qué pasa con una orden que excede el poder de compra y fijar la regla (una sola vez en `quantagent/trading/`). Usa las claves reales.
6. `QuantAgent-qr6`: selector de broker y bloqueo del endpoint live.
7. `QuantAgent-yme`: `AlpacaBroker` paper (alpaca-py). Depende de 4 y 6.
8. `QuantAgent-y62`: reconciliación de posiciones DB local contra Alpaca. Depende de 7.

**D. Carteras en paper (tickets nuevos, a crear)**
9. **D1** Pesos objetivo por fecha para cartera-transversal y núcleo+curva (reusa `quantagent/fabrica/cartera.py`; comparte pieza con `e4jc.2`/X08 de L20: se coordina cuál va primero).
10. **D2** Rebalanceador paper: de pesos objetivo y posiciones actuales a órdenes, con redondeo de acciones y regla de 8659.
11. **D3** "Sombra": curva/pesos del backtest para cada fecha, guardados para comparar. Depende de D1.

**E. Chequeos de paper liviano y puesta en marcha**
12. `QuantAgent-e4jc.13` (X19, P-1): pesos contra la sombra, órdenes rechazadas, dividendos. Depende de brwj, D2 y D3.
13. `QuantAgent-b41`: digest diario en texto plano (incluye el resultado de P-1).
14. `QuantAgent-55x` (P3, opcional): reloj por velas en corridas online; solo si D2 lo necesita.
15. Puesta en marcha en la VM: dos carteras corriendo en paper con Alpaca durante 2 semanas sin intervención. Esto no es un ticket del loop: lo arranca Fede o el PM con el servicio del scheduler y requiere OK de Fede (es un cambio de servicio en la VM).

## Lo que no entra

Dinero real, cuenta margin y cortos (X11 queda en L20), escalera de incubación, DBC/USO, retención del 30 % de dividendos (todo se decide antes de dinero real, DC R01).

## Gate y paralelismo con la notebook

- Distinto lote que L19 (`lote/medicion-y-generador`, PR #360): con `LOOP_LOTE` el gate evalúa solo la entrega previa del lote activo (K01, en main), así que el PR loop pendiente de L19 no frena este lote. Si algún día la notebook abre un PR loop en `lote/paper-alpaca`, la VM no lanza hasta mergearlo.
- Archivos distintos (L19: informe y catálogo de hipótesis; L21: `quantagent/trading/`). El único roce probable es `PLAN-CONTINUACION.md` (§4 y la tabla de lotes): lo resuelve el PM al mergear.
- Agente: Codex con gpt-6-astra, un ticket por vez con `hermes_launch.sh --run --lote lote/paper-alpaca`.

## Cierre

PR del lote a main marcado listo y pedido de `R: lote-paper-alpaca vi: ... decido: merge`. La puesta en marcha (ítem 15) va después del merge.
