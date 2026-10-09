# M2 — Validación con datos reales (plan posterior a M1)

Ticket: `QuantAgent-lp1` (T25 de `PLAN-CONTINUACION.md`). Estado: ~~**borrador**, escrito el 2026-10-06.~~ **aprobada por Fede
el 2026-10-09 (PR #62), con la decisión previa 5 cambiada: ver §13.** §1–§7 son el borrador del 2026-10-06; §8–§14, el cierre.
~~Se cierra después de T23 (ADR del engine): si el engine cambia, las filas de los lotes A a C se reescriben.~~ T23 eligió el engine propio (opción B).
Los tickets de BEADS se crean cuando Fede apruebe este documento; hasta entonces las filas usan ~~`V01`…`V41`~~ `V01`…`V55`
de §11. Texto completo de los lotes 1 y 2: [`QuantAgent-lp1-tickets-m2.md`](./QuantAgent-lp1-tickets-m2.md).

**Respuesta corta (2026-10-09).** M1 cerró con el engine propio igual a backtesting.py trade por trade, pero solo con
RSI, sobre datos sintéticos sin huecos y sin comisiones. M2 tiene 7 lotes y 55 filas. Los dos primeros dejan las 3
estrategias corriendo sobre SPY diario real, con costos y contra comprar y mantener. Antes de arrancar, Fede tiene que
tomar 7 decisiones (§13). El riesgo principal: el engine avanza por una grilla de calendario y no por las velas del
dato, y eso nunca se probó (§12, riesgo 1).

---

## 1. Por qué esta etapa

M1 deja un backtester que calcula bien, probado sobre datos sintéticos (`tests/fixtures/spy-90d.csv` tiene
velas 24/7 y volumen fijo) ~~y sin ninguna estrategia que gane plata (RSI sobre ese fixture: profit factor 0.61)~~.
Corrección 2026-10-09: el 0.61 era con 1% de slippage por lado; desde el PR #60 RSI da 1.35 (`backtest_metrics.md` §3), sobre datos sintéticos.
Conectar un broker en ese estado solo prueba la plomería.

Decisiones de Fede del 2026-10-06:

1. Antes del broker va una etapa de validación: datos históricos reales, costos realistas, comparación de
   N estrategias y prueba fuera de muestra.
2. La estrategia de 4 agentes LLM es una estrategia más. No se le reserva presupuesto ni trabajo propio.
3. La IA generativa entra en otras instancias: informe de corrida, revisor crítico, nota macro semanal y
   generador de hipótesis.
4. Se valida sobre ETFs en velas diarias **y** sobre un par de activos en 4h.
5. Numeración de milestones nueva (§2).

## 2. Milestones

| Milestone | Nombre | Criterio de completitud |
|---|---|---|
| M1 | Backtesting estable | 3 estrategias, resultados reproducibles, métricas auditadas, ADR del engine (T23) |
| M2 | Validación con datos reales | §3 de este documento |
| M3 | Broker Alpaca en paper | Integración funcionando contra la cuenta paper (`13a`, `yme`, `qr6`, `y62`) |
| M4 | Paper trading estable | 2 semanas sin intervención manual (`b41`) |
| M5 | Live con capital mínimo | Primera semana con dinero real, sin errores críticos |

"Arquitectura multi-estrategia", el M2 anterior de `milestone-tracker.md`, queda absorbido por el lote C.

## 3. Criterio de salida de M2

M2 termina cuando existen las dos cosas:

1. **La herramienta:** un comando compara estrategias sobre datos reales con costos, contra comprar y mantener,
   y emite un veredicto dentro y fuera de muestra con su informe.
2. **El veredicto:** al menos una estrategia pasa los umbrales sobre la reserva final, o Fede decide por
   escrito pasar a M3 sin estrategia validada (o cortar el proyecto en esa capa).

Los umbrales se deciden en V21. Propuesta inicial: después de costos, sobre el período de prueba, Sharpe mayor
que el de comprar y mantener y drawdown máximo no mayor. Los umbrales de 2025 (win rate ≥ 40%, Sharpe ≥ 1.0,
drawdown ≤ 15%) no comparaban contra una referencia.

## 4. Reglas de la etapa

- **Modo de trabajo:** el de `PLAN-CONTINUACION.md` §3 (entregas de ~100 líneas, revisión del PM, un PR de lote
  por punto de control).
- **Reserva final:** un tramo de la historia que ninguna corrida de ajuste ni de comparación puede leer. Se abre
  una sola vez, en V41. El CLI lo hace cumplir (V22).
- **Intentos contados:** cada combinación estrategia × parámetros probada queda registrada. Un resultado bueno
  entre cien intentos no vale lo mismo que entre tres.
- **La IA no calcula:** los números salen de código determinista. Los modelos leen artefactos (CSV, JSON) y
  escriben texto; un verificador comprueba que cada número del texto existe en el artefacto.
- **Sin clave también funciona:** `backtest run`, `compare` y el veredicto corren sin API key. El informe y la
  crítica son un paso aparte.
- **La IA no ve la reserva:** ni el generador de hipótesis ni el revisor crítico reciben datos de ese tramo.

## 5. Lotes y cola

> **Reemplazada el 2026-10-09 por §10–§11.** Queda como registro. Los `V` de §3 a §7 son los del borrador, no los de §11.
> Equivalencias, también para los docs que los citan (`iuf`, `6ie`): V01 → decisión previa 1; V02 → V01–V02; V06 → V09;
> V07 → V20; V08 → V13; V09 → V14; V11 → V22; V21 → V30 (y decisión previa 3); V22 → V08; V27 → V43; V28 → V45;
> V32 → decisión previa 5; V34 → V41; V41 → V54–V55.

Tipos de revisión: **leer**, **decidir**, **probar** (igual que en `PLAN-CONTINUACION.md` §4).

| Lote | Rama | Filas | Punto de control | Decisiones de Fede |
|---|---|---|---|---|
| A | `lote/datos-reales-diarios` | V01–V07 | Las 3 estrategias corren sobre un snapshot versionado de ETFs diarios reales, con reporte de calidad | V01 |
| B | `lote/costos-y-referencia` | V08–V13 | Cada corrida informa costos y se compara contra comprar y mantener | V08, V13 |
| C | `lote/comparacion` | V14–V20 | Un comando corre la matriz estrategia × activo; agregar una estrategia no toca el engine | V19 |
| D | `lote/fuera-de-muestra` | V21–V26 | Veredicto PASA / NO PASA por estrategia, dentro y fuera de muestra, con la reserva bajo candado | V21 |
| E | `lote/informe-y-critico` | V27–V31 | Cada comparación trae un informe en lenguaje llano y una crítica con evidencia | V27 |
| F | `lote/intradia-4h` | V32–V35 | La misma comparación y veredicto sobre 2 activos en 4h | V32 |
| G | `lote/generador-de-hipotesis` | V36–V41 | Un modelo propone estrategias, se implementan como código determinista y pasan solas por el veredicto | V36, V41 |

Corte de control después del lote D: con la herramienta completa y los primeros veredictos, Fede decide si E, F
y G siguen en este orden. G depende de D: sin prueba fuera de muestra no hay forma de distinguir una idea buena
de un sobreajuste.

| # | Objetivo | Aceptación binaria | Revisión | Lote |
|---:|---|---|---|---|
| V01 | Decisión: lista de ETFs, rango de fechas y dónde vive el snapshot | Doc ≤40 líneas con opciones y tamaño en disco | decidir | A |
| V02 | `data snapshot`: baja diarios y escribe snapshot con manifiesto (símbolo, rango, filas, sha256) | Releer el snapshot da el mismo hash; con `--offline` no hay llamadas de red | probar | A |
| V03 | Reporte de calidad: huecos contra el calendario de mercado, velas inválidas, saltos anómalos | Test: a un snapshot se le borra un día y el reporte lo nombra | leer | A |
| V04 | Precios ajustados por splits y dividendos, documentado y testeado | Test con un split conocido: no hay salto de precio en la fecha | leer | A |
| V05 | `backtest run --snapshot <nombre> --symbol <s> --timeframe 1d` | Corre RSI sobre SPY diario real, exit 0 | probar | A |
| V06 | Las 3 estrategias de M1 sobre SPY diario real | Cada una con trades > 0 y `backtest verify` en `OK reproducible` | probar | A |
| V07 | Test golden sobre el snapshot real | Falla si cambia un trade o una métrica | leer | A |
| V08 | Decisión: comisión y slippage por clase de activo | Doc con 2–3 perfiles y el efecto de cada uno sobre una corrida | decidir | B |
| V09 | Perfil de costos por clase de activo; el CLI imprime el perfil usado | Cambiar el perfil cambia el PnL en el monto que predice el recálculo | probar | B |
| V10 | Referencia comprar y mantener: misma ventana, mismos costos | `recalc_metrics.py` reproduce la referencia sin importar `quantagent` | probar | B |
| V11 | Métricas nuevas: retorno anualizado, exceso sobre la referencia, % del tiempo en mercado | Recálculo independiente = CLI | probar | B |
| V12 | Diagnóstico: cómo afectan el tamaño de posición y el límite diario de pérdida en velas diarias | Doc con una corrida con y sin límite | decidir | B |
| V13 | Aplicar lo decidido en V12 | Según la decisión | probar | B |
| V14 | `backtest compare --strategies … --symbols … --snapshot …` | Tabla en stdout y CSV, una fila por estrategia × activo, con la referencia | probar | C |
| V15 | Agregar una estrategia = un archivo + una línea de registro | Test que registra una estrategia nueva sin tocar `quantagent/backtesting/` | leer | C |
| V16 | Parámetros de estrategia desde un archivo | La misma estrategia con dos archivos da dos filas distintas en `compare` | leer | C |
| V17 | Estrategia clásica diaria 1: cruce de medias 50/200 | Trades > 0 sobre SPY diario; test de señal a mano | probar | C |
| V18 | Estrategia clásica diaria 2: momentum absoluto de 12 meses | Ídem | probar | C |
| V19 | Diagnóstico: ¿el engine simula una cartera con capital compartido entre activos? | Doc con una corrida de 2 activos y la respuesta | decidir | C |
| V20 | La comparación se guarda como reporte en markdown | El archivo tiene la tabla y los parámetros de la corrida | leer | C |
| V21 | Decisión: partición (ajuste / prueba / reserva final) y umbrales de salida | Doc con fechas concretas y umbrales | decidir | D |
| V22 | `--from` / `--to` y candado de la reserva | Sin el flag explícito el CLI se niega a leer la reserva; cada acceso queda registrado | probar | D |
| V23 | Barrido de parámetros con registro de intentos | El registro tiene una fila por combinación probada | probar | D |
| V24 | Walk-forward: parámetros elegidos en cada ventana de ajuste, aplicados en la de prueba | Test con serie conocida; ninguna ventana de prueba se usa para elegir | probar | D |
| V25 | Veredicto por estrategia: tabla dentro / fuera de muestra y PASA / NO PASA | La salida coincide con los umbrales de V21 aplicados a mano | probar | D |
| V26 | Estadísticas de robustez: cantidad de trades, concentración del PnL, resultado por año, sensibilidad a ±20% de parámetros | Test con una serie donde un solo trade explica todo el PnL | leer | D |
| V27 | Decisión: proveedor, modelo y tope de gasto por informe | Doc con costo medido de un informe | decidir | E |
| V28 | `report explain`: informe en lenguaje llano desde los artefactos | El verificador confirma que cada número del informe existe en el artefacto | probar | E |
| V29 | Glosario: cada término técnico se explica la primera vez que aparece | Informe de ejemplo revisado por Fede | leer | E |
| V30 | `report critique`: revisor crítico con las estadísticas de V26 | Una estrategia sobreajustada a propósito recibe la objeción correcta | probar | E |
| V31 | `compare` y el veredicto adjuntan informe y crítica; sin API key siguen funcionando | `env -u` de las claves: exit 0 sin informe | probar | E |
| V32 | Decisión: los 2 activos en 4h y la fuente de datos | Doc: Yahoo (730 días) contra una fuente alternativa | decidir | F |
| V33 | Snapshot intradía con sesiones de mercado y reporte de calidad | Sin velas fuera de horario para activos con horario | probar | F |
| V34 | Anualización del Sharpe verificada con datos reales con horario | Recálculo independiente = CLI | leer | F |
| V35 | `compare` y veredicto sobre el snapshot intradía | Tabla y veredicto para los 2 activos | probar | F |
| V36 | Decisión: formato de una hipótesis (idea, fuente, regla, parámetros, qué la refutaría) | Doc con 2 ejemplos escritos a mano | decidir | G |
| V37 | `research propose`: el modelo genera hipótesis en ese formato, sin código | Las hipótesis validan contra el esquema | leer | G |
| V38 | `research implement`: hipótesis → estrategia determinista con tests | La estrategia se registra y corre en `compare` | probar | G |
| V39 | Cada candidata pasa sola por ajuste y prueba; suma al registro de intentos | Ninguna corrida del generador toca la reserva | probar | G |
| V40 | Tabla de todas las candidatas probadas, incluidas las descartadas | La cantidad de filas = intentos registrados | leer | G |
| V41 | Cierre de M2: las que pasaron se evalúan una vez sobre la reserva final | Informe final y decisión de Fede | decidir | G |

**Fuera del loop**

- `QuantAgent-wwi`, nota macro semanal (capa L1): arranca en paralelo. Instalar el skill en `~/.hermes` requiere
  confirmación de Fede y las 4 decisiones de `2026-09-17_borrador_skill_revision_macro_semanal.md` §2.
- Siguen fuera: `u0w`, la UI de Streamlit, `kkj.10`, `kkj.11`, datos en tiempo real y todo M3.

## 6. Qué depende de M1

| De M1 | Afecta |
|---|---|
| T23, ADR del engine | Lotes A a C: comandos y archivos cambian si se adopta `backtesting.py` |
| T13 y T14, anualización del Sharpe | V11 y V34 |
| T08c, slippage por defecto | V08 y V09 parten de ese valor |
| T15 y T20, `verify` y test golden | V06 y V07 los reutilizan |

Estado al 2026-10-09: T23 eligió la opción B (PR #56), T14 la opción D (`backtest_metrics.md` §5.2), T08c 0,05% por lado, T15 y T20 hechos.

## 7. Preguntas abiertas

- V19: el engine recorre un activo y después el siguiente (`quantagent/backtesting/backtest.py`). Si no simula
  capital compartido, las estrategias de rotación entre activos quedan fuera de M2 o piden un ticket propio.
  **Respondido el 2026-10-09: no lo simula.** `backtest.py:363-367` recorre un activo entero y después el siguiente
  con la misma cartera. `compare` corre un par por vez (V23) y las carteras de varios activos quedan fuera de M2.
- Con 5 estrategias deterministas puede no haber ninguna que pase. Ese resultado es válido y está previsto en
  el criterio de salida.

---

## 8. Cierre de M1 (2026-10-09)

M1 se cumple al cerrar T23d (`PLAN-CONTINUACION.md` §4). Falta mergear `lote/engine-y-cierre-m1` a `main`; T23e
(`QuantAgent-dj0`) sigue abierto.

**Qué se hizo**
- Métricas auditadas contra un recálculo que no importa `quantagent`, y corrida reproducible e independiente de la base:
  lote L1, PR #22 (`scripts/recalc_metrics.py`, `docs/03_design/backtest_metrics.md` §3).
- Tres estrategias con trades sobre fixtures deterministas (`tests/fixtures/spy-2y-1d.csv`, `spy-1y-4h.csv`): lote L2, PR #45.
- Engine igual a backtesting.py trade por trade: 227 trades al cierre (PR #53 y #55) y 340 dentro de la vela (PR #57 y
  #59), según `docs/05_acceptance_tests/QuantAgent-832-AC-crossval-trade-por-trade.md`.
- ADR: engine propio con stops dentro de la vela, opción B (PR #56, `docs/04_decisions/2026-10-ADR-backtest-engine-propio-vs-terceros.md`).
- Stops dentro de la vela por defecto (PR #60). Referencias: rsi/spy-90d 340 trades, PnL 4571.27; fifty-two-week-high/spy-2y-1d
  6 y −249.35; triple-screen/spy-1y-4h 6 y 174.34 (`docs/loop/REVIEW-LOG.md`, línea de #60; `tests/test_golden_rsi_spy_90d.py`).
- Inventario de ramas: 51 borrables y 18 para decidir (PR #54, `docs/loop/inventario-ramas-2026-10.md`). Nadie corrió los comandos.

**Qué no se hizo, o qué no prueba la evidencia**
- Una sola estrategia comparada contra backtesting.py (rsi), sobre un fixture sintético: spy-90d es 24/7 y su volumen
  tiene un único valor en las 2160 filas.
- Los tres fixtures tienen sábados y domingos (624, 148 y 624 velas): el engine nunca corrió sobre datos con huecos.
- Sin comisiones: existen en `quantagent/trading/paper_broker.py:52`, pero están apagadas y el backtest no las configura.
- Ninguna vela de spy-90d toca stop loss y take profit a la vez: el desempate solo lo cubren tests unitarios (REVIEW-LOG, #59).
- Datos (`QuantAgent-iuf`, PR #28 y #58, el segundo solo en `main`) y OpenBB (`QuantAgent-6ie`, PR #49): propuestas sin decisión.
- `milestone-tracker.md` sigue con el estado de septiembre ("2/14 tickets M1"). Lo actualiza el PM al mergear L3.

**Deuda que pasa a M2:** `QuantAgent-cub` → V17 y V21; `QuantAgent-col` → V37–V38; `QuantAgent-beo` → V53. El manual de
usuario (`docs/user-manual/`, último cambio el 2026-07-01; `backtesting.md:452` dice 1% de slippage) va a la decisión previa 6.

## 9. Qué cambia respecto del borrador

- **El lote 1 suma el reloj del engine (V05–V06) y el control con backtesting.py sobre datos reales (V10).** El candado de la reserva se adelanta (V08).
- **Las comisiones van antes que la referencia (V11–V14),** porque hoy el backtest no puede aplicarlas.
- **SPY 4h sale de Yahoo y no de Alpaca:** M2 no lleva tickets de Alpaca (decisión previa 5).
- **El generador de hipótesis no escribe código:** cada hipótesis aprobada entra a la cola del loop como estrategia (V51).
- **Cada criterio es un comando que corre el PM** con `scripts/loop/try.sh`. La prueba propia del agente no alcanzó en L1–L3:
  #37 y #39 volvieron con `cambio` después de la prueba del PM (REVIEW-LOG). Los tests con datos reales solo corren en la VM.
- **Librerías antes que piezas propias:** `yfinance`, `pandas_market_calendars` y `scipy` ya son dependencias; `pyarrow` para
  Parquet; backtesting.py solo en tests (ADR); `requests` para las velas de Binance (no está verificado que `ccxt` acepte el
  host espejo); OpenBB por MCP solo para la nota macro (`QuantAgent-6ie`).

## 10. Lotes de M2

Cada comando se corre en la VM con `scripts/loop/try.sh lote/<rama> -- '<comando>'`. Después del lote 4 hay un corte de
control: con la herramienta completa y los primeros veredictos, Fede decide si siguen los lotes 5, 6 y 7, y en qué orden.

**Lote 1 · `lote/datos-reales-diarios` · V01–V10.** Las 3 estrategias de M1 corren sobre SPY diario real 2007–2018 desde
un snapshot con manifiesto. El engine avanza por las velas del dato y backtesting.py coincide con RSI.
- Probar: `python -m quantagent.cli data snapshot check --name etf-1d-2026-10 && python scripts/crossval_compare.py --snapshot etf-1d-2026-10 --symbol SPY --from 2007-01-03 --to 2018-12-31 --intrabar | tail -3 && for s in rsi fifty-two-week-high triple-screen; do python -m quantagent.cli backtest verify --strategy $s --snapshot etf-1d-2026-10 --symbol SPY --from 2007-01-03 --to 2018-12-31; done`
- Esperado: `faltan 0` en los 12 símbolos; `TODO DENTRO DE TOLERANCIA` (o las diferencias que Fede aceptó en V10); tres `OK reproducible`.
- Decide Fede: V10, qué regla manda si el engine y el control difieren sobre datos reales.

**Lote 2 · `lote/costos-y-referencia` · V11–V20.** La corrida informa comisión, slippage y tamaño, y se compara contra comprar
y mantener con los mismos costos. Un recálculo independiente coincide.
- Probar: `python -m quantagent.cli backtest run --strategy rsi --snapshot etf-1d-2026-10 --symbol SPY --from 2007-01-03 --to 2018-12-31 --costos etf --tamano-posicion 1.0 --out /tmp/t.csv && python scripts/recalc_metrics.py /tmp/t.csv --commission-pct <la del perfil etf> | grep "PnL por trade" && python -m pytest -q -m snapshot tests/test_golden_snapshot.py`
- Esperado: líneas `Costos: etf`, `Tamaño: 100% por trade` y `Comprar y mantener: …`; `N/N filas coinciden`; `1 passed`.
- Decide Fede: V13 (costos), V17 y V18 (límite diario y tamaño).

**Lote 3 · `lote/comparacion` · V21–V29.** Un comando corre la matriz de estrategias × 12 ETFs y escribe CSV y markdown.
Agregar una estrategia es un archivo y una línea, y backtesting.py controla una segunda estrategia.
- Probar: `python -m quantagent.cli backtest compare --strategies rsi,fifty-two-week-high,triple-screen,sma-cross,momentum-12m --symbols SPY,IWM,EFA,EEM,TLT,SHY,GLD,DBC,XLK,XLE,XLF,XLU --snapshot etf-1d-2026-10 --to 2018-12-31 --costos etf --out /tmp/cmp.csv && wc -l /tmp/cmp.csv`
- Esperado: tabla en pantalla y `61 /tmp/cmp.csv` (60 pares más el encabezado).
- Decide Fede: nada previsto (V21 ejecuta lo decidido en V18).

**Lote 4 · `lote/fuera-de-muestra` · V30–V36.** Hay veredicto PASA / NO PASA por estrategia y activo sobre la prueba 2019–2022,
con intentos contados y la reserva cerrada.
- Probar: `python -m quantagent.cli backtest verdict --snapshot etf-1d-2026-10 --to 2022-12-31 --costos etf --out /tmp/v.csv && python scripts/recalc_verdict.py /tmp/v.csv && cat $QUANTAGENT_SNAPSHOT_DIR/reserva-accesos.log 2>/dev/null | wc -l`
- Esperado: tabla PASA / NO PASA, el recálculo coincide en todas las filas y `0` aperturas de la reserva.
- Decide Fede: V30 (umbrales, antes de ver la prueba) y el corte de control.

**Lote 5 · `lote/intradia-4h` · V37–V42.** El mismo veredicto sobre SPY 4h y BTC 4h, con hora UTC explícita.
- Probar: `python -m quantagent.cli data snapshot check --name intradia-4h-2026-10 && python -m quantagent.cli backtest verdict --snapshot intradia-4h-2026-10 --out /tmp/v4.csv`
- Esperado: SPY con 0 velas fuera de horario, BTC con sus huecos listados y veredicto para los 2.
- Decide Fede: V37 (convención de hora) y los cortes de 4h antes de V42.

**Lote 6 · `lote/informe-y-critico` · V43–V48.** Cada comparación trae un informe en lenguaje llano y una crítica; cada número
del texto existe en el artefacto. Sin API key, todo lo demás sigue andando.
- Probar: `env -u OPENAI_API_KEY -u ANTHROPIC_API_KEY python -m quantagent.cli backtest compare --strategies rsi --symbols SPY --snapshot etf-1d-2026-10 --to 2018-12-31 --informe; echo $?`
- Esperado: `informe omitido: sin clave` y `0`. Con clave, el verificador de V44 da OK.
- Decide Fede: V43 (proveedor, modelo, tope de gasto y formato).

**Lote 7 · `lote/hipotesis-y-cierre` · V49–V55.** Un modelo propone hipótesis, el loop las implementa como estrategias y las
que pasan la prueba se miden una sola vez sobre la reserva.
- Probar: `cat $QUANTAGENT_SNAPSHOT_DIR/reserva-accesos.log | wc -l`
- Esperado: `1` (la apertura de V54) e informe final en el PR del lote.
- Decide Fede: V49, V53 (`beo`, antes de M3) y V55 (cierre de M2).

## 11. Cola de M2

Mismo formato que `PLAN-CONTINUACION.md` §4. El PM la copia ahí cuando Fede la aprueba. **(VM)** = necesita el snapshot.

| # | Ticket | Objetivo | Aceptación binaria | Revisión | Lote |
|---:|---|---|---|---|---|
| V00 | `QuantAgent-vfd` | Diagnóstico de performance de la suite: tests más lentos, causa y mejoras con ahorro estimado | Doc con la tabla de `--durations` de una corrida real y un objetivo de tiempo; nada fuera de `docs/` | decidir | 1 |
| V01 | nuevo | Snapshot: Parquet por símbolo con manifiesto (hashes, rango, filas, versión de `yfinance`) | `pytest -q tests/test_snapshot.py`: un byte cambiado hace fallar `verify_snapshot` | leer | 1 |
| V02 | nuevo | `data snapshot create` y `verify` desde Yahoo; no pisa un nombre existente | (VM) crea `etf-1d-2026-10` y `verify` da `OK 12 símbolos`; repetir `create` sale con 1 | probar | 1 |
| V03 | nuevo | Precios ajustados por splits y dividendos; guarda el cierre sin ajustar | Test con split 2:1 a mano: el ajustado no salta y el sin ajustar sí | leer | 1 |
| V04 | nuevo | `data snapshot check`: sesiones contra NYSE, OHLC incoherente, volumen 0, saltos | Test que borra un día y lo nombra; (VM) `faltan 0` en los 12 | probar | 1 |
| V05 | nuevo | Fixture `spy-90d-habiles` (sin fines de semana) y `--fixture` en `crossval_*` | `crossval_compare.py --fixture spy-90d-habiles --intrabar` imprime resultado y `Entradas en fin de semana: N` | probar | 1 |
| V06 | nuevo | El engine avanza por las velas del dato, no por una grilla de calendario | Sobre `spy-90d-habiles`: `TODO DENTRO DE TOLERANCIA` y 0 en fin de semana; referencias de M1 sin cambio | probar | 1 |
| V07 | nuevo | `backtest run --snapshot --symbol --from --to` | (VM) rsi/SPY 2007–2018 exit 0 con 6 líneas; golden de M1 verde | probar | 1 |
| V08 | nuevo | Candado de la reserva (desde 2023-01-01) con registro de aperturas | `--to 2024-01-01` sin `--abrir-reserva` sale con 1; con el flag, el registro suma una línea | probar | 1 |
| V09 | nuevo | `backtest verify --snapshot`; las 3 estrategias sobre SPY 2007–2018 | (VM) tres `OK reproducible`; el PR informa trades y tiempo | probar | 1 |
| V10 | nuevo | Control con backtesting.py sobre SPY real; cuenta velas que tocan stop y take profit | (VM) `TODO DENTRO DE TOLERANCIA` o cada diferencia con su causa | decidir | 1 |
| V11 | nuevo | Comisiones del `PaperBroker` en backtest y CLI (`--commission-pct`, default 0) | Default: referencias sin cambio; con 0.001, PnL menor y `Comisión: 0.10% por lado` | probar | 2 |
| V12 | nuevo | `recalc_metrics.py --commission-pct` (entrada y salida) | `N/N filas coinciden`, o evidencia de que falta la comisión de entrada (escala) | probar | 2 |
| V13 | nuevo | Decisión: perfiles de costos ETF y cripto, con fuente | Doc ≤40 líneas con efecto medido por perfil; el PM reproduce una fila | decidir | 2 |
| V14 | nuevo | `--costos <perfil>` | Diferencia de PnL entre perfiles = la de `recalc_metrics.py` (±0.01); sin flag, 340 / 4571.27 | probar | 2 |
| V15 | nuevo | Referencia comprar y mantener en el CLI, con los mismos costos y ventana | Test con 3 velas a mano; (VM) aparece `Comprar y mantener:` | probar | 2 |
| V16 | nuevo | `recalc_metrics.py --comprar-y-mantener` sin importar `quantagent` | Igual a la línea del CLI sobre spy-90d (±0.01) | probar | 2 |
| V17 | `QuantAgent-cub` | ¿El límite de pérdida diaria actúa en backtest? (`date.today()`) | `xfail(strict=True)` si lo confirma; respuesta con evidencia en el PR | decidir | 2 |
| V18 | nuevo | Decisión: tamaño de posición contra la referencia (hoy 5%) y regla del límite diario | Doc ≤40 líneas con una corrida por opción; el PM reproduce una fila | decidir | 2 |
| V19 | nuevo | `--tamano-posicion` (opción B de V18) | Test: qty = equity × tamaño × confianza / precio en 3 trades; sin flag, referencias sin cambio | probar | 2 |
| V20 | nuevo | Golden sobre el snapshot real, solo en la VM | (VM) pasa y falla con un dígito cambiado; en CI, skipped con motivo | leer | 2 |
| V21 | `QuantAgent-cub` | Si V18 lo decide: el límite diario toma el día de la vela | El `xfail` de V17 pasa a test normal; referencias cambian solo como anticipó V18 | probar | 3 |
| V22 | nuevo | Métricas: retorno anualizado, exceso sobre la referencia, % del tiempo en mercado | `recalc_metrics.py` = CLI en las tres | probar | 3 |
| V23 | nuevo | `backtest compare --strategies --symbols --snapshot`, una corrida por par | Una fila por estrategia × activo con su referencia; cada fila = `backtest run` del par | probar | 3 |
| V24 | nuevo | `compare` escribe markdown con parámetros, hash del snapshot, costos y fechas | Dos corridas dan el mismo archivo, con esos 4 datos | leer | 3 |
| V25 | nuevo | Parámetros de estrategia desde un archivo JSON | Dos archivos dan dos filas distintas en `compare` | leer | 3 |
| V26 | nuevo | Agregar una estrategia = un archivo + una línea de registro | Test que registra una sin tocar `quantagent/backtesting/` ni el CLI | leer | 3 |
| V27 | nuevo | Estrategia diaria: cruce de medias 50/200 (`sma-cross`) | Trades > 0 sobre SPY 2007–2018; test de señal con serie a mano | probar | 3 |
| V28 | nuevo | Estrategia diaria: momentum absoluto de 12 meses (`momentum-12m`) | Ídem | probar | 3 |
| V29 | nuevo | backtesting.py controla también `sma-cross` | `crossval_compare.py --strategy sma-cross --intrabar`, fixture (CI) y SPY (VM): `TODO DENTRO DE TOLERANCIA` | probar | 3 |
| V30 | nuevo | Decisión: umbrales de salida, antes de abrir la prueba 2019–2022 | Doc con umbrales numéricos por métrica, contra la referencia | decidir | 4 |
| V31 | nuevo | Barrido de parámetros con registro de intentos (CSV que solo crece) | Filas del registro = combinaciones probadas; nada lee después de 2018-12-31 | probar | 4 |
| V32 | nuevo | Walk-forward dentro del ajuste | Test con serie conocida: ninguna ventana de prueba elige parámetros | probar | 4 |
| V33 | nuevo | `backtest verdict`: dentro y fuera de muestra, PASA / NO PASA por par | Coincide con los umbrales de V30 aplicados a mano | probar | 4 |
| V34 | nuevo | Robustez: trades, concentración del PnL, resultado por año, ±20% de parámetros | Test donde un trade explica todo el PnL: concentración 100% | leer | 4 |
| V35 | nuevo | Sharpe deflactado por la cantidad de intentos (Bailey y López de Prado) | Test contra un valor a mano; con más intentos, baja | leer | 4 |
| V36 | nuevo | `scripts/recalc_verdict.py`: veredicto recalculado sin importar `quantagent` | Coincide con el CLI en todas las filas | probar | 4 |
| V37 | `QuantAgent-col` | Confirmar o descartar el corrimiento de hora con velas reales de SPY | Test que lo reproduce o evidencia de que no ocurre, en diario y 4h | decidir | 5 |
| V38 | nuevo | UTC explícito de punta a punta (si V37 lo confirma) y destino del caché | 0 velas de SPY fuera de 9:30–16:00 de Nueva York | probar | 5 |
| V39a | nuevo | Adaptador de datos de Alpaca, solo lectura (velas históricas); sin órdenes ni cuenta | Test con respuestas grabadas; un test falla si el adaptador llama a un endpoint que no sea de datos de mercado; sin claves, sale con un mensaje claro | leer | 5 |
| V39 | nuevo | Snapshot de SPY 4h desde Alpaca (feed SIP, desde 2016); Yahoo (`period="730d"`) queda como alternativa si la cuenta gratuita no alcanza | (VM) `snapshot check`: 0 fuera de horario; el PR informa primera vela y cantidad (iuf estima ~10,7 años, sin medir) | probar | 5 |
| V40 | nuevo | Snapshot de BTC 4h desde Binance (`data-api.binance.vision`, host configurable) | Desde 2017-08-17; `snapshot check` lista los huecos (iuf midió 16) | probar | 5 |
| V41 | nuevo | Sharpe anualizado verificado con horario (SPY) y 24/7 (BTC) | `recalc_metrics.py` = CLI en los dos | leer | 5 |
| V42 | nuevo | `compare` y `verdict` sobre SPY 4h y BTC 4h | Tabla y veredicto de los 2; el registro de la reserva no cambia | probar | 5 |
| V43 | nuevo | Decisión: proveedor, modelo, tope de gasto por informe y formato (markdown o HTML) | Doc con el costo medido de un informe | decidir | 6 |
| V44 | nuevo | Verificador: cada número del texto existe en el artefacto | Test: un texto con un número inventado falla | leer | 6 |
| V45 | nuevo | `report explain`: informe en lenguaje llano desde `compare` y `verdict` | El verificador de V44 da OK sobre el informe de ejemplo | probar | 6 |
| V46 | nuevo | Glosario: cada término se explica en su primera aparición | Test sobre el informe de ejemplo con la lista del glosario | leer | 6 |
| V47 | nuevo | `report critique`: revisor crítico con las estadísticas de V34–V35 | Una estrategia sobreajustada a propósito recibe la objeción correcta | probar | 6 |
| V48 | nuevo | `compare` y `verdict` con `--informe` adjuntan informe y crítica; sin clave siguen andando | `env -u` de las claves: exit 0 y `informe omitido: sin clave` | probar | 6 |
| V49 | nuevo | Decisión: formato de una hipótesis (idea, fuente, regla, parámetros, qué la refuta) | Doc con 2 ejemplos escritos a mano | decidir | 7 |
| V50 | nuevo | `research propose`: hipótesis en ese formato, sin código ni datos de la reserva | Las hipótesis validan contra el esquema | leer | 7 |
| V51 | nuevo | Hipótesis aprobada → ticket del loop → estrategia (V26); no hay generador de código | Una hipótesis de ejemplo llega a `compare` con sus tests | probar | 7 |
| V52 | nuevo | Tabla de candidatas probadas, incluidas las descartadas | Filas = intentos del registro | leer | 7 |
| V53 | `QuantAgent-beo` | Diagnóstico de la regla de salida duplicada entre backtest y paper (previo a M3) | Doc con tabla de diferencias (archivo y línea) y unificación partida en entregas | decidir | 7 |
| V54 | nuevo | Apertura única de la reserva para las que pasaron la prueba | El registro de la reserva tiene exactamente una apertura | probar | 7 |
| V55 | nuevo | Cierre de M2: informe final y decisión (M3 con estrategia, M3 sin estrategia o cortar) | Decisión escrita en la `R:` del lote | decidir | 7 |

## 12. Riesgos y supuestos

1. **Reloj del engine (alto).** El engine avanza por una grilla de calendario: cada 1 h, 4 h o 1 día desde la fecha inicial (`quantagent/backtesting/backtest.py:706-732`). El CLI apaga el filtro de horario (`quantagent/cli/backtest.py:149` y `:233`). Leyendo el código, sin correrlo: un sábado o un feriado, el engine vuelve a evaluar la última vela que existe. Por ejemplo, después de una salida el viernes puede entrar el sábado al cierre del viernes, un precio al que no se opera. Lo cubren V05 y V06, antes de cualquier número real.
2. **Cartera compartida.** Una corrida con varios activos no simula capital compartido (§7). `compare` corre un par por vez.
3. **Comisión de entrada.** El PnL de un trade descuenta solo la comisión de la orden de cierre (`quantagent/portfolio/manager.py:155-178`, leído, sin probar). V12 lo mide; si se confirma, es un bug de plata y el PM escala.
4. **Licencias y pérdida de datos.** Los términos de Yahoo prohíben la extracción automática y Binance Vision es CC BY-NC-SA, sin fin comercial (iuf D4 en `main`). El snapshot fuera del repo no se recrea igual (iuf D5). Lo cubren las decisiones 1 y 2, más la copia en Drive.
5. **Tiempo de corrida, no medido.** El AC de `832` da ≈90 s para los dos motores sobre 2160 velas. 60 pares de 12 años diarios pueden llevar del orden de una hora; es una estimación. V09 lo mide y, si no entra, V23 corre una muestra.
6. **Sesgo por intentos.** Cada combinación probada sube la chance de un falso positivo. Lo cubren el registro (V31) y el Sharpe deflactado (V35). Puede que ninguna estrategia pase: es un resultado válido (§3) y Fede decide en V55.
7. **Doble toque y SPY 4h corto.** En velas diarias reales una vela puede tocar stop y take profit, y ese desempate nunca se comparó contra el control (V10). SPY 4h trae ~2,9 años en Yahoo (iuf: 1.450 velas desde 2023-11-09): sirve para comparar horizontes, no para un veredicto fuerte.
8. **Ritmo.** El tope es de 10 entregas por día (§3.7); el 07 y el 08/10 se cerraron 7 y 8 (REVIEW-LOG). Una corrida del agente tarda de 20 a 100 min y la suite unos 15 (dato del encargo de T25, no verificado en el repo). A 5 filas por día, las 55 llevan unas 11 jornadas de loop (estimación), más las esperas de 10 filas `decidir` y 7 PR de lote.

Supuestos: la VM llega a Yahoo (`QuantAgent-577`). `data-api.binance.vision` respondió desde el entorno de medición de iuf; desde la VM no está verificado. No está verificado que `pyarrow` ya esté en la VM: V01 lo declara. `try.sh` solo exporta `DATABASE_URL` (`scripts/loop/try.sh:17`), así que `QUANTAGENT_SNAPSHOT_DIR` tiene que estar exportada en el shell. L3 se mergea a `main`, con `dj0` cerrado, antes del lote 1.

## 13. Decisiones de Fede antes de arrancar

1. **Datos (`QuantAgent-iuf`):** 12 ETFs desde 2007-01-03 (D1-B, D2-A), en Parquet fuera del repo con manifiesto versionado (D5-C), en `QUANTAGENT_SNAPSHOT_DIR` de la VM y con copia en Drive. **Recomiendo** aprobarlo tal cual: sin esto no arranca V02.
2. **Licencia de Yahoo:** **recomiendo** aceptarla para investigación personal, sin publicar precios ni trades derivados, y revisarla antes de cualquier uso comercial.
3. **Cortes de la historia diaria, desde ya:** ajuste 2007–2018, prueba 2019–2022 y reserva desde 2023 (el ejemplo de iuf D2). Los lotes 1 a 3 solo leen el ajuste; la prueba se abre en el lote 4, con los umbrales ya fijados (V30), y la reserva una sola vez (V54). **Recomiendo** sí: lo que se mira antes de tiempo deja de servir como prueba.
4. **El engine avanza por las velas del dato (V06),** con la condición de que no cambie ningún número de referencia de M1. **Recomiendo** sí; si cambia alguno, el PM escala.
5. ~~**4h sin Alpaca:** SPY desde Yahoo y BTC desde Binance.~~ **Cambiada por Fede el 2026-10-09 (`R:` del PR #62): SPY 4h desde Alpaca, solo datos; BTC desde Binance.** Alpaca como broker (órdenes, posiciones, conciliación) sigue en M3. Condición: sin suscripciones ni depósito mínimo; si la cuenta gratuita no alcanza, se vuelve a Yahoo. Fede crea la cuenta y deja las claves en la VM antes del lote 5. Los datos diarios siguen en Yahoo: Alpaca arranca en 2016 y deja afuera 2008 (iuf D2-C). Filas V39a y V39 de §11. Además, por pedido de Fede del mismo día, el lote 1 suma V00 (`QuantAgent-vfd`).
6. **Manual de usuario:** **recomiendo** darlo de baja. Al abrir el lote 1, el PM agrega en `docs/user-manual/index.md` un aviso que apunta al README. El manual cubre UI, paper y agentes LLM, que M2 no toca, y mantenerlo en cada lote cuesta una revisión que nadie lee. El README sigue al día por la regla 4 de §3.4.
7. **OpenBB (`QuantAgent-6ie`):** **recomiendo** aceptar lo que propone el doc: no usarlo para precios y usar su MCP para la nota macro, fuera del loop. No bloquea el lote 1.

Las demás decisiones se toman dentro de los lotes: costos (V13), límite diario y tamaño (V17–V18), umbrales (V30), hora (V37), informe (V43), hipótesis (V49) y cierre (V55).

**Fuera de M2 o fuera del loop.** Alpaca: `13a`, `yme`, `qr6` y `y62` son M3, y `b41` es M4; sus labels `m2` son de la numeración vieja. La nota macro (`QuantAgent-wwi`, con los números de `QuantAgent-577`) es uno de los cuatro usos de la IA generativa; los otros tres son V45, V47 y V50. Corre en paralelo, fuera del loop. `QuantAgent-jh2` lo hace el PM antes del 2026-11-04 y el borrado de ramas lo corre Fede. Quedan fuera de M2 la UI, la estrategia LLM de 4 agentes, `kkj.10`, `kkj.11` y `u0w`.

## 14. Glosario

- **Snapshot:** los datos bajados una vez y congelados, fuera del repo. **Manifiesto:** un archivo chico, dentro del repo, con el hash de cada archivo: prueba que dos corridas usaron el mismo dato.
- **Ajuste, prueba y reserva:** tres tramos de la historia. En el ajuste se eligen los parámetros, en la prueba se juzga el resultado y la reserva se abre una sola vez, al final. Lo que se mira antes deja de servir para juzgar.
- **Comprar y mantener:** comprar el primer día y vender el último. Es la vara mínima: una estrategia que no la supera después de costos no agrega nada.
- **Dentro de la vela:** el stop loss y el take profit se comparan con el máximo y el mínimo de cada vela, no solo con el cierre.
- **Control (backtesting.py):** una librería de terceros que rehace los mismos trades por su cuenta. Si da distinto, uno de los dos está mal.
- **Golden:** un test que guarda una corrida completa y falla si cambia un trade o una métrica.
- **Walk-forward:** elegir los parámetros en un tramo y medirlos en el siguiente, avanzando en el tiempo. **Sharpe deflactado:** un Sharpe que exige más cuantas más combinaciones se probaron.
