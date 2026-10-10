# QuantAgent-beo — La regla de salida está escrita dos veces

Diagnóstico, sin cambios de código. Rama base: `lote/costos-y-referencia` (60318ff4).
Test que lo demuestra: `tests/test_exit_rule_divergence.py` (4 casos, funciones reales).

**En una línea:** el backtest y paper deciden distinto cuándo cerrar, a qué precio
y con qué texto de razón. Lo que hoy valida el backtest no es lo que corre en paper.

Abreviaturas: `base` = `quantagent/strategy/base.py`, `bt` = `quantagent/backtesting/backtest.py`,
`sch` = `quantagent/trading/scheduler.py`, `pm` = `quantagent/trading/position_monitor.py`,
`om` = `quantagent/trading/order_manager.py`.

## a. Tabla de diferencias

| Aspecto | Backtest | Paper |
|---|---|---|
| Función | `should_exit` (`base:81`) + `check_intrabar_stops` (`bt:102`) | `_check_exit_conditions` (`sch:583`) |
| Condiciones | stop, take profit, trailing, tiempo (`base:103-130`); antes, stop y TP dentro de la vela (`bt:137-171`) | stop, take profit, max hold (`sch:598-617`) |
| Orden | 1) gap de apertura, 2) stop intravela, 3) TP intravela (`bt:139-151`); si nada, 4) stop, 5) TP, 6) trailing, 7) tiempo al cierre (`base:104-130`) | 1) stop, 2) TP, 3) max hold (`sch:599-617`) |
| Precio que compara | apertura, máximo y mínimo de la vela (`bt:139-146`); después el cierre (`bt:810`, `bt:837`) | solo el cierre de la última fila que devuelve el proveedor (`sch:242`, `sch:269`) |
| Precio de salida | nivel del stop o del TP, o la apertura si hubo gap (`bt:140`, `bt:149`, `bt:151`); el cierre en el resto (`bt:843`). Se guarda en `Trade.exit_price` (`bt:1286`) | el cierre (`sch:275`). `close_position` recibe el precio y no lo guarda (`pm:121-145`); `sch:680-686` tampoco escribe `exit_price` |
| Razones (texto) | `STOP_LOSS`, `TAKE_PROFIT` (`bt:140-151`, `base:106`, `base:114`), `TRAILING_STOP` (`base:122`), `TIME_EXPIRED` (`base:130`) | `stop_loss` (`sch:601`), `take_profit` (`sch:609`), `max_hold` (`sch:617`) |
| Trailing stop | sí, si la política es `TRAILING_STOP` (`base:120`). Piso = máximo de cierres × (1 − pct) (`base:155-164`) | no existe. Además la posición se abre con `"sl_tp_only"` y sin `trailing_stop_pct` (`sch:370-382`) |
| Salida por tiempo | solo si la política es `TIME_BASED` (`base:125-130`) | con cualquier política (`sch:615-617`) |
| Conteo de velas | suma después de evaluar, y solo si no cerró (`bt:854`) | suma antes de evaluar (`sch:254`, `sch:269`), una vez por ciclo, no por vela nueva |
| Quién llama | `_analyze_and_trade` (`bt:821`, `bt:837`) y `_replay_and_trade` (`bt:559`, `bt:573`) | `_process_asset` (`sch:269`) |
| Cada cuánto | una vez por vela del período (ver QuantAgent-55x para corridas online) | una vez por ciclo: `interval_hours` (`sch:102`) |
| Consulta a la estrategia | sí: `self.strategy.should_exit` | no |
| Cómo cierra | `order_manager.close_trade` (`bt:1262`) | `execute_decision` con la señal contraria (`sch:643-663`) |
| Parámetros al abrir | política, trailing y max hold de la señal (`bt:944-948`); TP por defecto +3% (`bt:939`) | política fija, sin trailing ni max hold (`sch:377`); TP por defecto +4% (`sch:366`) |

**Estrategias que sobreescriben la regla:** ninguna. `def should_exit` y
`def _check_trailing_stop` aparecen una sola vez en el repo (`base:81`, `base:134`).
Las cuatro estrategias (RSI, 52 semanas, Triple Screen, LLM) usan la regla base y
todas piden trailing: `rsi_strategy.py:121-122`, `fifty_two_week_high_strategy.py:95-96`,
`triple_screen_strategy.py:116-117`, `llm_agent_strategy.py:113` (más el default de
`base:30`). Ninguna define `max_hold_candles`.

**Replay del Streamlit:** `apps/streamlit/views/replay.py:74-86` arma un `Backtest`
y llama `run_replay`, o sea usa la regla del backtest (`bt:557-588`), no la de paper.

## b. Consecuencias con números

Los tres primeros están en el test. Posición: LONG a 100, stop 90, TP 130.

**1. Trailing stop (5%).** Cierres: 110, 120, 113, 105, 95, 89.
- Backtest: máximo visto 120, piso 120 × 0,95 = 114. En 113 cierra por
  `TRAILING_STOP`. Resultado: **+13** por acción.
- Paper: no mira trailing. Sigue hasta 89, cierra por `stop_loss`. Resultado: **−11**.
- Diferencia: 24 por acción sobre una entrada de 100. El backtest muestra una
  ganancia que paper convierte en pérdida.

**2. Stop tocado dentro de la vela.** Vela: apertura 100, máximo 101, mínimo 89, cierre 95.
- Backtest: el mínimo 89 pasa el stop de 90. Cierra a **90** (−10), `STOP_LOSS`.
- Paper: el cierre 95 está arriba de 90. No sale. Si la vela siguiente cierra en
  84, sale ahí: **−16**. Si rebota a 110, paper sigue adentro de una posición que
  en el backtest ya no existe.

**3. Max hold.** Política `sl_tp_only`, `max_hold_candles=5`, 5 velas contadas, precio 105.
- Backtest: no sale (exige `TIME_BASED`). Paper: sale por `max_hold`.
- Hoy es latente: ninguna estrategia define max hold y el scheduler no lo pasa.

**4. Texto de la razón.** Con precio 89 las dos salen, pero una escribe `STOP_LOSS`
y la otra `stop_loss` en `close_reason` / `exit_signal`. Cualquier conteo por razón
que mezcle corridas de backtest y de paper las cuenta como dos motivos distintos.

**5. La salida de paper puede dar vuelta la posición (leído, no ejecutado).**
Paper cierra mandando la señal contraria (`sch:643-663`). `execute_decision`
detecta posición contraria y entra en `_execute_reversal` (`om:114-127`), que
según su docstring "cierra la existente y abre una nueva" (`om:268-280`). Después
`_process_asset` hace `return` (`sch:290`) sin crear `ActivePosition` para esa
posición nueva: quedaría una posición opuesta sin stop. Es más grave que la
duplicación y conviene confirmarlo antes de M3.

## c. Propuesta de unificación

**Opciones**

- **A. Paper llama a `strategy.should_exit`.** Diff mínimo y respeta futuras
  sobreescrituras. Problemas: no cubre el stop intravela (vive en `bt`); el
  adaptador del piloto no hereda de `TradingStrategy` y no tiene `should_exit`
  (`scripts/run_paper_pilot.py:79-111`); sola no activa el trailing, porque paper
  abre con `sl_tp_only` (`sch:377`).
- **B. Función pura compartida** `(posición, vela) → (razón, precio de salida)` en
  un módulo neutral. `should_exit` delega en ella; backtest y paper la usan.
  Una sola verdad, incluido el precio de salida. Problema: toca el backtest.
- **C. Dejar las dos y agregar un test de paridad** que falle si divergen. Barato,
  pero siguen siendo dos reglas.

**Recomiendo B como destino, llegando por A.** Paper pasa a preguntarle a la
estrategia y el código del backtest no cambia de comportamiento en ninguna
entrega, así los números de referencia no se mueven. La función pura sale al
final como movida de código sin cambio de resultado.

**Entregas** (cada una ≤100 líneas de diff, en este orden)

| # | Qué | Aceptación (sí/no) | Números que mueve |
|---|---|---|---|
| 1 | Razones como constantes en un solo lugar; el scheduler emite mayúsculas | El caso 4 del test pasa con igualdad; no queda `"stop_loss"` como razón en `sch` | Backtest: ninguno. Paper: filas viejas quedan en minúscula |
| 2 | El scheduler abre la posición con política, trailing y max hold de la señal, igual que `bt:944-948`; se decide 3% o 4% de TP por defecto | Test: señal con trailing 5% deja `trailing_stop_pct == 0.05` y política `TRAILING_STOP` en la posición | Backtest: ninguno |
| 3 | El scheduler llama a `strategy.should_exit` (con la regla base si la estrategia no la tiene) y se borra `_check_exit_conditions` | Casos 1 y 3 del test pasan con igualdad | Backtest: ninguno. Paper: empieza a salir por trailing; max hold solo con `TIME_BASED` |
| 4 | `check_intrabar_stops` se muda a un módulo neutral (reexportada desde `bt`) y paper la evalúa sobre la última vela cerrada | Caso 2 del test pasa con igualdad; `tests/test_intrabar_stops.py` sigue verde | Backtest: ninguno si es solo mudanza. Ojo: `scripts/crossval_compare.py:80-93` la parchea por nombre |
| 5 | Paper cuenta velas igual que el backtest: después de evaluar y por vela nueva | Test: con max hold N las dos reglas salen en la misma vela | Backtest: ninguno. Depende de QuantAgent-55x |
| 6 | Función pura única que junta intravela y cierre; `should_exit` delega | Golden y crossval dan exactamente los mismos números que antes | No debería mover ninguno: ese es el criterio |

Fuera de esta unificación, para ticket aparte: que el trailing use el máximo de
la vela en vez del cierre (`base:155-164`). Eso sí mueve los números de referencia.

**Para decidir (Fede)**

1. ¿B llegando por A, como arriba? Recomiendo sí.
2. ¿Cuál regla es la verdad cuando difieren: la del backtest? Recomiendo sí: es la
   que tiene números validados.
3. Max hold: ¿solo con `TIME_BASED` (backtest) o siempre (paper)? Recomiendo la del
   backtest; hoy nadie usa max hold.
4. TP por defecto: ¿3% (`bt:939`) o 4% (`sch:366`)? Recomiendo 3%, el del backtest.
5. ¿El punto b.5 va como ticket propio y antes que esto? Recomiendo sí.

## d. Qué NO verifiqué

- No corrí el scheduler ni un backtest completo. El test llama a las tres
  funciones sueltas; no ejercita el ciclo `_process_asset` ni `_analyze_and_trade`.
- El punto b.5 (reversión al salir en paper) sale de leer el código, no de ejecutarlo.
- No verificado: si la última fila que recibe paper es una vela cerrada o una en
  formación (`sch:417-434`). De eso depende si paper compara contra "cierre" o
  contra "último precio".
- No verificado: qué precio de llenado termina en `Trade.exit_price` cuando hay
  orden de cierre con costos (`bt:1304-1316`).
- No verificado: qué lee las razones en minúscula. Solo confirmé que
  `tests/trading/test_scheduler_position_monitor*.py` nombran la regla de paper.
- No medí cuántas salidas por trailing hay en las corridas de referencia, o sea
  cuánto pesa hoy el escenario 1 en los números.
- No corrí la suite completa (no se tocó código de producción).
- No revisé `docs/03_design/QuantAgent-nu7-DS-active-position-monitoring.md`, el
  diseño original del monitoreo de posiciones.
