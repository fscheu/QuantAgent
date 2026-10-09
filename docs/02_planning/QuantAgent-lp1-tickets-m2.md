# QuantAgent-lp1 — Tickets de M2, lotes 1 y 2 (listos para BEADS)

Plan: [`QuantAgent-lp1-PL-m2-validacion-datos-reales.md`](./QuantAgent-lp1-PL-m2-validacion-datos-reales.md) §10–§13.
Fecha: 2026-10-09. `V01`…`V20` son marcadores: el PM los reemplaza por IDs al cargarlos (labels `plan-continuacion`, `m2`).
Solo se cargan si Fede aprueba las "decisiones previas", que son las de §13 del plan.

Reglas comunes:
- Objetivo ≤100 líneas de diff (`PLAN-CONTINUACION.md` §3.2). Si el agente estima más, entrega partición.
- Probar = `~/repos/projects/QuantAgent/scripts/loop/try.sh loop/<ID> -- '<comando>'`. Los marcados **(VM)** necesitan
  `QUANTAGENT_SNAPSHOT_DIR` exportada y el snapshot `etf-1d-2026-10` (decisión previa 1).
- Ningún test usa red. Los que leen el snapshot real llevan `@pytest.mark.snapshot` y se saltean con motivo si no existe.
- En los lotes 1 a 3 ningún comando lee después del 2018-12-31 (decisión previa 3).
- Referencias de M1, que no cambian salvo que el ticket lo diga: rsi/spy-90d 340 / 4571.27; fifty-two-week-high/spy-2y-1d
  6 / −249.35; triple-screen/spy-1y-4h 6 / 174.34 (`docs/loop/REVIEW-LOG.md`, línea de #60).
- La entrega trae una comprobación que falla si se rompe la lógica. El PM la repite: la prueba propia del agente no alcanza.

Orden y dependencias (la cola toma la primera fila elegible):
- Lote 1: V01 → V02 → V03 y V04. V05 → V06. V07 (necesita V01) → V08 → V09 (necesita V06). V10 necesita V06 y V07.
  El PM crea `etf-1d-2026-10` en la VM apenas se mergea V02, y antes de V04, porque V04 lo necesita.
- Lote 2: V11 → V12 → V13 → V14. V15 → V16. V17 y V15 → V18 → V19. V20 va último: congela lo decidido en V13 y V18.
- Filas `decidir` (V10, V13, V17, V18): el PM escala y el lote sigue con las filas que no dependen de esa respuesta
  (`PLAN-CONTINUACION.md` §3.7).

## Lote 1 · `lote/datos-reales-diarios`

### V01 · Snapshot: Parquet con manifiesto
- **Contexto:** M2 corre sobre datos reales congelados fuera del repo, con un manifiesto adentro (`docs/04_decisions/QuantAgent-iuf-DC-datos-m2.md` D5, opción C). Hoy solo hay fixtures CSV (`quantagent/backtesting/fixtures.py`).
- **Cambio requerido:** `quantagent/data/snapshot.py` con `write_snapshot`, `read_snapshot` y `verify_snapshot`: un Parquet por símbolo en `$QUANTAGENT_SNAPSHOT_DIR/<nombre>/` y `manifest.json` con símbolo, fuente, timeframe, rango, filas, sha256 por archivo, fecha de bajada y versión de `yfinance`. `verify_snapshot` compara contra el manifiesto versionado en `tests/fixtures/snapshots/<nombre>.json` si existe. Declarar `pyarrow` en `pyproject.toml`.
- **Criterio de aceptación:** `python -m pytest -q tests/test_snapshot.py` pasa, con un test que escribe y relee un DataFrame idéntico y otro que cambia un byte de un Parquet y hace fallar `verify_snapshot` nombrando ese archivo.
- **Archivos relevantes:** `quantagent/data/snapshot.py` (nuevo), `tests/test_snapshot.py` (nuevo), `pyproject.toml`.
- **Fuera de scope:** bajar datos, CLI, cargar en la base.
- **Revisión de Fede:** leer.

### V02 · `data snapshot create` y `verify` desde Yahoo
- **Contexto:** Yahoo es la fuente diaria (iuf D4) y ya se usa en `quantagent/data/provider.py`. iuf D6: el snapshot se congela y nunca se vuelve a bajar con el mismo nombre, porque el ajuste por dividendos reescribe la historia.
- **Cambio requerido:** grupo `data` del CLI (`quantagent/cli/data.py`, registrado en `quantagent/cli/__main__.py`) con `snapshot create --name --symbols --start` (baja con `yfinance`, `auto_adjust=False`, guarda OHLC, `Adj Close` y volumen tal cual, escribe con V01; si el nombre existe, exit 1 sin tocar nada) y `snapshot verify --name` (imprime `OK <n> símbolos, <filas> filas` o el archivo que no coincide).
- **Criterio de aceptación:** tests con la descarga reemplazada por un DataFrame fijo: crea, `verify` da OK, un segundo `create` con el mismo nombre sale con 1. (VM, lo corre el PM una vez) `python -m quantagent.cli data snapshot create --name etf-1d-2026-10 --symbols SPY,IWM,EFA,EEM,TLT,SHY,GLD,DBC,XLK,XLE,XLF,XLU --start 2007-01-03 && python -m quantagent.cli data snapshot verify --name etf-1d-2026-10` imprime `OK 12 símbolos` (iuf midió 59.676 filas al 2026-10-08). El PM commitea el manifiesto en `tests/fixtures/snapshots/`.
- **Archivos relevantes:** `quantagent/cli/data.py` (nuevo), `quantagent/cli/__main__.py`, `tests/test_cli_data_snapshot.py` (nuevo).
- **Fuera de scope:** ajuste de precios (V03), calidad (V04), 4h, otras fuentes, commitear datos.
- **Revisión de Fede:** probar.

### V03 · Precios ajustados por splits y dividendos
- **Contexto:** sin ajuste, un split parece una caída y se pierden los dividendos de TLT y SHY (iuf D6). V02 guarda `Close` y `Adj Close` sin tocar.
- **Cambio requerido:** `read_snapshot` devuelve por defecto el OHLC multiplicado por `Adj Close / Close` de cada fila y conserva `close_sin_ajustar`. Fórmula en el docstring. El volumen queda como viene.
- **Criterio de aceptación:** `python -m pytest -q tests/test_snapshot.py -k ajuste` pasa con una serie armada a mano con un split 2:1 y un dividendo: el cierre ajustado no salta en la fecha del split y `close_sin_ajustar` sí. Sin el ajuste, el test falla.
- **Archivos relevantes:** `quantagent/data/snapshot.py`, `tests/test_snapshot.py`.
- **Fuera de scope:** volver a bajar datos, cambiar el provider de Yahoo.
- **Revisión de Fede:** leer.

### V04 · `data snapshot check`: reporte de calidad
- **Contexto:** iuf D6 exige comparar contra el calendario NYSE. En `main` (PR #58) iuf midió 0 sesiones faltantes en 15 ETFs y de 1 a 29 filas por ETF con máximo o mínimo fuera de rango (SPY: 29, hasta 0,06%). `pandas_market_calendars` ya es dependencia.
- **Cambio requerido:** `data snapshot check --name` imprime por símbolo sesiones faltantes, días de más, filas con OHLC incoherente, volumen 0 y saltos de más de 20% sin split. Exit 1 solo si faltan sesiones.
- **Criterio de aceptación:** un test borra un día de un snapshot armado a mano y el reporte nombra esa fecha. (VM) `python -m quantagent.cli data snapshot check --name etf-1d-2026-10` imprime `faltan 0` en los 12 símbolos.
- **Archivos relevantes:** `quantagent/cli/data.py`, `quantagent/data/snapshot.py`, `tests/test_snapshot_quality.py` (nuevo).
- **Fuera de scope:** corregir datos, intradía (lote 5).
- **Revisión de Fede:** probar.

### V05 · Fixture con huecos y `--fixture` en el control con backtesting.py
- **Contexto:** el engine avanza por una grilla de calendario: cada 1 h, 4 h o 1 día desde la fecha inicial (`quantagent/backtesting/backtest.py:706-732`). No sigue las velas del dato, y el CLI apaga el filtro de horario (`quantagent/cli/backtest.py:149 y :233`). Los tres fixtures de M1 tienen velas en sábado y domingo (624, 148 y 624), así que nunca se probó con huecos (AC de `QuantAgent-832`, ítem 8). Leyendo el código, sin correrlo: un sábado el engine vuelve a evaluar la vela del viernes y puede entrar a un precio al que no se puede operar.
- **Cambio requerido:** `tests/fixtures/spy-90d-habiles.csv` = `spy-90d.csv` sin sábados ni domingos (1536 filas), con el comando que lo genera en el docstring del test. Opción `--fixture` (default `spy-90d`) en `scripts/crossval_rsi.py` y `scripts/crossval_compare.py`. Si el port necesita replicar la ventana de lookback del engine con huecos, se ajusta acá. `crossval_compare.py` imprime `Entradas en fin de semana: N`.
- **Criterio de aceptación:** `python scripts/crossval_compare.py --fixture spy-90d-habiles --intrabar` imprime `TODO DENTRO DE TOLERANCIA` o la lista de trades que difieren, más la línea de fin de semana. `python scripts/crossval_compare.py --intrabar` sigue en `TODO DENTRO DE TOLERANCIA`.
- **Archivos relevantes:** `scripts/crossval_rsi.py`, `scripts/crossval_compare.py`, `tests/test_crossval_compare.py`, `tests/fixtures/spy-90d-habiles.csv`.
- **Fuera de scope:** cambiar el engine (V06).
- **Revisión de Fede:** probar. Es el diagnóstico de V06.

### V06 · El engine avanza por las velas del dato
- **Contexto:** V05. Decisión previa 4.
- **Cambio requerido:** `_get_date_range_for_asset` (`backtest.py:734`) devuelve los timestamps cargados en `market_data` para ese activo y timeframe dentro del rango, en vez de la grilla de `_get_date_range`. Test nuevo con huecos.
- **Criterio de aceptación:** `python scripts/crossval_compare.py --fixture spy-90d-habiles --intrabar` imprime `TODO DENTRO DE TOLERANCIA` y `Entradas en fin de semana: 0`. Los tres números de referencia no cambian: `for s in rsi:spy-90d fifty-two-week-high:spy-2y-1d triple-screen:spy-1y-4h; do python -m quantagent.cli backtest run --strategy ${s%%:*} --fixture ${s##*:} | grep -E "Trades|PnL"; done`.
- **Archivos relevantes:** `quantagent/backtesting/backtest.py`, `tests/test_intrabar_stops.py` o un test nuevo del engine.
- **Fuera de scope:** filtro de horario, zona horaria (`QuantAgent-col`, lote 5), `_bars_to_calendar_days`.
- **Revisión de Fede:** probar. Si cambia un número de referencia, el PM escala.

### V07 · `backtest run --snapshot`
- **Contexto:** `backtest run` solo lee `tests/fixtures/` (`quantagent/cli/backtest.py:79`).
- **Cambio requerido:** opciones `--snapshot`, `--symbol`, `--from` y `--to`, excluyentes con `--fixture`. Carga las velas ajustadas del rango en `market_data` como `load_fixture` (`quantagent/backtesting/fixtures.py:57`), con el mismo chequeo de base sucia, y corre con `offline_data`.
- **Criterio de aceptación:** test con un snapshot armado en `tmp_path`. (VM) `python -m quantagent.cli backtest run --strategy rsi --snapshot etf-1d-2026-10 --symbol SPY --from 2007-01-03 --to 2018-12-31` termina con exit 0 y 6 líneas. `python -m pytest -q tests/test_golden_rsi_spy_90d.py` sigue verde.
- **Archivos relevantes:** `quantagent/cli/backtest.py`, `quantagent/backtesting/fixtures.py`, `tests/test_backtest_cli.py`.
- **Fuera de scope:** candado (V08), `verify` (V09), varios símbolos por corrida.
- **Revisión de Fede:** probar.

### V08 · Candado de la reserva
- **Contexto:** la reserva empieza el 2023-01-01 (decisión previa 3). Ninguna corrida la lee antes de V54 y hoy nada lo impide.
- **Cambio requerido:** con `--snapshot`, sin `--to` el rango termina el 2022-12-31. Un `--to` posterior sale con exit 1 y un mensaje con la fecha, salvo `--abrir-reserva`. Cada apertura agrega fecha y comando a `$QUANTAGENT_SNAPSHOT_DIR/reserva-accesos.log`. La fecha se toma de `QUANTAGENT_RESERVA_DESDE` (default `2023-01-01`).
- **Criterio de aceptación:** tests: `--to 2023-06-30` sin el flag sale con 1 y el log no cambia; con el flag corre y el log suma una línea. (VM) `python -m quantagent.cli backtest run --strategy rsi --snapshot etf-1d-2026-10 --symbol SPY --to 2024-01-01; echo $?` imprime el mensaje y `1`.
- **Archivos relevantes:** `quantagent/cli/backtest.py`, `tests/test_backtest_cli.py`.
- **Fuera de scope:** scripts `crossval_*` (llevan `--to` explícito), candado del tramo de prueba (es convención).
- **Revisión de Fede:** probar.

### V09 · `backtest verify --snapshot` con las tres estrategias
- **Contexto:** `verify` (`quantagent/cli/backtest.py:250`) corre dos veces y compara, solo con fixtures.
- **Cambio requerido:** `verify` acepta las opciones de V07 y respeta V08. El PR informa `Trades:`, `Total PnL:` y el tiempo de cada estrategia sobre SPY 2007–2018.
- **Criterio de aceptación:** (VM) `for s in rsi fifty-two-week-high triple-screen; do time python -m quantagent.cli backtest verify --strategy $s --snapshot etf-1d-2026-10 --symbol SPY --from 2007-01-03 --to 2018-12-31; done` imprime tres veces `OK reproducible`. Si una estrategia hace 0 trades, se informa y el PM escala: no se fuerza.
- **Archivos relevantes:** `quantagent/cli/backtest.py`, `tests/test_backtest_cli.py`.
- **Fuera de scope:** cambiar estrategias o parámetros.
- **Revisión de Fede:** probar.

### V10 · Control con backtesting.py sobre SPY diario real
- **Contexto:** la coincidencia trade por trade se probó solo con RSI, sobre un fixture 24/7 de volumen constante y sin comisiones. En spy-90d ninguna vela toca stop loss y take profit a la vez, así que el desempate (gana stop loss) solo lo cubren tests unitarios (REVIEW-LOG, #59). backtesting.py es el control permanente en tests (ADR, opción B).
- **Cambio requerido:** `--snapshot`, `--symbol`, `--from` y `--to` en `crossval_rsi.py` y `crossval_compare.py` (el engine por `backtest run --snapshot`). La salida suma `Velas que tocan stop y take profit: N`. Registrar el marker `snapshot` en `pyproject.toml` y agregar un test que se saltea sin snapshot.
- **Criterio de aceptación:** (VM) `python scripts/crossval_compare.py --snapshot etf-1d-2026-10 --symbol SPY --from 2007-01-03 --to 2018-12-31 --intrabar` imprime `TODO DENTRO DE TOLERANCIA` o cada trade que difiere con su causa, y la línea de velas con los dos niveles.
- **Archivos relevantes:** `scripts/crossval_rsi.py`, `scripts/crossval_compare.py`, `tests/test_crossval_compare.py`, `pyproject.toml`.
- **Fuera de scope:** cambiar el engine o la regla de desempate; otras estrategias (V29).
- **Revisión de Fede:** decidir. Si hay diferencias o velas con doble toque, Fede elige qué regla manda.

## Lote 2 · `lote/costos-y-referencia`

### V11 · Comisiones en el backtest y en el CLI
- **Contexto:** `PaperBroker` calcula comisión `none | fixed | pct` (`quantagent/trading/paper_broker.py:52-54`) y está apagada. El backtest no la configura: solo pasa `slippage_pct` (`backtest.py:250` y `:689`, `strategy/assembler.py:63`). Ninguna evidencia de M1 incluye comisiones.
- **Cambio requerido:** `TRADING_COMMISSION_PCT` en `settings.py` (default 0) y `--commission-pct` en `backtest run` y `verify`. Si el valor es mayor que 0, se pasa al broker con `commission_model="pct"`. El CLI imprime siempre `Comisión: X% por lado` después del slippage, así que los tests que esperan 6 líneas pasan a 7.
- **Criterio de aceptación:** con el default no cambia ningún número de referencia y el golden sigue verde. `python -m quantagent.cli backtest run --strategy rsi --fixture spy-90d --commission-pct 0.001 | grep -E "PnL|Comisión"` imprime un `Total PnL` menor que 4571.27 y `Comisión: 0.10% por lado`.
- **Archivos relevantes:** `quantagent/settings.py`, `quantagent/backtesting/backtest.py`, `quantagent/cli/backtest.py`, `tests/test_backtest_cli.py`.
- **Fuera de scope:** comisión fija, perfiles (V14), `recalc_metrics.py` (V12), camino de paper.
- **Revisión de Fede:** probar. Con default 0 no cambia ningún número. Los valores se deciden en V13.

### V12 · `recalc_metrics.py` con comisión
- **Contexto:** el PnL de un trade cerrado descuenta solo la comisión de la orden de cierre (`quantagent/portfolio/manager.py:155-178`, leído en el código, sin probar). La de entrada queda en otra fila. `docs/03_design/backtest_metrics.md` §1.1 dice "comisión 0".
- **Cambio requerido:** `--commission-pct` en `scripts/recalc_metrics.py`, que recalcula el PnL con comisión de entrada y de salida. Actualizar §1.1 con la fórmula que use el engine.
- **Criterio de aceptación:** `python -m quantagent.cli backtest run --strategy rsi --fixture spy-90d --commission-pct 0.001 --out /tmp/c.csv && python scripts/recalc_metrics.py /tmp/c.csv --commission-pct 0.001 | grep "PnL por trade"` imprime `N/N filas coinciden`, o el PR muestra que no coinciden porque falta la comisión de entrada. En ese caso el PM escala: es un bug de plata.
- **Archivos relevantes:** `scripts/recalc_metrics.py`, `tests/test_recalc_metrics.py`, `docs/03_design/backtest_metrics.md`.
- **Fuera de scope:** corregir el engine, importar `quantagent`.
- **Revisión de Fede:** probar.

### V13 · Decisión: perfiles de costos
- **Contexto:** hoy el slippage es 0,05% por lado (`settings.py:75`, T08c) y la comisión 0. Con costos irreales, una estrategia que opera mucho parece mejor o peor que comprar y mantener sin serlo.
- **Cambio requerido:** doc `docs/04_decisions/QuantAgent-<id>-DC-costos-m2.md`, de 40 líneas como máximo: 2 o 3 perfiles (al menos ETF y cripto) con comisión y slippage por lado, la fuente de cada número ("no verificado" si no abre) y el efecto de cada perfil sobre rsi/SPY 2007–2018, medido con V11.
- **Criterio de aceptación:** el doc tiene la tabla con fuentes y el comando de cada fila de efecto. El PM reproduce una fila.
- **Archivos relevantes:** `docs/04_decisions/` (nuevo y su `README.md`).
- **Fuera de scope:** código.
- **Revisión de Fede:** decidir los valores.

### V14 · `--costos <perfil>`
- **Contexto:** V13.
- **Cambio requerido:** perfiles en `settings.py`; `--costos etf|cripto` en `run` y `verify` fija slippage y comisión, y el CLI suma la línea `Costos: <perfil>`.
- **Criterio de aceptación:** la diferencia de `Total PnL` de `backtest run --strategy rsi --fixture spy-90d` entre `--costos etf` y `--costos cripto` coincide con la de `recalc_metrics.py` con los mismos valores (±0.01). Sin `--costos` sigue 340 / 4571.27.
- **Archivos relevantes:** `quantagent/settings.py`, `quantagent/cli/backtest.py`, `tests/test_backtest_cli.py`.
- **Fuera de scope:** elegir valores, perfil automático por símbolo.
- **Revisión de Fede:** probar.

### V15 · Referencia comprar y mantener en el CLI
- **Contexto:** M2 compara cada estrategia contra comprar y mantener el mismo activo, en la misma ventana y con los mismos costos (§3 del plan). Hoy no existe.
- **Cambio requerido:** función de 40 líneas como máximo que compra con todo el capital al cierre de la primera vela y vende al cierre de la última, con los costos de la corrida. El CLI imprime `Comprar y mantener: PnL X, Sharpe Y, max drawdown Z`, con el N de `backtest_metrics.md` §5.2.
- **Criterio de aceptación:** un test con 3 velas armadas a mano compara contra el PnL calculado en el propio test. (VM) `python -m quantagent.cli backtest run --strategy rsi --snapshot etf-1d-2026-10 --symbol SPY --to 2018-12-31` imprime la línea.
- **Archivos relevantes:** `quantagent/backtesting/` (función nueva), `quantagent/cli/backtest.py`, `tests/test_buy_and_hold.py` (nuevo).
- **Fuera de scope:** métricas de exceso (lote 3), varios activos.
- **Revisión de Fede:** probar.

### V16 · Recálculo independiente de la referencia
- **Contexto:** cada número del CLI tiene un recálculo que no importa `quantagent` (lote L1).
- **Cambio requerido:** `recalc_metrics.py --comprar-y-mantener <csv o parquet de velas>`, con `--slippage-pct` y `--commission-pct`.
- **Criterio de aceptación:** `python scripts/recalc_metrics.py --comprar-y-mantener tests/fixtures/spy-90d.csv --slippage-pct 0.0005` imprime PnL, Sharpe y max drawdown iguales (±0.01) a la línea de V15 sobre spy-90d.
- **Archivos relevantes:** `scripts/recalc_metrics.py`, `tests/test_recalc_metrics.py`.
- **Fuera de scope:** importar `quantagent`.
- **Revisión de Fede:** probar.

### V17 · `QuantAgent-cub`: ¿el límite de pérdida diaria actúa en backtest?
- **Contexto:** el ticket ya existe en BEADS. `get_daily_pnl` filtra por `date.today()` (`quantagent/portfolio/manager.py:343`, `quantagent/trading/risk_manager.py:209`), así que en un backtest sobre fechas pasadas el límite (5%, `settings.py:73`) no se dispara. Con datos reales y posiciones más grandes (V18), sí importa.
- **Cambio requerido:** un test que reproduce un backtest con una pérdida en el día mayor al límite. Sin corregir.
- **Criterio de aceptación:** `python -m pytest -q tests/test_daily_loss_backtest.py` da `1 xfailed` (estricto) si confirma el bug. El PR responde "¿el límite diario actúa en backtest?" con la evidencia.
- **Archivos relevantes:** `quantagent/trading/risk_manager.py`, `quantagent/portfolio/manager.py`, `tests/test_daily_loss_backtest.py` (nuevo).
- **Fuera de scope:** la corrección (V21, si V18 la decide).
- **Revisión de Fede:** decidir.

### V18 · Decisión: tamaño de posición y límite diario en las comparaciones
- **Contexto:** cada trade usa 5% del capital por confianza (`settings.py:72`, tope 10% en `:74`). Contra comprar y mantener con todo el capital, una estrategia con 95% en efectivo pierde retorno y gana drawdown por construcción.
- **Cambio requerido:** doc en `docs/04_decisions/`, de 40 líneas como máximo, con tres opciones: A, 5% y la referencia escalada; B, 100% sin apalancamiento en las corridas de M2; C, otra que proponga el agente. Tiene que explicar cómo lo hace la práctica profesional (capital asignado por estrategia, tamaño por riesgo o volatilidad), con fuente o "no verificado". Inclinación de Fede (2026-10-09): que todas operen con 5%, o con 100% del capital asignado a cada estrategia si el 5% se aplica al nivel de la asignación. Lleva una corrida de rsi/SPY 2007–2018 por opción (PnL, Sharpe, max drawdown, exposición media y referencia) y la regla del límite diario según V17.
- **Criterio de aceptación:** el doc trae la tabla y los comandos. El PM reproduce una fila.
- **Archivos relevantes:** `docs/04_decisions/` (nuevo).
- **Fuera de scope:** código.
- **Revisión de Fede:** decidir.

### V19 · Aplicar el tamaño decidido
- **Contexto:** V18. Escrito para la opción B. Si Fede elige otra, el PM reescribe esta fila antes de lanzarla.
- **Cambio requerido:** `--tamano-posicion <fracción>` en `run` y `verify` fija `base_position_pct` y `max_position_pct` de la corrida. El CLI imprime `Tamaño: X% por trade`.
- **Criterio de aceptación:** un test del CLI con `--tamano-posicion 1.0` verifica `qty = equity × 1.0 × confianza / precio` en 3 trades. Sin la opción no cambia ningún número de referencia.
- **Archivos relevantes:** `quantagent/cli/backtest.py`, `tests/test_backtest_cli.py`.
- **Fuera de scope:** defaults de paper, apalancamiento.
- **Revisión de Fede:** probar.

### V20 · Golden sobre el snapshot real
- **Contexto:** `tests/test_golden_rsi_spy_90d.py` congela M1. El mismo control sobre datos reales no puede correr en CI, porque los datos no se publican (iuf D5).
- **Cambio requerido:** `tests/test_golden_snapshot.py`, marcado `snapshot`: rsi/SPY 2007–2018 con perfil ETF y el tamaño de V19, comparado contra un CSV guardado junto al snapshot, fuera del repo.
- **Criterio de aceptación:** (VM) `python -m pytest -q -m snapshot tests/test_golden_snapshot.py` pasa y falla si se cambia un dígito del CSV guardado. En CI aparece como skipped con el motivo.
- **Archivos relevantes:** `tests/test_golden_snapshot.py` (nuevo).
- **Fuera de scope:** datos o trades reales en el repo.
- **Revisión de Fede:** leer.
