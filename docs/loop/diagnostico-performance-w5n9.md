# Diagnóstico de performance: suite, oráculos y compare (QuantAgent-w5n9)

Medición realizada en la VM del loop (2 CPUs, Linux, PostgreSQL para suite/oráculos).

## 1. Corrida única: backtest run (rsi, SPY, 1d 2007-2018)
Comando: `python -m quantagent.cli backtest run --strategy rsi --snapshot etf-1d-2026-10 --symbol SPY --costos etf --tamano-posicion 1.0`

### Tiempos según base de datos (reproducidos en la VM)
- **PostgreSQL** (`quantagent_loop_test`, base de tests del loop):
  - Reloj: 26.97 s | CPU usuario: 14.74 s | Sistema: 1.18 s | CPU: 59% | Trades: 187, PnL: -9141.21.
- **SQLite en disco** (`/tmp/test_disk.db` en ext4, comportamiento actual de `_isolated_sqlite_db()`):
  - Reloj: 53.02 s | CPU usuario: 15.39 s | Sistema: 1.44 s | CPU: 31% | Trades: 187, PnL: -9141.21.
  - Métricas `/usr/bin/time -v`: 19,170 context switches voluntarios, 368,272 fs outputs. Pasa ~36 s bloqueado esperando al disco por cada commit con fsync.
- **SQLite en `/dev/shm`** (`/dev/shm/test_shm.db` en tmpfs / RAM):
  - Reloj: 13.28 s | CPU usuario: 12.36 s | Sistema: 0.95 s | CPU: 100% | Trades: 187, PnL: -9141.21.
  - Métricas `/usr/bin/time -v`: 39 context switches voluntarios, 0 fs outputs. Ahorro de ~40 s por corrida (~75% de reducción de reloj).

### Perfil cProfile (15 funciones con mayor tiempo acumulado, medido sobre PostgreSQL)
1. `backtest.py:run`: 29.81 s
2. `backtest.py:_analyze_and_trade` (3020 llamadas): 29.64 s
3. `sqlalchemy...state_changes:_go`: 10.52 s
4. `sqlalchemy...session:commit` (3048 llamadas): 9.81 s
5. `sqlalchemy...base:execute` (12837 llamadas): 6.99 s
6. `backtest.py:_get_history_df` (3020 llamadas): 6.70 s
7. `sqlalchemy...query:_iter`: 6.56 s
8. `psycopg2.connection.commit` (2861 llamadas): 6.44 s
9. `psycopg2.cursor.execute` (15702 llamadas): 5.47 s
10. `data/provider.py:get_ohlc` (3020 llamadas): 5.10 s
11. `position_monitor.py:update_candle_tracking` (1736 llamadas): 4.80 s
12. `rsi_strategy.py:generate_signal`: 3.90 s
13. `order_manager.py:execute_decision`: 2.85 s
14. `paper_broker.py:execute_order`: 1.95 s
15. `buy_and_hold.py:worst_day`: 0.35 s

**Desglose:** En PostgreSQL, DB toma 7.71 s directos y >16 s acumulados (commits por vela y queries). En SQLite sobre disco, esos mismos commits generan fsyncs bloqueantes que elevan el tiempo a 53 s. En SQLite sobre `/dev/shm`, el cuello de botella de I/O desaparece y todo corre en 13.28 s a 100% CPU.

## 2. Duraciones de la suite y de los oráculos

### Tabla --durations=30 de la suite rápida (651.56 s total, 988 passed)
```
127.50s call   tests/test_backtest_cli.py::test_correr_par_devuelve_los_numeros_del_resumen_con_las_claves_documentadas
 76.85s call   tests/test_recalc_metrics.py::test_new_cli_lines_equal_recalc_on_the_equity_out_of_rsi_spy_90d
 31.02s call   tests/test_compare_control.py::test_compare_control_coincide_y_falla_si_se_altera_un_trade
 21.40s call   tests/test_backtest_cli_snapshot.py::test_compare_snapshot_arma_csv_e_imprime_tabla
 15.84s call   tests/test_compare_control.py::test_compare_control_con_params_valida_contra_las_mismas_estrategias
 15.64s setup  tests/test_backtest_cli.py::test_backtest_verify_success_exits_zero_and_prints_ok_reproducible
 14.42s call   tests/test_backtest_cli.py::test_backtest_run_triple_screen_trades_on_4h_fixture
 13.25s call   tests/test_backtest_cli_costos.py::test_diferencia_entre_perfiles_coincide_con_recalculo
 12.36s call   tests/test_backtest_cli_snapshot.py::test_compare_snapshot_markdown_escribe_metadatos_y_tabla
 11.41s call   tests/test_backtest_cli_snapshot.py::test_verify_snapshot_imprime_ok_reproducible
 10.29s call   tests/test_backtest_cli.py::test_backtest_verify_process_output_is_only_ok_reproducible
  9.94s call   tests/test_backtest_cli_costos.py::test_verify_con_costos_es_reproducible
  8.87s call   tests/test_backtest_cli.py::test_backtest_verify_usa_el_mismo_tamano_y_regla_en_las_dos_corridas
  8.83s call   tests/test_backtest_cli_snapshot.py::test_compare_params_dos_archivos_dan_dos_filas_y_el_markdown_muestra_los_usados
  8.55s call   tests/test_backtest_cli_costos.py::test_sin_costos_la_salida_es_la_de_antes
  8.52s setup  tests/test_crossval_rsi.py::test_script_exits_zero_and_writes_columns
  7.81s call   tests/test_recalc_metrics.py::test_buy_and_hold_matches_the_cli_line_on_spy_smoke
  7.64s call   tests/test_compare_control.py::test_compare_control_sensibilidad_apertura_siguiente_gap
  7.43s call   tests/test_backtest_cli.py::test_backtest_run_verbose_shows_insufficient_data_messages
  7.21s call   tests/test_backtest_cli.py::test_backtest_run_output_with_custom_slippage_env
  7.19s setup  tests/test_backtest_cli.py::test_backtest_run_output_is_only_metrics_lines
  7.08s call   tests/test_backtest_cli.py::test_backtest_run_consecutive_runs_on_same_database_produce_identical_output
  6.88s call   tests/test_backtest_cli_snapshot.py::test_run_params_quedan_en_el_config_guardado_y_sin_params_son_los_defaults
  6.88s call   tests/test_backtest_cli.py::test_backtest_run_commission_pct_charges_every_fill_and_lowers_pnl
  6.14s call   tests/test_backtest_cli.py::test_backtest_verify_detects_difference_and_exits_nonzero
  4.97s call   tests/test_backtest_cli_snapshot.py::test_to_en_prueba_sin_flag_sale_con_1_y_con_flag_registra
  4.97s call   tests/test_backtest_cli.py::test_backtest_default_llm_strategy_still_requires_openai_key
  4.82s setup  tests/test_backtest_cli.py::test_backtest_run_writes_csv_matching_reported_trade_count
  4.79s call   tests/test_momentum_12m_strategy.py::test_entry_only_on_last_session_of_month_with_positive_return
  4.68s call   tests/test_backtest_cli_costos.py::test_perfil_fija_costos_y_pnl_coincide_con_recalculo[etf]
```

### Tabla --durations=30 de los oráculos (679.94 s total, 22 passed)
```
290.83s setup  tests/test_crossval_compare.py::test_comparison_passes_on_unmodified_engines
119.37s call   tests/test_crossval_snapshot.py::test_crossval_snapshot_reports_comparison_and_double_touches
 87.64s call   tests/test_bt_adapter.py::test_motor_y_adaptador_coinciden_con_slippage[rsi-spy-90d-None-expected0]
 78.58s call   tests/test_golden_rsi_spy_90d.py::test_golden_rsi_spy_90d_metrics_and_trades
 66.63s call   tests/test_golden_snapshot.py::test_motor_igual_al_golden_y_al_adaptador
 17.74s call   tests/test_bt_adapter.py::test_motor_y_adaptador_coinciden_con_slippage[sma-cross-spy-2y-1d-1.0-expected1]
  7.88s call   tests/test_strategy_registration.py::test_estrategia_registrada_con_una_linea_corre_en_cli_y_adaptador
  4.69s setup  tests/test_bt_adapter.py::test_entrada_al_cierre_y_stop_tocado_dentro_de_la_vela
  4.19s call   tests/test_crossval_snapshot.py::test_crossval_rsi_snapshot_runs
  0.01s setup  tests/test_golden_rsi_spy_90d.py::test_golden_rsi_spy_90d_metrics_and_trades
(20 restantes <0.005s)
```

### Causa de los 10 tests más lentos
1. `test_crossval_compare.py` (setup 290.83 s): fixture corre 5 veces `crossval_compare` completo en subprocesos secuenciales (227 y 340 trades).
2. `test_backtest_cli.py::test_correr_par...` (127.50 s): corre `correr_par` sobre 2160 velas de `spy-90d` persistiendo en Postgres.
3. `test_crossval_snapshot.py::test_crossval_snapshot_reports...` (119.37 s): backtest completo de 3020 velas comparando contra backtesting.py.
4. `test_bt_adapter.py::test_motor_y_adaptador_coinciden_con_slippage[rsi...]` (87.64 s): backtest completo motor vs librería sobre 90d.
5. `test_golden_rsi_spy_90d.py::test_golden_rsi_spy_90d_metrics_and_trades` (78.58 s): backtest completo sobre spy-90d contra Postgres.
6. `test_recalc_metrics.py::test_new_cli_lines_equal_recalc...` (76.85 s): subproceso CLI que corre backtest completo sobre spy-90d.
7. `test_golden_snapshot.py::test_motor_igual_al_golden_y_al_adaptador` (66.63 s): backtest completo sobre snapshot real contra Postgres.
8. `test_compare_control.py::test_compare_control_coincide...` (31.02 s): corre `compare` y 3 subprocesos de `compare_control.py`.
9. `test_backtest_cli_snapshot.py::test_compare_snapshot_arma_csv...` (21.40 s): ejecuta `compare` CLI sobre snapshot.
10. `test_bt_adapter.py::test_motor_y_adaptador_coinciden_con_slippage[sma-cross...]` (17.74 s): doble corrida sobre 2 años diarios.

## 3. Oráculos con `-n 2`
- Resultado real: Falló con código 5 en 7.51 s (0 tests ejecutados).
- Causa: `tests/conftest.py:40` (`_configure_xdist_worker_database`) lanza `pytest.UsageError: pytest-xdist parallel execution is only supported with SQLite. Current DATABASE_URL is not SQLite. Run tests sequentially without -n.`

## 4. Concurrencia y estado compartido
- `compare`: Cada par usa `_isolated_sqlite_db()` y lee archivos Parquet de solo lectura. En procesos independientes (`ProcessPoolExecutor`) no comparten estado mutable y pueden correr en paralelo sin pisarse. Con hilos colisionan en variables de módulo `database._engine` y `settings.DATABASE_URL`.
- Oráculos: No pueden correr en paralelo sobre Postgres porque colisionan en las tablas de la base de datos única y `conftest.py` lo bloquea. Para paralelizarse requieren bases SQLite independientes por worker o bases Postgres dedicadas (`_gw0`, `_gw1`).

## 5. Comando exacto para la notebook (hipótesis Trellix)
Para contrastar el impacto del antivirus en el acceso a disco vs memoria, correr ambos en la notebook:

1. **Sobre SQLite en disco:**
```bash
rm -f /tmp/test_disk.db && python -c "import os; os.environ['DATABASE_URL']='sqlite:////tmp/test_disk.db'; from quantagent import database; database.init_db()" && DATABASE_URL=sqlite:////tmp/test_disk.db /usr/bin/time -v python -m quantagent.cli backtest run --strategy rsi --snapshot etf-1d-2026-10 --symbol SPY --costos etf --tamano-posicion 1.0
```

2. **Sobre SQLite en `/dev/shm`:**
```bash
rm -f /dev/shm/test_shm.db && python -c "import os; os.environ['DATABASE_URL']='sqlite:////dev/shm/test_shm.db'; from quantagent import database; database.init_db()" && DATABASE_URL=sqlite:////dev/shm/test_shm.db /usr/bin/time -v python -m quantagent.cli backtest run --strategy rsi --snapshot etf-1d-2026-10 --symbol SPY --costos etf --tamano-posicion 1.0
```

Si en la notebook el tiempo de usuario se mantiene en ~12-15 s pero en disco el tiempo de reloj supera ampliamente los 50 s con alto context switching voluntario, mientras que en `/dev/shm` baja a ~13 s con 100% CPU, se confirma que el antivirus demora las sincronizaciones a disco (`fsync`).

## 6. Propuesta de tickets (ordenados por ahorro real sobre compare, cada uno ≤100 líneas)

1. **Evitar fsync a disco en `_isolated_sqlite_db()`** (`quantagent/cli/backtest.py`):
   - **Qué hacer:** Configurar SQLite para evitar sincronización a disco mediante `PRAGMA synchronous=OFF` (vía connect hook de SQLAlchemy o connect_args) y crear la base en `tempfile.TemporaryDirectory(dir="/dev/shm" if os.path.isdir("/dev/shm") and os.access("/dev/shm", os.W_OK) else None)`.
   - **Por qué `PRAGMA synchronous=OFF`:** Es universal e independiente del sistema operativo (funciona en Linux, WSL, macOS y Windows nativo). Le indica a SQLite no bloquear en cada `commit` esperando `fsync` del filesystem. Dado que `_isolated_sqlite_db()` es una base efímera descartada al finalizar cada par o verificación, la durabilidad ante cortes de energía es prescindible.
   - **Qué pasa si `/dev/shm` no existe:** Si `/dev/shm` no existe (ej. en macOS, Windows o contenedores sin shm), `tempfile` usa el directorio temporal del sistema; gracias a `synchronous=OFF`, la base en disco no emite fsyncs y rinde prácticamente igual que en memoria.
   - **Ahorro estimado:** ~40 s por par (de 53 s a ~13.3 s). En los 48 pares de `compare`, el tiempo baja de ~42.4 min a ~10.6 min. **Ahorro real: ~32 a 35 minutos sobre compare (~75%)**. (~15 líneas).

2. **Paralelizar pares de `compare` con `ProcessPoolExecutor`** (`quantagent/cli/backtest.py`):
   - Sobre la nueva base de ~13.3 s por par conseguida en el Ticket 1:
   - En la VM (2 cores a 100% CPU): reduce los 10.6 min a ~5.5-6 min (**ahorro adicional: ~5 min**).
   - En la notebook (4-6 cores): reduce los 10.6 min a ~2-3 min (**ahorro adicional: ~7-8 min**). (~40 líneas).

3. **Diferir `db.commit()` por vela en `PositionMonitor.update_candle_tracking`**:
   - Evitar `commit()` por cada una de las 1736 llamadas por par; comitear únicamente al cierre de posición o al finalizar el backtest.
   - Ahorro estimado: sobre la base en memoria/shm, ahorra ~2 s de CPU por par = **~1.5 min sobre compare**. (~20 líneas).

4. **Lectura directa de snapshot en `_get_history_df`** (evitar queries repetidas a `MarketData`):
   - Reutilizar el DataFrame ya cargado en memoria en vez de consultar y convertir 3020 filas por par.
   - Ahorro estimado: ~1.5 s de CPU por par = **~1 min sobre compare**. (~35 líneas).

5. **Compartir corrida base en fixture de `test_crossval_compare.py`**:
   - Evitar re-ejecutar 5 veces el motor completo en subprocesos en la suite de oráculos.
   - Ahorro estimado: **~200 s en oráculos**. (~40 líneas).

6. **Reutilizar corrida en fixture para `test_correr_par_devuelve_los_numeros...`**:
   - Validar claves del resumen con un fixture sintético chico en vez de correr 2160 velas contra Postgres.
   - Ahorro estimado: **~100 s en suite rápida**. (~25 líneas).
