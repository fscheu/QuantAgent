# Diagnóstico de performance: suite, oráculos y compare (QuantAgent-w5n9)

Medición realizada en la VM del loop (2 CPUs, Linux, PostgreSQL para suite/oráculos).

## 1. Corrida única: backtest run (rsi, SPY, 1d 2007-2018)
Comando: `python -m quantagent.cli backtest run --strategy rsi --snapshot etf-1d-2026-10 --symbol SPY --costos etf --tamano-posicion 1.0`
- Reloj: 26.97 s | CPU usuario: 14.74 s | Sistema: 1.18 s | CPU: 59% | Trades: 187, PnL: -9141.21.

### Perfil cProfile (15 funciones con mayor tiempo acumulado)
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

**Desglose:** Base de datos (SQLAlchemy / psycopg2): 7.71 s directos (tottime), >16 s acumulados (commits por vela y queries). Indicadores / estrategia: 6.87 s directos. Bucle del motor sin DB: 0.79 s directos.

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
`/usr/bin/time -v python -m quantagent.cli backtest run --strategy rsi --snapshot etf-1d-2026-10 --symbol SPY --costos etf --tamano-posicion 1.0`
Si en la notebook el tiempo de usuario es similar (~15 s) pero el tiempo transcurrido (wall clock) supera ampliamente los 40 s con elevado context switching o I/O wait, se confirma la demora por Trellix.

## 6. Propuesta de tickets (ordenados por ahorro sobre compare, cada uno ≤100 líneas)
1. **Paralelizar pares de `compare` con `ProcessPoolExecutor`** en `quantagent/cli/backtest.py`: cada par corre en un proceso worker con su SQLite aislado. Ahorro estimado: ~22 min en VM (2 cores, ~48%) y ~30-33 min en notebook (4-6 cores, ~70%). (~40 líneas).
2. **Eliminar `db.commit()` por vela en `PositionMonitor.update_candle_tracking`**: diferir el commit al cierre de posición o fin de corrida en backtest. Ahorro estimado: ~9.5 s por par = ~7.6 min sobre compare secuencial (16% de compare). (~20 líneas).
3. **Lectura directa en memoria de velas de snapshot en `_get_history_df`**: evitar 3020 consultas `SELECT` a `MarketData` y conversiones fila a fila cuando los datos provienen de snapshot. Ahorro estimado: ~5 s por par = ~4 min sobre compare. (~35 líneas).
4. **Compartir corrida base en fixture de `test_crossval_compare.py`**: evitar re-ejecutar 5 veces el motor completo en subprocesos. Ahorro estimado: ~200 s en oráculos. (~40 líneas).
5. **Reutilizar corrida en fixture para `test_correr_par_devuelve_los_numeros...`**: validar claves con fixture reducido o corrida compartida en vez de 2160 velas contra Postgres. Ahorro estimado: ~100 s en suite rápida. (~25 líneas).
