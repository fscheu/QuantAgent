# Diagnóstico de Performance de la Suite y Cuellos de Botella (QuantAgent-vfd)

## 1. Medición Real de la Suite

- **Comando exacto**: `python -m pytest -q -m "not slow and not api" --durations=40 --durations-min=1.0`
- **Tiempo total**: 980.21s (16m 20s) · 867 passed, 21 skipped, 81 deselected, 80.207 warnings.
- **Entorno**: VM 4 CPU (Linux x86_64), Python 3.12.3.

### Tabla de duraciones agrupada por archivo (slowest 40)

| Archivo / Grupo | Duración (s) | % del Total | Detalle principal |
|---|---:|---:|---|
| `tests/test_backtest_cli.py` | 322.25s | 32.88% | 17 tests CLI (triple-screen 4h: 106s, verify 3 strats: 59s, subprocesos) |
| `tests/test_crossval_compare.py` | 307.18s | 31.34% | Setup de fixture `runs`: subproceso con 5 pasadas completas de crossval |
| `tests/test_backtest_run_isolation.py` | 186.41s | 19.02% | `test_trades_isolated_across_consecutive_backtest_runs`: 2 corridas de 90d |
| `tests/test_golden_rsi_spy_90d.py` | 84.27s | 8.60% | CLI en subproceso corriendo backtest completo 90d (2.160 velas) |
| `tests/test_triple_screen_strategy.py` | 10.80s | 1.10% | `test_backtest_reference_profile_completes`: 732 velas 4h completas |
| `tests/test_crossval_rsi.py` | 8.81s | 0.90% | Setup subproceso de crossval RSI |
| Otros en top 40 (`sqlite`, `stale`, `parallel`) | 9.85s | 1.00% | Tests individuales entre 1.8s y 4.0s |
| **Top 5 acumulado** | **910.91s** | **92.94%** | Concentran casi la totalidad del tiempo de la suite |

---

## 2. Causas de los 5 Grupos Más Lentos

1. **`tests/test_crossval_compare.py:38` (`runs`) — 307.18s**:
   - *Causa*: Ejecuta un subproceso con `DRIVER` que corre `crossval_compare.main()` 5 veces (dos con `--intrabar` sobre 90 días, 340 trades). Es un oráculo exhaustivo de paridad entre motores que no debería correr en cada ciclo de feedback rápido.
2. **`tests/test_backtest_cli.py` (L121, L345, L98, L251-305) — 322.25s**:
   - *Causa*: `test_backtest_run_triple_screen_trades_on_4h_fixture` (L121, 105.97s) corre 1 año de datos 4h (~2.190 velas); `test_backtest_verify_all_three_strategies` (L345, 59.12s) corre 2 pasadas por estrategia (6 backtests); múltiples tests invocan el CLI en subprocesos independientes (`_run_cli_process`) para verificar flags.
3. **`tests/test_backtest_run_isolation.py:403` (`test_trades_isolated...`) — 186.41s**:
   - *Causa*: Ejecuta 2 backtests completos consecutivos de 90 días horarios (2.160 velas cada uno) solo para comprobar que `Trade.backtest_run_id` queda persistido y aislado en la DB. Un fixture de 5 a 10 días generaría los trades requeridos en <2s.
4. **`tests/test_golden_rsi_spy_90d.py:45` (`test_golden_rsi...`) — 84.27s**:
   - *Causa*: Subproceso CLI que corre el backtest de 90d completo para congelar números golden. Es un test de regresión golden que pertenece a validación pre-merge o marcador `slow`.
5. **`tests/test_triple_screen_strategy.py:380` (`test_backtest_reference...`) — 10.80s**:
   - *Causa*: Simulación de 732 velas 4h completa solo para asertar que `total_pnl` es finito; innecesariamente largo para un test funcional unitario.
*Causa transversal (Warnings)*: 80.207 `DeprecationWarning` de `datetime.utcnow()` en Python 3.12. `pytest.ini` tiene `--disable-warnings` (oculta el reporte final pero no evita la captura en memoria) e ignora `pyproject.toml` (donde sí estaba `filterwarnings`).

---

## 3. Propuesta de Mejoras Ordenadas por Ahorro Estimado

| Ticket propuesto | Mejora | Ahorro est. | Criterio de aceptación binario |
|---|---|---:|---|
| **P1** | Marcar oráculos (`test_crossval_compare.py`, `test_golden_rsi_spy_90d.py`) como `slow` | ~390s | `python -m pytest -q -m "not slow and not api"` deselecciona ambos tests; tiempo suite baja de ~980s a <600s |
| **P2** | Reducir fixtures en tests unitarios (`test_backtest_run_isolation.py`, `test_triple_screen_strategy.py`) | ~190s | `test_trades_isolated...` corre en ≤5s usando fixture sintético reducido; mantiene 0 failed |
| **P3** | Optimizar tests de CLI (`test_backtest_cli.py`): usar fixtures smoke y evitar backtests duplicados | ~120s | Tests de CLI tardan ≤45s en total (ahorro ~200s vs 322s actuales) sin perder aserciones |
| **P4** | Pytest-xdist (`pytest -n auto` en VM / CI) | ~150s | `pytest -n 4` corre suite en 0 failed; tiempo restante se reduce al 40-50% |
| **P5** | Silenciar warnings de `datetime.utcnow` en `pytest.ini` / migrar a `timezone.utc` | ~30s | Suite reporta 0 warnings (o <100) en vez de 80.207; menor overhead de memoria y loop |

*Evaluación de opciones*:
- **Fixtures módulo/sesión**: Útiles en `test_backtest_cli.py` para compartir resultados de corridas base y verificar flags de formato sin recomputar trades.
- **pytest-xdist**: Altamente viable en la VM de 4 CPUs. Con SQLite aislado por `tmp_path`, la concurrencia es limpia y reduce a la mitad el tiempo.
- **Marcador `slow`**: Clave para separar la auditoría golden/crossval (que corre antes de mergear lotes) del loop de desarrollo iterativo.
- **Silenciar warnings**: Añadir `filterwarnings = ["ignore::DeprecationWarning"]` a `pytest.ini` elimina el costo de crear 80.000 objetos de warning.

---

## 4. Objetivo de Tiempo y Factibilidad

- **Objetivo propuesto para la suite del loop**: **≤180s (3 minutos)**.
- **Factibilidad**: Completamente alcanzable sin hardware adicional. La suma de P1 + P2 + P3 reduce el tiempo secuencial a ~280s; aplicando P4 (`pytest-xdist -n 4`), el tiempo total cae a ~90-120s.
