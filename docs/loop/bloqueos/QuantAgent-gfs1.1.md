# F03a: tests dirigidos fallan antes de implementar

Se estimaron 170 líneas y se preparó una partición en dos hijos (95 / 85).
No se modificó código. La partición no se entrega: se retiró el cambio de cola
y los hijos quedan fuera-loop hasta que el PM resuelva este bloqueo.

Comando del ticket, sin excluir marcadores:
`python -m pytest -q tests/test_backtest*.py`
Resultado real: 8 failed, 201 passed, 67 warnings in 552.14s; exit 1.
Los ocho fallos están en `tests/test_backtest_integration.py`, marcado `api`.
Todos fallan al entrar en `patch("quantagent.backtesting.backtest.TradingGraph")`:
`AttributeError: module quantagent.backtesting.backtest does not have the attribute TradingGraph`.
Ejemplo mínimo reproducible:
`python -m pytest -q tests/test_backtest_integration.py::TestBacktestIntegration::test_full_backtest_flow_minimal`

El motor actual ya no expone ese atributo. No se corrigieron mocks ni imports:
sería alcance nuevo para una entrega de partición. La suite del CI excluye `api`,
pero el comando explícito del ticket los incluye; no se ignoró ese comando.
Smoke: exit 0, 9 trades / PnL 542.36.
Oráculos: `timeout 10m python -m pytest -q -m oraculo` exit 124 al 76%,
en `tests/test_crossval_snapshot.py`; sin fallos reportados antes del timeout.
`try.sh` exacto falla antes de ejecutar: `.git/FETCH_HEAD: Read-only file system`;
el wrapper es quien puede publicar la rama.

Decidir: A) corregir estos ocho tests en un ticket previo, o B) autorizar en el
ticket la exclusión de `api` para esta partición. Luego retomar los hijos
QuantAgent-gfs1.1.1 → QuantAgent-gfs1.1.2 y actualizar la cola al integrarlos.
El padre permanece in_progress; gfs1.2 sigue dependiendo del padre.
