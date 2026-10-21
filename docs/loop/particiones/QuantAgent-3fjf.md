# Partición de QuantAgent-3fjf

La entrega completa se estima por encima del objetivo de 100 líneas porque elimina
una regla de salida duplicada, adapta el piloto y migra tests del método eliminado.
Se divide en dos cambios revisables:

| Hijo | Objetivo | Criterio binario | Revisión |
|---|---|---|---|
| `QuantAgent-3fjf.1` | Scheduler delega en `strategy.should_exit`; el adaptador del piloto usa la regla base | Casos 1 y 3 de `tests/test_exit_rule_divergence.py` coinciden con backtest y el método duplicado no existe | decidir |
| `QuantAgent-3fjf.2` | Migrar las pruebas del scheduler que llamaban al método eliminado | `tests/trading/test_scheduler_position_monitor.py` no referencia el método y pasa con la delegación | leer |

Primero se implementa U3a y después U3b. Ambos conservan el alcance original de
U3: no cambian backtest, salida intravela ni cierre de posición.
