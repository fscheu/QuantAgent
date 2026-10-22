# Bloqueo: QuantAgent-qr6

Intenté implementar el selector `paper_local | alpaca_paper`, el guardrail de URL y el registro de constructores. Los cuatro tests nuevos pasaron y los tests de ensamblador pasaron.

La verificación dirigida completa no quedó verde: `tests/trading/test_scheduler_position_monitor_integration.py` falla de forma reproducible en tres casos de stop loss y take profit, porque `position.is_active` permanece en `True`. El archivo volvió a fallar al correrlo aislado (3 failed, 3 passed). No modifiqué el scheduler.

Decisión que destraba: reparar o excluir explícitamente esos tres casos de la lista dirigida para que se pueda verificar el ticket sin fallos ajenos al selector. El smoke todavía no se corrió porque la entrega no alcanzó el gate de tests verdes.
