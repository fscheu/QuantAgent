# QuantAgent-c03q — Reconciliación de cartera dedicada

Delta del [runbook de paper](QuantAgent-av1p.5-IM-runbook-paper-ops.md), chequeo de pesos contra sombra: el digest con `--portfolio` usa `reconciliar_cartera` en `quantagent/fabrica/chequeos_paper.py`. El objetivo es `peso_sombra * (1 - PAPER_CASH_RESERVE_PCT)`. El assembler comparte esa constante de reserva del 5 %.

El peso efectivo es `(cantidad_posición + compras_pendientes - ventas_pendientes) * precio / equity`; cada pendiente aporta solamente `quantity - filled_quantity`. Los precios de Alpaca para posiciones prevalecen sobre `Position.current_price`; los demás precios disponibles salen de la DB dedicada. Si falta el precio de una cantidad efectiva, se informa una alerta explícita. La tolerancia continúa siendo una acción (`precio / equity`), con el fallback existente para símbolos presentes solo en sombra sin precio disponible.

No cambia el rebalanceador ni la comparación directa con `--archivo`. Los tests puros cubren compras, ventas, ejecuciones parciales, acumulación de órdenes, activos sin posición y activos fuera de sombra. El test del digest usa SQLite y un cliente Alpaca simulado: DBC con 212 acciones más 10 pendientes contra 222,5 de objetivo pasa; con 9 pendientes alerta.

Validación:

```bash
NUMBA_DISABLE_JIT=1 /home/azureuser/repos/projects/QuantAgent/.venv/bin/python -m pytest tests/test_trading_digest.py tests -k "digest or reconcil or chequeos" -q
```

Resultado: 91 passed, 2 skipped. Este worktree tiene `DATABASE_URL` vacía: se ejecutó el comando con `DATABASE_URL=sqlite:////tmp/quantagent-c03q-tests.db`. Sin esa variable, seis casos de `test_reconcile_cli.py` fallan al interpretar la URL; se reprodujo el mismo error cargando el módulo original de `HEAD`. `scripts/loop/smoke.sh` terminó con exit 0 usando el Python del mismo virtualenv.
