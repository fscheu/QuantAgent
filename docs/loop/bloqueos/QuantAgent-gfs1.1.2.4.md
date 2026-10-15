# QuantAgent-gfs1.1.2.4 — bloqueo por contrato del test de sensibilidad

Se intentó recalcular largos y cortos al ejecutar en apertura, guardando confianza
y fracción en la orden. El diff intentado sumó 34 líneas en los dos archivos permitidos.
Tests dirigidos: **22 passed, 1 failed in 224.15s**.
Falla: `test_compare_control_sensibilidad_apertura_siguiente_gap`, línea 253:
`462.70 != 500.65 ± 0.01`. La fórmula `gap * qty_cierre` supone igual cantidad
en ambos llenados; deja de aplicar cuando la apertura dimensiona su propia cantidad.
Los tests de largo y corto en apertura pasaron también en la revisión independiente.

Decisión necesaria: habilitar `tests/test_compare_control.py` en este ticket y
reemplazar esa fórmula por el PnL calculado a mano con cantidades distintas.
No cambiar tolerancias ni congelar 462.70 como nueva referencia.
El ticket queda in_progress. Se retiró el código; esta entrega solo documenta el bloqueo.
No hay rama WIP: el aislamiento de esta corrida prohíbe escribir Git y hacer push.

Reproducción mínima: aplica el cálculo intentado, ejecuta el test real y restaura
el archivo en `finally`. Esperado: el test falla con los valores anteriores y el
reproductor sale 0. Ejecutar solo en el worktree temporal de `try.sh`.

```python
# REPRO_START
from pathlib import Path
import subprocess
p = Path("scripts/bt_adapter.py")
original = p.read_text()
changed = original.replace("for order in entries:\n", "for order in entries:\n                    confidence, fraction = order.tag\n                    equity = broker._cash + sum(t.size * (price - t.entry_price) for t in broker.trades)\n                    units = round(equity * fraction * confidence / price)\n                    order._replace(size=units if order.is_long else -units)\n")
changed = changed.replace("tp=take / SCALE)", "tp=take / SCALE, tag=(signal.confidence, settings.TRADING_BASE_POSITION_PCT))")
try:
    p.write_text(changed)
    r = subprocess.run(["python", "-m", "pytest", "-q", "tests/test_compare_control.py::test_compare_control_sensibilidad_apertura_siguiente_gap"], capture_output=True, text=True, timeout=180)
    print(r.stdout)
    assert r.returncode == 1 and "462.7" in r.stdout and "500.65" in r.stdout, r.stderr
finally:
    p.write_text(original)
# REPRO_END
```
