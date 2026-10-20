# QuantAgent-w12l / vbnh: limpieza de intentos (2026-10-19)

Copia exclusiva creada y verificada byte a byte antes de escribir: `local-data/intentos-20261019-previo-a-vbnh.csv`.
SHA256 original: `3e521684ea9f26c06e307c7fa5f9ff71bdf7f0c4205060885e886b612016811b`.
SHA256 limpio: `624918165e0c2156c4d38da44538fb1a7149474a1c9c2255587539114c08d4e0`.
109 → 99 filas de datos; se retiraron solo las 10 con `snapshot=sintetico`, sin reserializar las demás.
DSR etapa 2 de `cartera-transversal`: 0,0 → 0,5735043131699945; mismo cálculo y curvas, solo cambia el registro.
`diff -u <(awk -F, '$4 != "sintetico"' "$QUANTAGENT_SNAPSHOT_DIR/intentos-20261019-previo-a-vbnh.csv") "$QUANTAGENT_SNAPSHOT_DIR/intentos.csv"`: sin salida, exit 0. `grep -c sintetico`: 0 (exit 1 esperado). Reproducción de solo lectura con 46 curvas de etapa 2 guardadas (sin nuevos backtests ni registros):

```python
import csv, hashlib
from pathlib import Path
import pandas as pd
from quantagent.data.snapshot import snapshot_root, read_snapshot
from quantagent.fabrica.hipotesis import cargar_hipotesis
from quantagent.fabrica.cartera import curva_cartera
from quantagent.backtesting.verdict import dsr_contra_intentos
root = snapshot_root(); old = root / 'intentos-20261019-previo-a-vbnh.csv'; new = root / 'intentos.csv'
assert hashlib.sha256(old.read_bytes()).hexdigest() == '3e521684ea9f26c06e307c7fa5f9ff71bdf7f0c4205060885e886b612016811b'
assert hashlib.sha256(new.read_bytes()).hexdigest() == '624918165e0c2156c4d38da44538fb1a7149474a1c9c2255587539114c08d4e0'
lines = old.read_bytes().splitlines(keepends=True); rows = list(csv.DictReader(old.read_text().splitlines()))
assert new.read_bytes() == lines[0] + b''.join(l for l, r in zip(lines[1:], rows) if r['snapshot'] != 'sintetico')
h = cargar_hipotesis('hipotesis/cartera-transversal.yaml'); returns = {}
for i, p in enumerate(h.piernas):
    for a in p.activos:
        df = pd.read_csv(f'/tmp/corrida_transversal/equity/{p.clase}-{a}.csv', index_col='date', parse_dates=True); assert df.index.min().year == 2007 and df.index.max().year == 2018; returns[i, a] = df.equity.pct_change().dropna()
closes = pd.DataFrame({a: read_snapshot('etf-1d-2026-10-amplio', a)['close'].loc['2007-01-01':'2018-12-31'] for p in h.piernas for a in p.activos})
eq = curva_cartera(closes, {i: list(p.activos) for i, p in enumerate(h.piernas)}, returns)
print('DSR antes:', dsr_contra_intentos(eq, old), 'después:', dsr_contra_intentos(eq, new))
```
