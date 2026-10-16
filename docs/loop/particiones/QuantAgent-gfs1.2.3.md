# QuantAgent-gfs1.2.3: alcance de los oráculos de llenado

Inventario documental; no modifica código, umbrales ni números esperados.
Continúa [la partición del padre](QuantAgent-gfs1.2.md).

## Inventario completo

Los seis archivos con marca `oraculo` son los siguientes:

| Archivo en tests/ | Referencia actual | Propuesta para el PM y justificación |
|---|---|---|
| test_golden_rsi_spy_90d.py | Apertura default, intravela; 227 trades / 13532.99; CSV congelado byte a byte | Conservar: ya migrado por gfs1.2.1. |
| test_golden_snapshot.py | Apertura default y explícita: 181 / -21841.01; cierre explícito: 187 / -9141.21; adaptador y recálculo en ambos | Conservar ambos: apertura verifica el default y cierre protege la opción pública. El supuesto golden se regenera en cada test; el congelado externo sigue pendiente en gfs1.2.4. |
| test_crossval_compare.py | Cierre explícito del motor frente a crossval_rsi: 227 trades sin intravela y 340 con intravela; mutaciones y tolerancias | Conservar cierre: comparación histórica independiente del port RSI, incluidos los stops al cierre. No demuestra apertura ni el default actual. |
| test_crossval_snapshot.py | Cierre: motor/port intravela 187 trades, 3 dobles toques; port sin intravela 153 | Conservar cierre: control histórico con datos reales, diferente del golden apertura. |
| test_bt_adapter.py | Mixto: escenarios a mano en close y next-open; CLI del adaptador default close; comparación con motor pide cierre explícito (340 / 4571.27 y 2 / -6305.04) | Conservar ambos: cubre la API del adaptador, cuyo default sigue siendo close, y las reglas de apertura (gaps, niveles reescalados, costos y liquidación final). |
| test_strategy_registration.py | CLI motor default apertura; adaptador default cierre; exige registro y trades > 0, sin igualdad entre motores | Conservar: prueba extensibilidad, no equivalencia de llenados ni referencia numérica al cierre. |

`scripts/crossval_rsi.py:run_port_stats` fija `trade_on_close=True`;
`scripts/crossval_compare.py:run_engine` pide `--llenado cierre`.
Cambiar solamente el motor a apertura invalidaría la comparación.
El adaptador genérico ya permite `next-open`; no hace falta migrar el port legacy
para controlar apertura: `test_golden_snapshot.py` usa ese adaptador en ambos modos.

## Decisión solicitada y pendientes

Se propone conservar todas las referencias al cierre de la tabla como controles
explícitos. La autorización de Fede del 2026-10-15 las mantiene hasta este inventario;
la ratificación por referencia del PM sigue pendiente de revisión de esta entrega.
No se declara migración completa ni se cierra el padre antes de esa decisión.
No se propone código nuevo ni hijos adicionales: gfs1.2.4 ya cubre el golden externo
congelado y gfs1.2.5 cubre `sweep`/`walkforward` con apertura default y cierre opcional.
Estos dos tickets siguen pendientes; esta entrega no los implementa.

## Verificación

Comando dirigido: `timeout 20m python -m pytest -q -m oraculo`.
En el aislamiento, Numba necesita `NUMBA_CACHE_DIR` en un directorio escribible;
sin esa variable la colección falla en vectorbt con `no locator available`.
Con caché escribible: 34 passed, 1239 deselected en 847,23 s; smoke exit 0
(9 trades / PnL 561,18). Comando reproducible y limitación de try.sh en el PR.
