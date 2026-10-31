# H4 contra H3: etapa 2 (M5-34)

Corrida del 2026-10-31; 2011-01-03 a 2018-12-31, 2.012 sesiones idénticas para ambas curvas.
Reproducir: `python scripts/m5/h4.py --cache /tmp/ai-loop/h3-cache.pkl --sin-registro`.
Cache H3 original, sin recalcular candidatos: SHA256 `162b95e3e20e689f527d7adf6e4df20b48d5a2f25bcd68156072b7dbce73f76e`.
Costo `etf`: 0,05% por lado sobre nocional, comisión cero; capital inicial 100.000. Preparación en cuatro procesos; calentamiento de `required_history_bars`, con ventanas A idénticas a L24.

| Curva | Sharpe tasa 0 | CAGR | maxDD | PnL neto |
| --- | ---: | ---: | ---: | ---: |
| H4 | -0,129886 | -0,561669% | 10,165606% | -4.401,58 |
| H3 (curva de `scripts/m5/h3.py`) | 0,645269 | 9,055095% | 20,271043% | — |

ΔSharpe H4−H3: **−0,775156 ± 0,303409**, un error estándar bootstrap pareado (`error_estandar_delta_sharpe`).
H4: 335 trades cerrados, 1 posición abierta al final; sesiones invertidas 89,7614%; heat medio sobre todas las sesiones 0,2854%; rechazos por heat: 0.
Rotación bruta anual H4/H3: 6,396224 / 33,927688; costos acumulados H4/H3: 2.550,45 / 21.677,07 unidades del capital normalizado.
**V30: CORTA**, por `pnl <= 0` y `sharpe_tasa_0 < 0.10`; trades y maxDD cumplen. No habilita etapa 3.
Registro local (2026-10-31 07:08:20 UTC): `m5-h4`, etapa 2, una fila en `hipotesis.csv` y exactamente un intento en `intentos.csv`; corrida inicial sin `--sin-registro`, reproducciones con esa opción.
Limitaciones: hereda cobertura del cache H3; curva conjunta propia, sin control del motor; entradas por orden de columnas y heat sin liberar intradiario. Posiciones finales valuadas al cierre, sin liquidación forzada.
No se abrieron etapa 3, precios 2019+ ni reserva 2023+; tampoco se modificó `quantagent/trading/`.
Pruebas: `python -m pytest -q tests/test_m5_h4.py tests/test_arquetipo_a.py`: 16 passed, incluye identidad de bytes de ambos registros; `scripts/loop/smoke.sh`: exit 0; `ruff check --no-fix scripts/m5/h4.py tests/test_m5_h4.py`: PASS.
