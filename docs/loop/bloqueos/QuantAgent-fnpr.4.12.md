# M5-36: bloqueo del conteo contrafactual de supervivencia

H1 y H3 corrieron 2019–2022 (1.008 sesiones, 208 semanas), costo `etf` idéntico.
**Ambas CORTA. M5-55: NO sumar acciones al paper:** H3 tiene solo 2/4 años positivos.

| Métrica | H1 | H3 |
|---|---:|---:|
| Sharpe | 0,071614 | 0,236142 |
| CAGR | −0,2909% | 2,7208% |
| MaxDD | 37,0196% | 34,6243% |
| Rotación anual | 20,5676 | 29,7032 |
| Costos sobre capital inicial 100.000 | 3.818,71 | 5.819,48 |

ΔSharpe H3−H1 = +0,164528 ± 0,212723 (un error estándar bootstrap pareado).
H1 además corta por Sharpe < 1/N (0,600181) y maxDD > 35%.

| Año | Descartes antes del ranking | % semanas expuestas | Cota de % peso medio |
|---|---:|---:|---:|
| 2019 | 396 | 92,31 | 62,42 |
| 2020 | 495 | 80,77 | 60,98 |
| 2021 | 479 | 100,00 | 79,16 |
| 2022 | 102 | 34,62 | 20,30 |
| Total | 1.472 | 76,92 (160/208) | 55,71 |

Falta el precio necesario para calcular fuerza relativa, score, z-scores y cuartil: no se sabe qué
acciones ausentes habrían entrado al top 3. Las cotas no son el conteo ni el peso efectivo solicitados.
H1 no tiene este sesgo de selección de acciones; la comparación favorece a H3.
La verificación independiente dio FAIL en esos dos criterios. No hay corrección posible con los mismos datos.
Destraba: el PM acepta explícitamente cotas, o autoriza recuperar los precios faltantes y repetir la medición.

Implementación e [informe completo por año y sector](https://github.com/fscheu/QuantAgent/blob/9fc6da22/docs/06_implementation/QuantAgent-fnpr.4.12-IM-etapa3.md)
preservados en `loop-wip/QuantAgent-fnpr.4.12` (commit `9fc6da22`); este PR entrega solo el bloqueo.
Cache nuevo `/tmp/ai-loop/h3-etapa3-2019-2022.pkl`; el cache 2011–2018 conserva su fecha del 30/10.
Se registró un intento de etapa 3 por hipótesis, con prueba abierta y reserva cerrada; repetición idempotente.
WIP validado: 9 tests sin red, ruff `check --no-fix` OK, smoke exit 0 (9 trades), hashes del snapshot 10c OK.
No se modificó `quantagent/trading/` ni se evaluó 2023+. Ticket `in_progress`, pendiente decisión del PM.
