# M5-34: partición de H4 hasta etapa 2

Estimación: 300–375 líneas entre preparación de señales, ejecución multi-activo,
riesgo, tests e informe. PLAN-CONTINUACION.md §3.2 exige partir antes de implementar.
Esta entrega no implementa H4 ni produce veredicto o intentos.

| Hijo | Objetivo y aceptación binaria | Máximo estimado, con tests/docs | Revisión |
|---|---|---:|---|
| QuantAgent-fnpr.4.5.1 | Datos desde el cache H3 y señales A del YAML; tests de elegibilidad semanal y truncamiento pasan | 90 | leer |
| QuantAgent-fnpr.4.5.2 | Órdenes stop con vencimiento y gestión A; tests de entrada habilitada y persistencia tras rebalanceo pasan | 95 | leer |
| QuantAgent-fnpr.4.5.3 | Curva conjunta D2 y costos etf; tests de sizing, equity y rechazo por heat pasan | 100 | leer |
| QuantAgent-fnpr.4.5.4 | Corrida 2011–2018 contra H3, V30, un intento y el informe solicitado con todas las métricas | 90 | decidir |

Orden: .1 → .2 → .3 → .4; padre `in_progress` hasta integrar y revisar el cierre.
Los contratos completos y tests dirigidos están en los hijos de Beads.
Las primeras tres entregas exponen piezas invocables sin abrir una corrida pública;
el entrypoint y el único registro real se conectan en la cuarta.

Se conserva el contrato del comentario del 2026-10-31: A solamente, entradas
nuevas solo en el conjunto semanal H3, sin entradas en risk-off, posiciones
abiertas siguen con A al salir del conjunto y no cierran por rebalanceo.
D2: riesgo 0,25%, tope 10%, heat 2%, capital normalizado y acciones fraccionarias.
Mismo perfil etf de H3; sin parámetros nuevos ni cambios en `quantagent/trading/`.
Etapa 3 espera M5-35 y no se abre la reserva 2023+.

Reutilización verificada: `scripts/m5/h3.py` expone pesos y curva;
`/tmp/ai-loop/h3-cache.pkl` contiene candidatos, paneles y precios (76 MB).
`bar_signals` conserva stop de entrada, vencimiento, stop inicial y Chandelier;
`check_stop_entry` resuelve los gaps. No recalcular candidatos ni la clasificación
de supervivencia de H3 para esta comparación.
