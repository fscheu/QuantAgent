# Chequeo documental

## Resultado general

La documentación requerida existe y cubre el marco funcional/técnico para la validación M2 de paper runtime hardening. También existe manual operativo de usuario para paper trading automation.

## Matriz rápida

| Documento | Estado | Observación concreta |
|---|---|---|
| `docs/05_acceptance_tests/QuantAgent-sft-AC-paper-runtime-hardening.md` | OK | Define AC1..AC6 sobre heartbeat, flujo order/trade/position, visibilidad en UI y evidencia reproducible. Fue el oracle principal para clasificar el run. |
| `docs/01_requirements/QuantAgent-sft-RQ-paper-runtime-hardening.md` | OK | Aclara que el milestone apunta a runtime estable/verificable en QA y que los checks deben distinguir ausencia de datos vs falla operativa real. Esto fue relevante para no sobrediagnosticar. |
| `docs/03_design/QuantAgent-sft-DS-paper-runtime-hardening.md` | OK | Refuerza los cuatro seams: boot/runtime, heartbeat, execution y visibility. La validación browser se concentró en visibility y señales indirectas de heartbeat/runtime. |
| `docs/06_implementation/QuantAgent-app-IM-qa-streamlit-cutover.md` | OK | Documenta el cutover de QA a Streamlit en `8501`, healthcheck local y contrato de validación local. Coincide con lo observado en health endpoint y UI. |
| `docs/user-manual/paper-trading-automation.md` | OK | Existe. Explica arranque por `python apps/paper_trading.py`, estados del scheduler en dashboard y monitoreo del runtime paper. Fue útil para contrastar el mensaje de la pestaña Paper Trading. |

## Alineación doc -> evidencia observada

- El cutover a `8501` está alineado con el entorno real observado: `/_stcore/health` respondió `ok`.
- La UI efectivamente expone pestañas relevantes para `Dashboard`, `Paper Trading`, `Orders & Positions` y `Logs`, consistente con la documentación operativa.
- La documentación espera distinguir runtime sano sin actividad versus runtime caído/no inicializado. En este run sólo pudo observarse el caso degradado/no inicializado (`No scheduler heartbeat found`), no el caso sano sin actividad.
- El manual habla de visibilidad de órdenes, posiciones y logs. En este deploy la UI intentó consultarlos, pero devolvió errores de esquema ausente (`relation "trades"/"orders"/"logs" does not exist`), por lo que la capacidad documental está pero la base observable no quedó lista para validación integral.

## Conclusión documental

Documentación suficiente para ejecutar la validación manual y producir evidencia reproducible. El resultado incompleto del run no se debe a falta de docs sino a estado del entorno/datos de QA observados en la UI.
