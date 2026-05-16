# Run report

## Veredicto

`PARTIAL`

## Qué se hizo

1. Se revisó la documentación fuente del ticket (`requirements`, `design`, `acceptance`, `implementation`) y el manual `paper-trading-automation.md`.
2. Se verificó liveness local del deploy en `http://127.0.0.1:8501/_stcore/health`.
3. Se abrió la UI real en browser y se recorrieron manualmente las pestañas relevantes para paper runtime:
   - `Dashboard`
   - `Paper Trading`
   - `Orders & Positions`
   - `Logs`
4. Se revisó la consola del browser.
5. Se guardó evidencia durable en este sobre, incluyendo screenshots.

## Qué quedó demostrado

- El deploy de Streamlit responde localmente y la UI carga.
- El selector de entorno visible es `paper`.
- No hubo errores JavaScript en el browser.
- La UI expone claramente un estado degradado/no inicializado del runtime paper:
  - `Status: 🔴 Stopped`
  - `No scheduler heartbeat found`
- La visibilidad de datos operativos está bloqueada por errores SQL de tablas inexistentes:
  - `trades`
  - `orders`
  - `logs`

## Qué no se pudo validar

- heartbeat reciente y verificable del scheduler (AC1 positivo)
- flujo order -> trade -> active position para LONG/SHORT (AC2)
- ausencia de contaminación por HOLD (AC3)
- caso `runtime sano sin trades` mostrando vacío graceful sin falsa caída (AC4 en su variante sana)

## Lectura técnica corta

El deploy/browser está arriba, pero el entorno QA observable no está listo para certificar el hardening del paper runtime. Lo que falla no es el front-end sino la base observable del runtime: heartbeat ausente y persistencia incompleta o schema no inicializado.

## Recomendación inmediata

Antes de rerun de QA validator para cerrar `QuantAgent-sft`, conviene asegurar en QA:
- scheduler paper arrancado efectivamente
- heartbeat reciente persistido
- schema/tablas necesarias disponibles (`trades`, `orders`, `logs`, y la superficie requerida para active positions)
- al menos un escenario verificable para distinguir `sin actividad pero sano` vs `caído`

## Artifacts del run

- `input-envelope.md`
- `docs-check.md`
- `browser-findings.md`
- `result.json`
- `console.log`
- `screenshots/dashboard.png`
- `screenshots/paper-trading.png`
