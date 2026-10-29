# PLAN-POST-M5

Plan de los hitos que vienen después del motor top-down (M5). Una pantalla por hito. El detalle, las opciones y la revisión crítica están en [`docs/02_planning/2026-10-28_roadmap_post_m5.md`](docs/02_planning/2026-10-28_roadmap_post_m5.md); este archivo solo lleva objetivo, punto de control, decisiones tomadas y cola. `PLAN-CONTINUACION.md` no se toca: queda para L20, L21 y M5 hasta que cierren.

Marco (Fede, 2026-10-28): M5 es el motor top-down. El dinero real no es un hito con número: es la salida de M-ciclos + M-cuenta, con autorización explícita de Fede y ganándole a comprar y mantener SPY con USD 2.500. Ningún ticket se abre en Beads sin aprobación de Fede, salvo M-manual. Los nombres son por tema, no M6, M7.

Orden: M-manual ya y en paralelo; M-datos (decisión) antes de L22; M-web (solo diseño) en paralelo con M5; M-ciclos después de M5; M-cuenta adelantado sobre M4 y M5; M-web (construcción) cuando existan M-datos y el registro único del embudo.

## M-manual (hecho 2026-10-29)

- **Objetivo:** un manual en inglés, en Markdown, para contestar en un minuto qué pide cada etapa, qué criterio rige y qué comando usar.
- **Punto de control:** `docs/manual/` tiene índice, criterios, decisiones, glosario, embudo, operación y referencia de comandos generada desde `click`; el test de cobertura de la CLI pasa; Streamlit y `docs/user-manual/` ya no están en el repo.
- **Decidido:** arrancar ya; borrar el manual viejo; inglés; se lee en GitHub y la web lo renderiza después; toda la ayuda de la CLI a inglés (U1–U4, W1).
- **Cola:** lote L32 `lote/manual`, PR #615 mergeado a `main` el 2026-10-29; padre `QuantAgent-dpr3` cerrado. El manual vive en `docs/manual/` y se revisa al cerrar cada lote (regla en `AGENTS.md`).

## M-datos

- **Objetivo:** decidir y construir dónde viven los datos de mercado y fundamentales.
- **Punto de control:** ADR escrito, catálogo con un archivo por dataset y su test, congelado mensual, política de retención.
- **Decidido:** Parquet para lo histórico, DuckDB para consultar, registros del embudo a Postgres cuando la web lo necesite; congelado mensual (D1–D3).
- **Cola:** sin tickets. La decisión y el catálogo van antes de que los datos de acciones entren al repo (L22).

## M-web

- **Objetivo:** aplicación web propia (cuentas paper y live, embudo, datos de mercado, después asistente con IA) que reemplaza Streamlit.
- **Punto de control:** primero pantallas diseñadas con Fede y documento de arquitectura; después API de solo lectura, Cuenta y Embudo.
- **Decidido:** Streamlit no convive y se borra (en MAN-01); FastAPI + React; diseño visual en sesiones con Fede (W1–W2). Falta acordar cuándo arrancan (W3).
- **Cola:** sin tickets. Construcción cuando existan M-datos y el registro único del embudo.

## M-ciclos

- **Objetivo:** que la generación de estrategias sea un ciclo repetible con reglas fijas, cuyas sobrevivientes maduran en paper hasta ser candidatas.
- **Punto de control:** escalera de incubación y registro único de estado, después calendario automático e informe por ciclo.
- **Decidido:** C1–C4 (cadencia, cupo por ciclo, tiempo mínimo en paper y uso de la reserva), ver §1 del roadmap.
- **Cola:** sin tickets. Después de M5, con el generador de L20 y paper andando.

## M-cuenta

- **Objetivo:** probar que una candidata funciona con la cuenta real de USD 2.500 antes de poner dinero.
- **Punto de control:** modo cuenta chica del backtest, prueba de sensibilidad a USD 1.000, 2.500 y 5.000, puerta escrita y cuenta paper al capital real.
- **Decidido:** capital USD 2.500; cuenta cash, sin cortos en la primera cuenta; el costo fiscal lo revisa Fede con una contadora y entra al modo cuenta chica (K1–K3).
- **Cola:** sin tickets. Se adelanta: sobre M4 apenas exista el modo, y sobre M5 antes de decidir el pago de Sharadar (L25).
