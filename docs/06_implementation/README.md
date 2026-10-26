# Implementation Notes

This folder contains implementation notes, technical details, and post-implementation documentation for specific features and changes.

## Reading Order

Agents working in this repository should read implementation documents relevant to the feature being worked on:

### Current Implementation Notes
- [backtest_data_pipeline_alignment.md](./backtest_data_pipeline_alignment.md) - OHLCV data pipeline alignment and remediation
- [20251130_Implementation_AgentParallelization.md](./20251130_Implementation_AgentParallelization.md) - Agent parallelization implementation details
- [SHORT_POSITIONS_IMPLEMENTATION.md](./SHORT_POSITIONS_IMPLEMENTATION.md) - Short position support implementation
- [IMPLEMENTATION_WEEK5_TRADING_EXECUTION.md](./IMPLEMENTATION_WEEK5_TRADING_EXECUTION.md) - Trading execution layer implementation
- [RISK_MANAGER_DUPLICATION.md](./RISK_MANAGER_DUPLICATION.md) - Risk manager duplication analysis and resolution

## Current Truth

Implementation notes document the actual code changes, technical decisions made during implementation, and any deviations from the original design. When working on related features:

- Review relevant implementation notes to understand what was actually built
- Check for known issues or technical debt mentioned in the notes
- Use these documents to understand implementation patterns and conventions

## Implementation Document Content

Implementation documents should include:

1. **Overview** - What was implemented
2. **Approach** - How it was implemented (key code changes)
3. **Deviations** - Any changes from the original design/plan
4. **Known Issues** - Technical debt or limitations
5. **Testing** - How the implementation was validated

## Active Per-Change Implementation

- [QuantAgent-jpwz-IM-diagnostico-conteo-f1.md](./QuantAgent-jpwz-IM-diagnostico-conteo-f1.md) - F1 tiene cambios de pesos que el conteo por señales omite.
- [QuantAgent-e4jc.3-IM-f1-etapas-0-2.md](./QuantAgent-e4jc.3-IM-f1-etapas-0-2.md) - F1 pasa etapas 0 y 1; en f1c etapa 2 evalúa V30 pero corta con 0 trades por activo, sin apertura de reserva.
- [QuantAgent-46pk-IM-diagnostico-adaptador-cortos.md](./QuantAgent-46pk-IM-diagnostico-adaptador-cortos.md) - Diagnóstico F2 etapa 2 y normalización de brackets cortos en el adaptador.
- [QuantAgent-e4jc.6-IM-f2-etapas-0-2.md](./QuantAgent-e4jc.6-IM-f2-etapas-0-2.md) - F2 `tendencia-larga-corta` por etapas 0 a 2: etapa 1 pasa (0,54), etapa 2 corta; correlación 0,86 con 1/N.
- [QuantAgent-7v45-IM-normalizacion-transversal.md](./QuantAgent-7v45-IM-normalizacion-transversal.md) - Objetivos T4 en consultas futuras y persistencia de sombra con evidencia sintética.
- [QuantAgent-v2ll-IM-atribucion.md](./QuantAgent-v2ll-IM-atribucion.md) - Atribución reproducible en pesos, señal y tasa de cuatro carteras (etapa 2 guardada).
- [QuantAgent-lrf8.3.2-IM-convivencia-fuentes.md](./QuantAgent-lrf8.3.2-IM-convivencia-fuentes.md) - Convivencia de filas `alpaca` con fixture y snapshot en CLI run y verify.
- [QuantAgent-d4cq.75-IM-punto-de-control-capa-macro.md](./QuantAgent-d4cq.75-IM-punto-de-control-capa-macro.md) - Punto de control L14: comparador sobre datos reales y veredicto por capa.
- [QuantAgent-d4cq.68-IM-punto-de-control-catalogo-primas.md](./QuantAgent-d4cq.68-IM-punto-de-control-catalogo-primas.md) - Punto de control L13: carteras de primas de riesgo, catálogo, N total y DSR.
- [QuantAgent-d4cq.47-IM-universo-amplio.md](./QuantAgent-d4cq.47-IM-universo-amplio.md) - Universo amplio de ETFs (2007) para cartera de primas de riesgo (D6).

- [QuantAgent-d4cq.34-IM-punto-de-control-tanda.md](./QuantAgent-d4cq.34-IM-punto-de-control-tanda.md) - Punto de control L11: tanda de hipotesis/ sobre datos reales.
- [QuantAgent-d4cq.31-IM-resumen.md](./QuantAgent-d4cq.31-IM-resumen.md) - Resumen por estado final, causas e intentos del período.
- [QuantAgent-d4cq.28-IM-reapertura.md](./QuantAgent-d4cq.28-IM-reapertura.md) - Reapertura de prueba: sin filtro, intentos ni estados nuevos.
- [QuantAgent-d4cq.15-IM-punto-control-l9.md](./QuantAgent-d4cq.15-IM-punto-control-l9.md) - Dos hipótesis YAML: admisión, filtro, veredictos e intentos aislados en 2007–2018.
- [QuantAgent-d4cq.14.3-IM-filtro-cli.md](./QuantAgent-d4cq.14.3-IM-filtro-cli.md) - RSI en doce ETFs: tabla, veredicto, tiempo e intentos de etapa 1.
- [QuantAgent-gfs1.2.1-IM-golden-apertura.md](./QuantAgent-gfs1.2.1-IM-golden-apertura.md) - Golden RSI/spy-90d con apertura siguiente y comando de regeneración.
- [QuantAgent-app-IM-qa-streamlit-cutover.md](./QuantAgent-app-IM-qa-streamlit-cutover.md) - QA Streamlit cutover, Cloudflare tunnel alignment, and deploy_finished -> qa_verified contract.
- [QuantAgent-uzq-IM-fix-scheduler-heartbeat.md](./QuantAgent-uzq-IM-fix-scheduler-heartbeat.md) - Fix TradingScheduler heartbeat and scheduler unit-test regressions exposed by CI gate (P1, bug)
- [QuantAgent-b8r-IM-52week-high-momentum.md](./QuantAgent-b8r-IM-52week-high-momentum.md) - M1 Strategy 3: FiftyTwoWeekHighStrategy (52-week high momentum/breakout for US equities)
- [QuantAgent-c69-IM-llm-agent-strategy.md](./QuantAgent-c69-IM-llm-agent-strategy.md) - M1 Strategy 2: LLMAgentStrategy reference pipeline (m1, strategy)
- [QuantAgent-94d-IM-backtest-isolation.md](./QuantAgent-94d-IM-backtest-isolation.md) - Backtest Run ID Isolation (P1, bug)

## Per-Change Implementation Documents

Paper trading admite seleccionar un YAML de cartera con `--portfolio`; crea SQLite y archivo de sombra por cartera, y no inicia el ciclo. Credenciales siguen entrando por entorno.

Las carteras con una pierna `curva-pendiente` construyen `FredClient` al preparar el scheduler y lo inyectan en `paper_portfolio`; si falta `FRED_API_KEY`, el arranque falla explícitamente. Las demás carteras no requieren la clave.

`tests/trading/test_scheduler_cartera.py` prueba dos carteras paper aisladas con SQLite y brokers simulados, incluidos fills incrementales y reapertura de sesión.

Per-change implementation documents should follow this naming convention:

```
QuantAgent-<issue-id>-IM-<short-slug>.md
```

Example: `QuantAgent-a3f2dd-IM-fee-calculation.md`

These files describe implementation deltas and technical notes for specific changes linked to Beads issues (when applicable for system implementations).
