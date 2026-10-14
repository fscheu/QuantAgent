# Decisions

This folder contains architectural decision records (ADR-like entries) documenting significant technical choices and their rationale.

## Reading Order

Agents working in this repository should review decision documents chronologically to understand the evolution of technical choices:

### Current Decisions
- [ui_framework_decision.md](./ui_framework_decision.md) - UI framework selection (Streamlit vs alternatives)
- [QuantAgent-iuf-DC-datos-m2.md](./QuantAgent-iuf-DC-datos-m2.md) - Datos reales de M2: ETFs diarios, 2 activos en 4h, fuentes y dónde vive el snapshot (propuesta)
- [2026-10-ADR-backtest-engine-propio-vs-terceros.md](./2026-10-ADR-backtest-engine-propio-vs-terceros.md) - ADR: Engine de backtesting propio vs. alternativas de terceros (QuantAgent-lcv)
- [QuantAgent-ld3-DC-costos-m2.md](./QuantAgent-ld3-DC-costos-m2.md) - Perfiles de costos de backtest para M2 (ETF, ETF conservador, cripto) con fuentes y comandos de efecto medido (propuesta, V13)
- [QuantAgent-80x-DC-motor-oficial-m2.md](./QuantAgent-80x-DC-motor-oficial-m2.md) - Qué motor da el número oficial de M2: cinco estrategias sobre SPY 2007–2018 en el motor propio y en backtesting.py, trade por trade (propuesta, decide Fede)
- [QuantAgent-cbp-DC-tamano-posicion-m2.md](./QuantAgent-cbp-DC-tamano-posicion-m2.md) - Tamaño de posición y límite diario en las comparaciones de M2: 5% con referencia escalada, 100% del capital asignado o tamaño fijo sin confianza, con corridas de rsi/SPY (propuesta, V18)
- [QuantAgent-7nle-DC-datos-para-operar-m3.md](./QuantAgent-7nle-DC-datos-para-operar-m3.md) - Fuente de velas para paper en M3 (Alpaca gratis o Yahoo), proveedor abstracto con dos implementaciones, convivencia con snapshots, ajuste y zona horaria, 9 tickets (decidido por Fede el 2026-10-12)
- [2026-10-12-DC-quantconnect-y-pivot-lean.md](./2026-10-12-DC-quantconnect-y-pivot-lean.md) - QuantConnect y LEAN: comparación con QuantAgent, piezas no contempladas y pivot a LEAN autoalojado (spike `QuantAgent-ryaw` en curso; Fede define el pivot al cierre del lote 4)

## Current Truth

Decision documents capture the context, alternatives considered, and rationale for significant technical choices. When working on related features:

- Review relevant decision docs to understand constraints and trade-offs
- Respect decisions unless there's a compelling reason to revisit them
- Document new significant decisions in this folder

## Decision Document Format

Decision documents should include:

1. **Context** - What problem are we solving?
2. **Options Considered** - What alternatives were evaluated?
3. **Decision** - What was chosen and why?
4. **Consequences** - What are the implications?
5. **Status** - Accepted, superseded, or deprecated

## Active Per-Change Decisions

- [QuantAgent-69d-DC-reuse-logs-for-llm-telemetry.md](./QuantAgent-69d-DC-reuse-logs-for-llm-telemetry.md) - Choose existing `logs` over new metrics tables for this P3 feature
- [QuantAgent-hx0.7-DC-sharpe-annualization.md](./QuantAgent-hx0.7-DC-sharpe-annualization.md) - Sharpe ratio annualization periods for 24/7 fixtures
- [QuantAgent-u8lf-DC-diagnostico-rsi-gld-apertura-siguiente.md](./QuantAgent-u8lf-DC-diagnostico-rsi-gld-apertura-siguiente.md) - Diagnóstico: rsi/GLD 243 trades al cierre vs 242 en apertura siguiente (V27c)
- [QuantAgent-yo4s-DC-umbrales-y-llenado.md](./QuantAgent-yo4s-DC-umbrales-y-llenado.md) - Decisión: umbrales del veredicto, llenado, tamaño mínimo y costo de cortos antes de abrir 2019–2022 (V30)


## Per-Change Decision Documents

Per-change decision documents should follow this naming convention:

```
QuantAgent-<issue-id>-DC-<short-slug>.md
```

Example: `QuantAgent-a3f2dd-DC-caching-strategy.md`

These files describe decision deltas and architectural choices for specific changes linked to Beads issues (when applicable for system implementations).
