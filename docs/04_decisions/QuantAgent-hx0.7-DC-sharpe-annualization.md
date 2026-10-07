# Decisión: Periodos por año para Sharpe ratio en fixtures 24/7

**Issue:** QuantAgent-hx0.7 · **Estado:** Decidida (D)

## Contexto
`_get_periods_per_year()` en `quantagent/backtesting/backtest.py` usa $252 \times 6.5 = 1.638$ para 1h.
Los fixtures actuales (`spy-90d`) son continuos 24/7 (`market_hours_filter=False`), con 24 velas por día los 7 días de la semana. Anualizar retornos horarios continuos con horas bursátiles de NYSE sub-escala el Sharpe por un factor de $\sqrt{8760/1638} \approx 2.31$.

## Opciones

| Opción | Base $N$ (1h) | $N$ | Sharpe (0.05% slip) | Sharpe (1.00% slip) |
|---|---|---:|---:|---:|
| **A (Actual)** | Rueda NYSE ($252 \times 6.5$) | 1.638 | **0.57** | 0.40 |
| **B (Continuo 24/7)** | Calendario anual ($365 \times 24$) | 8.760 | **1.37** | 0.99 |
| **C (Hábil 24h)** | Días hábiles 24h ($252 \times 24$) | 6.048 | **1.13** | 0.82 |
| **D (Derivada de datos)** | Velas obs. / años ($2159 / 0.2463$) | 8.766 | **25.14** (antes 1.37) | -7.98 (antes 0.99) |

*Nota: las filas A, B y C se calcularon con la equity curve anterior a hx0.3.*

- **Opción A (NYSE):** Asume 6.5 h/día, 252 días/año. Válido solo si se activa filtro de horario bursátil.
- **Opción B (24/7):** Asume 24 h/día, 365 días/año. Fiel a la frecuencia horaria continua de los fixtures actuales.
- **Opción C (Hábil 24h):** Asume 24 h/día, 252 días/año (formato FX/futuros). Inconsistente con velas de fin de semana en el fixture.
- **Opción D (Derivada de datos):** Periodos por año derivados de la equity curve: retornos observados / años entre primer y último timestamp (con año = 365,25 días). Para `spy-90d` da $N = 2159 / (89.9583 / 365.25) = 8.766$. Sin tabla fija por activo ni dependencia de `market_hours_filter`.

## Decisión
Decidida por Fede (2026-10-06, PR #24, implementada en QuantAgent-hx0.4): se adopta la **Opción D**.
Los periodos por año se derivan automáticamente de los datos de la corrida a partir de los timestamps de la equity curve.
El Sharpe de referencia en RSI/spy-90d con la equity curve corregida (QuantAgent-hx0.3) pasa de 8.74 ($N=1638$) a **25.14** ($N=8766$).
