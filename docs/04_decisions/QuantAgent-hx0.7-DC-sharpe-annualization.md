# Decisión: Periodos por año para Sharpe ratio en fixtures 24/7

**Issue:** QuantAgent-hx0.7 · **Estado:** Propuesta

## Contexto
`_get_periods_per_year()` en `quantagent/backtesting/backtest.py` usa $252 \times 6.5 = 1.638$ para 1h.
Los fixtures actuales (`spy-90d`) son continuos 24/7 (`market_hours_filter=False`), con 24 velas por día los 7 días de la semana. Anualizar retornos horarios continuos con horas bursátiles de NYSE sub-escala el Sharpe por un factor de $\sqrt{8760/1638} \approx 2.31$.

## Opciones

| Opción | Base $N$ (1h) | $N$ | Sharpe (0.05% slip) | Sharpe (1.00% slip) |
|---|---|---:|---:|---:|
| **A (Actual)** | Rueda NYSE ($252 \times 6.5$) | 1.638 | **0.57** | 0.40 |
| **B (Continuo 24/7)** | Calendario anual ($365 \times 24$) | 8.760 | **1.37** | 0.99 |
| **C (Hábil 24h)** | Días hábiles 24h ($252 \times 24$) | 6.048 | **1.13** | 0.82 |

- **Opción A (NYSE):** Asume 6.5 h/día, 252 días/año. Válido solo si se activa filtro de horario bursátil.
- **Opción B (24/7):** Asume 24 h/día, 365 días/año. Fiel a la frecuencia horaria continua de los fixtures actuales.
- **Opción C (Hábil 24h):** Asume 24 h/día, 252 días/año (formato FX/futuros). Inconsistente con velas de fin de semana en el fixture.

## Recomendación
Elegir **Opción B (365×24 = 8.760)** mientras se utilicen fixtures continuos 24/7 (o condicional según `market_hours_filter` al implementar `QuantAgent-hx0.4`: 8.760 para 24/7 y 1.638 para rueda bursátil).
El Sharpe de referencia en RSI/spy-90d pasa de 0.57 a **1.37**.
