# QuantAgent-ou3: Fix SPX symbol mapping for yfinance

**Issue ID:** QuantAgent-ou3
**Priority:** P2 (Medium)
**Type:** Bug
**Status:** Open

---

## Resumen Ejecutivo

yfinance no soporta datos intraday (4h) para indices como SPX (`^GSPC`). El provider mapea correctamente SPX a `^GSPC` pero la API retorna error "possibly delisted" cuando se solicitan datos con timeframe menor a 1d. Requiere una estrategia de fallback o restriccion de timeframes validos.

---

## Evidencia del Log

```
2026-01-06 21:58:30,001 - quantagent.data.provider - INFO - Fetching OHLC data for SPX (4h) from 2025-12-03...
2026-01-06 21:58:30,294 - yfinance - ERROR - $^GSPC: possibly delisted; no price data found (4h 2025-12-03...)
```

---

## Root Cause

En `quantagent/data/provider.py` linea 38:
```python
SYMBOL_MAPPING = {
    "SPX": "^GSPC",  # S&P 500 Index
    ...
}
```

**Limitacion de yfinance:** Para indices (`^` prefix), solo soporta datos diarios o mayores. Timeframes intraday (1h, 4h) no estan disponibles via API gratuita.

---

## Soluciones Propuestas

| Opcion | Descripcion | Pros | Contras |
|--------|-------------|------|---------|
| **A. Fallback a SPY** | Usar ETF proxy SPY cuando SPX + intraday | Disponible, correlacion alta | No es identico al indice |
| **B. Restriccion de timeframe** | Forzar timeframe >= 1d para indices | Solucion limpia | Limita funcionalidad |
| **C. Validacion + error claro** | Detectar combinacion invalida y error descriptivo | Transparente | No resuelve necesidad del usuario |

**Recomendacion:** Opcion A con logging que indique el fallback.

---

## Criterios de Aceptacion

- [ ] Backtest con SPX (4h) no produce error "possibly delisted"
- [ ] Log indica claramente si se usa fallback (ej: "Using SPY as proxy for SPX intraday data")
- [ ] Datos retornados son validos y completos para el rango solicitado
- [ ] Comportamiento para SPX con timeframe diario (1d) no cambia
