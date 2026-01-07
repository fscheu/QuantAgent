# QuantAgent-lmn: Fix deprecated wait_sec parameter

**Issue ID:** QuantAgent-lmn
**Priority:** P3 (Low)
**Type:** Task (Technical debt)
**Status:** Open

---

## Resumen Ejecutivo

El parámetro `wait_sec` en `invoke_with_retry()` esta marcado como deprecated desde la implementacion del retry con backoff exponencial. Los callers internos deben migrar a `base_wait`. Es un refactor simple de actualizacion de parametros sin cambio de comportamiento.

---

## Evidencia del Log

```
2026-01-06 21:58:17,452 - quantagent.agent_utils - WARNING - wait_sec parameter is deprecated, use base_wait instead
```

---

## Root Cause

En `quantagent/agent_utils.py` linea 100, `invoke_with_retry()` acepta `wait_sec` por retrocompatibilidad pero emite warning. Los siguientes modulos llaman con `wait_sec`:

- `indicator_agent.py`
- `pattern_agent.py`
- `trend_agent.py`
- `decision_agent.py`

---

## Solucion Propuesta

1. **Buscar y reemplazar** todas las llamadas `wait_sec=` por `base_wait=` en los agentes
2. **Opcional (futuro):** Remover soporte de `wait_sec` en version futura

---

## Criterios de Aceptacion

- [ ] No aparece warning "wait_sec parameter is deprecated" en logs durante backtest
- [ ] Comportamiento de retry identico (mismos tiempos de espera)
- [ ] Tests existentes pasan sin modificacion
