# Requirements: API Retry Logic Enhancement

**Issue ID:** QuantAgent-aao
**Type:** RQ (Requirements)
**Level:** ESTÁNDAR
**Created:** 2026-01-03

---

## Objetivo

Mejorar el sistema actual de retry para llamadas a LLM APIs (`invoke_with_retry` en `agent_utils.py`) para hacerlo más robusto, configurable y compatible con múltiples proveedores de LLM.

---

## Contexto

### Estado Actual

El sistema actualmente tiene:
- `quantagent/agent_utils.py:invoke_with_retry()` - Retry wrapper básico
- Maneja `RateLimitError` de OpenAI específicamente
- Retry con espera fija (no exponencial verdadera)
- Parámetros: `retries=3`, `wait_sec=4`
- Usado en: `trend_agent.py`, `pattern_agent.py`

### Limitaciones Identificadas

1. **Backoff no exponencial**: Usa espera fija en lugar de exponencial real
2. **Provider-specific**: Solo maneja `RateLimitError` de OpenAI
3. **No configurable**: Parámetros hardcodeados en cada llamada
4. **Sin fallback**: No intenta providers alternativos automáticamente
5. **Logging básico**: Prints simples sin niveles de logging estructurados

---

## Alcance (In-Scope)

### 1. Exponential Backoff Real
- Implementar backoff exponencial con jitter: `wait = base * 2^attempt + random(0, jitter)`
- Configurar: `min_wait`, `max_wait`, `multiplier`

### 2. Multi-Provider Error Handling
Soportar errores de rate limiting de:
- OpenAI: `openai.RateLimitError`
- Anthropic: `anthropic.RateLimitError`
- Otros errores comunes: timeout, connection errors
- Errores genéricos de HTTP 429

### 3. Configurabilidad
- Permitir configuración global en `default_config.py`
- Permitir override por agente/llamada específica
- Parámetros configurables:
  - `max_retries` (default: 5)
  - `base_wait` (default: 4)
  - `max_wait` (default: 60)
  - `exponential_multiplier` (default: 2)
  - `jitter` (default: True)

### 4. Logging Estructurado
- Usar `logging` module en lugar de `print()`
- Niveles apropiados: WARNING para retry, ERROR para fallo final
- Incluir contexto: attempt number, error type, wait time

### 5. Backwards Compatibility
- La función `invoke_with_retry` debe mantener su firma actual
- Código existente debe seguir funcionando sin cambios

---

## No-Alcance (Out-of-Scope)

### Explícitamente Excluido
1. **Fallback automático a providers alternativos** - Considerado pero pospuesto para fase posterior
2. **Circuit breaker pattern** - No necesario en esta iteración
3. **Métricas/telemetría de retry** - Puede agregarse después si se necesita
4. **Retry selectivo por tipo de error** - Mantener comportamiento actual de retry en todos los errores
5. **Rate limiter proactivo** - Solo reactivo (retry después de error)

### Fuera de Alcance
- Cambios a la arquitectura de LangGraph
- Modificación de lógica de agentes más allá de llamadas de retry
- Testing de integración con APIs reales (solo mocks)

---

## Casos de Uso

### UC1: Rate Limit en OpenAI (Actual)
**Given:** El agente llama a OpenAI API y recibe RateLimitError
**When:** `invoke_with_retry` captura el error
**Then:**
- Espera con backoff exponencial
- Reintenta hasta max_retries
- Loggea cada intento con nivel WARNING
- Si falla después de max retries, lanza RuntimeError con contexto

### UC2: Rate Limit en Anthropic (Nuevo)
**Given:** El agente llama a Anthropic API y recibe RateLimitError
**When:** `invoke_with_retry` captura el error
**Then:** Mismo comportamiento que UC1 (multi-provider)

### UC3: Error Transitorio (Nuevo)
**Given:** El agente llama a LLM y recibe timeout o connection error
**When:** `invoke_with_retry` captura el error
**Then:** Reintenta con backoff exponencial

### UC4: Configuración Custom por Agente (Nuevo)
**Given:** Un agente específico necesita más reintentos (ej: pattern_agent con imágenes grandes)
**When:** Se llama `invoke_with_retry(..., retries=7, base_wait=4)`
**Then:** Usa parámetros custom, ignora defaults globales

### UC5: Código Legacy Sigue Funcionando
**Given:** Código existente que usa `invoke_with_retry(llm.invoke, messages)`
**When:** Se ejecuta después del cambio
**Then:** Funciona sin modificaciones, usa defaults mejorados

---

## Constraints & Non-Functional Requirements

### Performance
- El overhead de retry no debe exceder 100ms por llamada exitosa
- Jitter aleatorio debe ser reproducible en tests

### Reliability
- Debe manejar errores inesperados sin crashear el agente
- Error messages deben ser descriptivos para debugging

### Maintainability
- Código debe estar en un solo lugar (`agent_utils.py`)
- Debe ser testeable con mocks (sin llamadas reales a APIs)
- Documentación inline (docstrings) clara

### Compatibility
- Python 3.11+
- Compatible con LangChain 0.1+
- No debe romper tests existentes

---

## Edge Cases

### EC1: Max Wait Alcanzado
Si el backoff exponencial calcula wait > max_wait, debe usar max_wait

### EC2: Retry Infinito (Prevención)
Debe haber un límite máximo absoluto de retries para prevenir loops infinitos

### EC3: Error No-Retryable
Si el error es 4xx (excepto 429), no debe reintentar, debe fallar inmediatamente

### EC4: Múltiples Proveedores en Paralelo
Si se usan múltiples providers simultáneamente (futuro), cada uno debe tener su propio retry state

---

## Dependencies

### Técnicas
- `time` module (stdlib) - Para sleep
- `logging` module (stdlib) - Para logging estructurado
- `random` module (stdlib) - Para jitter
- Tipo hints de `typing` para mejor DX

### Cambios en Otros Módulos
- `quantagent/default_config.py` - Agregar configuración de retry
- `quantagent/agent_utils.py` - Refactorizar `invoke_with_retry`
- Tests en `tests/` - Agregar tests unitarios

### Sin Nuevas Dependencias Externas
No se requieren nuevas bibliotecas en `requirements.txt`

---

## Definición de "Done"

✅ El cambio está completo cuando:

1. **Código**
   - `invoke_with_retry` implementa exponential backoff real
   - Soporta errores de OpenAI, Anthropic y genéricos
   - Configuración agregada a `default_config.py`
   - Logging estructurado implementado

2. **Tests**
   - Tests unitarios para `invoke_with_retry` con mocks
   - Casos: éxito inmediato, retry exitoso, fallo después de max retries
   - Coverage >80% de las nuevas líneas

3. **Documentación**
   - Docstring actualizado en `invoke_with_retry`
   - Ejemplo de uso en docstring
   - CLAUDE.md actualizado si es necesario

4. **Validación**
   - Tests existentes siguen pasando (backwards compatibility)
   - Tests nuevos pasan
   - Código existente en agentes funciona sin cambios

5. **Integración**
   - Cambios commiteados con mensaje descriptivo
   - Vinculado a Issue de Beads
   - Documentación de planning/design completada

---

## References

- Código actual: `quantagent/agent_utils.py:invoke_with_retry()`
- Uso en: `quantagent/trend_agent.py:98-99`, `quantagent/pattern_agent.py`
- Conversación original: [2026-01-03] Mejora de retry logic para LangGraph

---

## Notes

- **Nivel ESTÁNDAR**: No es un cambio arquitectónico (no COMPREHENSIVE), pero tampoco es trivial (no MÍNIMO)
- **No romper lo existente**: Prioridad alta en backwards compatibility
- **Testeable sin APIs**: Todos los tests deben funcionar con mocks
