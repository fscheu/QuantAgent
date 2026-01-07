# Planning: Azure OpenAI LLM Provider Support

**Issue ID**: QuantAgent-7bn  
**Type**: Feature Enhancement  
**Estimated Effort**: 1.5–2.5 hours

---

## Tareas

### 1. Extender `settings.py` con configuración Azure
**Estimado**: 30 min  
**Dependencias**: Ninguna

- Agregar variables de entorno:
  - `AZURE_OPENAI_API_KEY`
  - `AZURE_OPENAI_ENDPOINT`
  - `AZURE_OPENAI_DEPLOYMENT`
  - `AZURE_OPENAI_API_VERSION` (default: `"2024-02-01"`)
- Seguir patrón existente de `OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, etc.
- Agregar helper `_get_azure_config()` si necesario

### 2. Modificar `trading_graph.py._create_llm()`
**Estimado**: 45 min  
**Dependencias**: Tarea 1

- Agregar branch `elif provider == "azure":`
- Instanciar `AzureChatOpenAI` con parámetros:
  - `azure_endpoint`
  - `api_key`
  - `azure_deployment`
  - `api_version`
  - `temperature`
- Validar que variables requeridas estén presentes
- Agregar mensajes de error claros si faltan

### 3. Actualizar `trading_graph._get_api_key()`
**Estimado**: 15 min  
**Dependencias**: Tarea 1

- Agregar case para `provider == "azure"`
- Retornar `AZURE_OPENAI_API_KEY`
- Mensaje de error específico si falta

### 4. Actualizar `.env.example`
**Estimado**: 10 min  
**Dependencias**: Tarea 1

- Agregar sección comentada con configuración Azure
- Incluir ejemplo completo con placeholders
- Documentar parámetros obligatorios vs opcionales

### 5. Testing y validación
**Estimado**: 30 min  
**Dependencias**: Tareas 1-4

- Verificar proveedores existentes no se rompan (OpenAI, Anthropic, Qwen)
- Test manual con configuración Azure (si hay acceso)
- Validar error handling cuando faltan variables

---

## Dependencias Externas

- **LangChain**: Ya incluye `langchain-openai` con soporte Azure (no requiere instalación adicional)
- **Azure OpenAI Service**: Usuario debe tener deployment activo y credenciales

---

## Riesgos

| Riesgo | Probabilidad | Impacto | Mitigación |
|--------|--------------|---------|------------|
| Breaking change en otros proveedores | Baja | Alto | Testing exhaustivo de regresión |
| API version incompatible | Media | Bajo | Usar versión estable default (`2024-02-01`) |
| Falta de acceso a Azure para testing | Alta | Medio | Validar con mocks o documentación |

---

## Estrategia de Testing

### Unit Testing (opcional)
- Mock de `AzureChatOpenAI` para verificar parámetros de inicialización

### Integration Testing
- Backtest end-to-end con Azure configurado (si hay acceso)
- Verificar que `graph.invoke()` funciona correctamente

### Regression Testing
- Re-ejecutar backtests con OpenAI, Anthropic, Qwen
- Verificar que no hay cambios en comportamiento

---

## Rollout

1. **Feature branch**: `feature/QuantAgent-7bn-azure-openai-support`
2. **Testing**: Validación local + CI (si aplica)
3. **Merge**: A main después de aprobación humana
4. **Documentación**: `.env.example` es suficiente (no requiere docs adicionales)

---

## Checkpoints

- [ ] Variables de entorno agregadas en `settings.py`
- [ ] `_create_llm()` detecta y maneja `provider=azure`
- [ ] `.env.example` actualizado con ejemplo Azure
- [ ] No hay regresión en proveedores existentes
- [ ] Error handling claro para configuración incompleta
