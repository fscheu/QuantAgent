# Design: Azure OpenAI LLM Provider Support

**Issue ID**: QuantAgent-7bn  
**Type**: Feature Enhancement  
**Level**: ESTÁNDAR

---

## Componentes Afectados

- `quantagent/settings.py` — Agregar variables de configuración Azure
- `quantagent/trading_graph.py` — Extender `_create_llm()` y `_get_api_key()`
- `.env.example` — Documentar configuración Azure

---

## Decisiones Técnicas

### 1. Proveedor "azure" vs "azure-openai"
**Decisión**: Usar `"azure"` como valor de `AGENT_LLM_PROVIDER`  
**Razón**: Más corto, consistente con otros proveedores (`openai`, `anthropic`, `qwen`)

### 2. Namespace de variables
**Decisión**: Prefijo `AZURE_OPENAI_` para todas las variables  
**Razón**: 
- Claridad (diferencia de `OPENAI_API_KEY`)
- Consistencia con convención LangChain
- Evita ambigüedad con otros servicios Azure

### 3. API version default
**Decisión**: `"2024-02-01"` como default si no se especifica  
**Razón**: Versión GA estable, soporta modelos recientes (gpt-4o, gpt-4-turbo)

### 4. Deployment name obligatorio
**Decisión**: `AZURE_OPENAI_DEPLOYMENT` es requerido (no tiene default)  
**Razón**: Es específico del deployment del usuario en Azure, no hay valor "razonable" para defaultear

---

## Contratos

### Variables de Entorno (nuevas)

```python
# settings.py
AZURE_OPENAI_API_KEY: str = os.getenv("AZURE_OPENAI_API_KEY", "")
AZURE_OPENAI_ENDPOINT: str = os.getenv("AZURE_OPENAI_ENDPOINT", "")
AZURE_OPENAI_DEPLOYMENT: str = os.getenv("AZURE_OPENAI_DEPLOYMENT", "")
AZURE_OPENAI_API_VERSION: str = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01")
```

### Ejemplo (mínimo)

```python
# trading_graph.py._create_llm()
elif provider == "azure":
    from langchain_openai import AzureChatOpenAI
    
    endpoint = settings.AZURE_OPENAI_ENDPOINT
    deployment = settings.AZURE_OPENAI_DEPLOYMENT
    api_version = settings.AZURE_OPENAI_API_VERSION
    
    if not endpoint:
        raise ValueError("AZURE_OPENAI_ENDPOINT not set")
    if not deployment:
        raise ValueError("AZURE_OPENAI_DEPLOYMENT not set")
    
    return AzureChatOpenAI(
        azure_endpoint=endpoint,
        azure_deployment=deployment,
        api_version=api_version,
        api_key=api_key,
        temperature=temperature,
    )
```

---

## Flujo de Inicialización

```
Usuario configura .env
    ↓
settings.py carga variables
    ↓
TradingGraph.__init__()
    ↓
_create_llm(provider="azure", ...)
    ↓
Detecta provider == "azure"
    ↓
Valida endpoint, deployment, api_key
    ↓
Instancia AzureChatOpenAI
    ↓
Retorna instancia de BaseChatModel
```

---

## Compatibilidad

### LangChain
- `langchain-openai >= 0.0.5` ya incluye `AzureChatOpenAI`
- No requiere dependencias adicionales

### Proveedores Existentes
- No hay cambios en `openai`, `anthropic`, `qwen` branches
- Pattern matching es exhaustivo (no afecta defaults)

---

## Error Handling

### Estrategia
- **Fail fast**: Validar en `_create_llm()` antes de instanciar
- **Mensajes claros**: Incluir nombre de variable y ejemplo
- **Consistencia**: Mismo estilo que errores existentes

### Ejemplo (mínimo)

```python
if not settings.AZURE_OPENAI_ENDPOINT:
    raise ValueError(
        "AZURE_OPENAI_ENDPOINT not found in .env file. "
        "Example: https://myresource.openai.azure.com/"
    )
```

---

## Alternativas Consideradas

### Alt 1: Reusar `AGENT_LLM_PROVIDER=openai` + flag `USE_AZURE=true`
**Rechazada**: Más complejo, menos explícito, rompe simetría con otros proveedores

### Alt 2: Auto-detectar Azure basado en presencia de `AZURE_OPENAI_ENDPOINT`
**Rechazada**: Implícito, puede causar confusión si usuario tiene múltiples configuraciones

### Alt 3: Default deployment name basado en modelo
**Rechazada**: Deployment names son arbitrarios (definidos por usuario en Azure Portal)

---

## Impacto en Testing

### Mockeo
```python
# Para tests unitarios
@patch("quantagent.trading_graph.AzureChatOpenAI")
def test_azure_provider(mock_azure):
    os.environ["AGENT_LLM_PROVIDER"] = "azure"
    # ... configurar env vars ...
    graph = TradingGraph()
    mock_azure.assert_called_once()
```

### Testing Real
- Requiere Azure OpenAI Service deployment activo
- Opcional para merge (validación manual si no hay acceso)

---

## Dependencias de Implementación

1. `settings.py` debe existir antes de modificar `trading_graph.py`
2. `.env.example` se actualiza al final (documentación)
3. No hay dependencias de base de datos o migraciones
