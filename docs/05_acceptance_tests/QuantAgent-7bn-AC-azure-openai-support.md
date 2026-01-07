# Acceptance Criteria: Azure OpenAI LLM Provider Support

**Issue ID**: QuantAgent-7bn  
**Type**: Feature Enhancement

---

## Criterios de Aceptación

### AC1: Configuración válida de Azure
```
Given las variables de entorno están configuradas correctamente:
  - AGENT_LLM_PROVIDER=azure
  - AZURE_OPENAI_API_KEY=<valid-key>
  - AZURE_OPENAI_ENDPOINT=https://myresource.openai.azure.com/
  - AZURE_OPENAI_DEPLOYMENT=gpt-4o
  - AZURE_OPENAI_API_VERSION=2024-02-01
When TradingGraph se inicializa
Then se crea una instancia de AzureChatOpenAI con los parámetros correctos
  And no se lanza ninguna excepción
```

### AC2: API version default
```
Given las variables de entorno están configuradas sin AZURE_OPENAI_API_VERSION
When TradingGraph se inicializa
Then se usa el default "2024-02-01"
  And el sistema funciona normalmente
```

### AC3: Error handling - endpoint faltante
```
Given AGENT_LLM_PROVIDER=azure
  And AZURE_OPENAI_ENDPOINT no está configurado
When TradingGraph se inicializa
Then se lanza ValueError con mensaje claro
  And el mensaje indica que falta AZURE_OPENAI_ENDPOINT
```

### AC4: Error handling - API key faltante
```
Given AGENT_LLM_PROVIDER=azure
  And AZURE_OPENAI_API_KEY no está configurado
When TradingGraph._get_api_key("azure") se ejecuta
Then se lanza ValueError con mensaje claro
  And el mensaje indica que falta AZURE_OPENAI_API_KEY
```

### AC5: Error handling - deployment faltante
```
Given AGENT_LLM_PROVIDER=azure
  And AZURE_OPENAI_DEPLOYMENT no está configurado
When TradingGraph se inicializa
Then se lanza ValueError o error claro
  And el mensaje indica que falta AZURE_OPENAI_DEPLOYMENT
```

### AC6: Compatibilidad con configuración dual
```
Given AGENT_LLM_PROVIDER=azure
  And GRAPH_LLM_PROVIDER=openai
When TradingGraph se inicializa
Then agent_llm usa AzureChatOpenAI
  And graph_llm usa ChatOpenAI
  And ambos funcionan independientemente
```

### AC7: Backtest end-to-end con Azure
```
Given configuración válida de Azure
  And un backtest configurado con datos de mercado
When el backtest se ejecuta
Then completa sin errores
  And los agentes procesan señales correctamente
  And los resultados son consistentes con otros proveedores
```

---

## Criterios de Regresión

### REG1: OpenAI sin cambios
```
Given AGENT_LLM_PROVIDER=openai (configuración pre-existente)
When TradingGraph se inicializa
Then comportamiento es idéntico a versión anterior
  And no hay breaking changes
```

### REG2: Anthropic sin cambios
```
Given AGENT_LLM_PROVIDER=anthropic (configuración pre-existente)
When TradingGraph se inicializa
Then comportamiento es idéntico a versión anterior
  And no hay breaking changes
```

### REG3: Qwen sin cambios
```
Given AGENT_LLM_PROVIDER=qwen (configuración pre-existente)
When TradingGraph se inicializa
Then comportamiento es idéntico a versión anterior
  And no hay breaking changes
```

---

## Invariantes

- **Temperatura**: `AGENT_LLM_TEMPERATURE` y `GRAPH_LLM_TEMPERATURE` se respetan para Azure
- **Checkpointing**: Si `use_checkpointing=True`, funciona con Azure igual que otros proveedores
- **Error messages**: Siempre incluyen nombre de variable faltante y ejemplo de valor esperado

---

## Oráculos de Validación

### Validación de configuración
```bash
# Verificar que settings.py tiene las variables
grep -q "AZURE_OPENAI_API_KEY" quantagent/settings.py
grep -q "AZURE_OPENAI_ENDPOINT" quantagent/settings.py
```

### Validación de instanciación
```python
# En Python REPL o test
from quantagent.trading_graph import TradingGraph
import os

os.environ["AGENT_LLM_PROVIDER"] = "azure"
os.environ["AZURE_OPENAI_API_KEY"] = "test-key"
os.environ["AZURE_OPENAI_ENDPOINT"] = "https://test.openai.azure.com/"
os.environ["AZURE_OPENAI_DEPLOYMENT"] = "gpt-4o"

graph = TradingGraph()
assert "AzureChatOpenAI" in str(type(graph.agent_llm))
```

### Validación de .env.example
```bash
# Verificar que el ejemplo existe
grep -q "AZURE_OPENAI" .env.example
```

---

## Datos de Prueba

### Configuración mínima válida
```env
AGENT_LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=sk-test-key-here
AZURE_OPENAI_ENDPOINT=https://myresource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
```

### Configuración completa
```env
AGENT_LLM_PROVIDER=azure
GRAPH_LLM_PROVIDER=azure
AZURE_OPENAI_API_KEY=sk-test-key-here
AZURE_OPENAI_ENDPOINT=https://myresource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_API_VERSION=2024-02-01
AGENT_LLM_TEMPERATURE=0.1
GRAPH_LLM_TEMPERATURE=0.1
```
