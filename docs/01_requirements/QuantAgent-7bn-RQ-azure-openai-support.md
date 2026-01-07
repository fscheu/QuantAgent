# Requirements: Azure OpenAI LLM Provider Support

**Issue ID**: QuantAgent-7bn  
**Type**: Feature Enhancement  
**Level**: ESTÁNDAR

---

## Objetivo

Agregar Azure OpenAI como proveedor de LLM adicional al sistema multi-agente, permitiendo a los usuarios utilizar modelos OpenAI desplegados en infraestructura Azure.

---

## Alcance

### Incluye
- Configuración de Azure OpenAI via variables de entorno
- Soporte para `azure_endpoint`, `api_version`, y `azure_deployment`
- Detección automática del proveedor "azure" en `trading_graph.py`
- Instanciación correcta de `AzureChatOpenAI` de LangChain
- Actualización de `.env.example` con ejemplos de configuración
- Compatibilidad con configuración separada para Agent LLM y Graph LLM

### No Incluye
- Cambios en UI (Streamlit/Flask) para selección de proveedor
- Validación de credenciales en tiempo de configuración
- Migración automática de configuraciones existentes
- Soporte para Azure AI Studio endpoints (solo Azure OpenAI Service)
- Cambios en lógica de agentes individuales

---

## Constraints

- **Compatibilidad**: Debe mantener 100% compatibilidad con proveedores existentes (OpenAI, Anthropic, Qwen)
- **Configuración**: Usa el patrón existente en `settings.py` (variables de entorno + dotenv)
- **Dependencias**: LangChain ya incluye `langchain-openai` que soporta Azure
- **Defaults**: Si no se especifica `api_version`, usar `"2024-02-01"` (versión estable actual)

---

## Flujo de Configuración

1. Usuario configura variables en `.env`:
   ```
   AGENT_LLM_PROVIDER=azure
   AZURE_OPENAI_API_KEY=<key>
   AZURE_OPENAI_ENDPOINT=https://<resource>.openai.azure.com/
   AZURE_OPENAI_DEPLOYMENT=gpt-4o
   AZURE_OPENAI_API_VERSION=2024-02-01
   ```

2. Sistema detecta `provider=azure` en `_create_llm()`

3. Instancia `AzureChatOpenAI` con parámetros de Azure

4. Agentes operan normalmente sin cambios

---

## Edge Cases

- **Provider mismatch**: Si `AGENT_LLM_PROVIDER=azure` pero falta `AZURE_OPENAI_ENDPOINT` → error claro
- **Deployment name**: `AZURE_OPENAI_DEPLOYMENT` es obligatorio (diferente al modelo base)
- **API version**: Si no está configurado, usar default `"2024-02-01"`
- **Temperature**: Respetar `AGENT_LLM_TEMPERATURE` y `GRAPH_LLM_TEMPERATURE` existentes

---

## Definition of Done

- [ ] Usuario puede configurar Azure OpenAI via `.env`
- [ ] Sistema instancia correctamente `AzureChatOpenAI` 
- [ ] Backtests corren exitosamente con Azure como proveedor
- [ ] `.env.example` incluye ejemplo comentado de configuración Azure
- [ ] No hay regresión en proveedores existentes (OpenAI, Anthropic, Qwen)
- [ ] Error messages claros si faltan variables requeridas
