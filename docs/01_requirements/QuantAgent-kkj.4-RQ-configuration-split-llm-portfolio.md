# QuantAgent-kkj.4 — RQ: Separar pestaña Configuration en LLM Settings y Portfolio & Universe

**Issue:** QuantAgent-kkj.4  
**Type:** UX / UI reorganization  
**Parent:** QuantAgent-kkj (M2 Milestone)  
**Status:** open

---

## Problema

La pestaña Configuration de Streamlit (`apps/streamlit/views/configuration.py`) mezcla dos categorías conceptualmente independientes en la misma vista sin separación visual clara:

1. **LLM profiles/presets** — configuración de providers (OpenAI, Anthropic, etc.), modelos, temperatura y parámetros de inferencia.
2. **Portfolio/Universe config** — portfolio profiles, risk profiles, universos de activos, defaults paper/backtest, carga de profiles JSON.

Problemas específicos observados (revisión funcional M2, 2026-05-25):

- Los combos "Paper default portfolio" y "Backtest default portfolio" aparecen vacíos sin explicación de cómo crear un portfolio profile.
- La sección de carga de Profile JSON está mezclada visualmente con los controles de LLM.
- El selector de universe está etiquetado "(for portfolio profiles only)" pero no queda claro en qué contexto aplica.
- Un operador con conocimiento del sistema tuvo que consultar el user manual para entender la UI.

---

## Cambio requerido

Reorganizar `configuration.py` usando **`st.tabs`** para crear dos sub-tabs dentro de la pestaña Configuration:

### Tab 1: LLM Settings
Contiene exclusivamente la configuración de inferencia LLM:
- Selector de preset (nombre)
- Selector de provider (openai / anthropic / qwen)
- Input de model name
- Slider de temperature
- Input "Save as (name)"
- Botón "Save preset"
- Preview de presets (dataframe)

### Tab 2: Portfolio & Universe
Contiene exclusivamente la configuración de portafolios y universos:
- Editor de profiles (kind selector: portfolio/risk/combined, name, JSON textarea)
- Universe multiselect (solo activo para kind=portfolio) con caption explicativo
- Universe preview y warnings de símbolos no soportados
- Botón "Save profile"
- Tabla de profiles existentes
- Sección "Defaults per environment" con:
  - Selectores "Paper default portfolio" / "Backtest default portfolio" + caption de ayuda
  - Selectores "Paper default strategy" / "Backtest default strategy"

### Help text a agregar

- En "Paper default portfolio" y "Backtest default portfolio": caption que explica que el selector se pobla con los portfolio profiles guardados. Ejemplo: *"Select a portfolio profile to use as default for this environment. Create profiles using the Profile editor above."*
- En el universe selector (cuando kind ≠ portfolio): caption existente ya explicativo ("Universe editing is available for portfolio profiles only.") — conservar.

---

## Archivos afectados

| Archivo | Cambio |
|---|---|
| `apps/streamlit/views/configuration.py` | Reorganización visual con `st.tabs`; sin cambios de lógica backend |

---

## Fuera de scope

- No cambiar la lógica backend de configuración ni los modelos de datos.
- No rediseñar otras pestañas.
- No implementar nuevos tipos de profiles ni nuevas integraciones de LLM.
- No modificar el modelo `StrategyConfig` ni las funciones `_collect_profiles_from_db` / `_get_profile_json_from_db`.
