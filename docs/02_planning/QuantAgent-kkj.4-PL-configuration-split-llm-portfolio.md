# QuantAgent-kkj.4 — PL: Separar pestaña Configuration en LLM Settings y Portfolio & Universe

**Issue:** QuantAgent-kkj.4  
**Phase:** planner  
**Run ID:** 20260528T124136Z-QuantAgent-kkj.4-planner  
**Requirements:** [QuantAgent-kkj.4-RQ-configuration-split-llm-portfolio.md](../01_requirements/QuantAgent-kkj.4-RQ-configuration-split-llm-portfolio.md)

---

## Approach

**Un solo archivo cambia:** `apps/streamlit/views/configuration.py`.

La reorganización usa `st.tabs(["LLM Settings", "Portfolio & Universe"])` inmediatamente después del `st.subheader` y la inicialización de session state. El contenido existente se distribuye entre los dos tabs sin modificar la lógica.

No hay cambios de backend, modelos, ni otras vistas.

---

## Estado actual de `configuration.py`

```
render(db, environment):
  st.subheader(...)
  st.caption(...)
  session_state init (ui_profiles, model_presets, default_profiles, default_strategy)

  colL, colR = st.columns([2, 1])
  with colL:
    [A] Profile kind selector + name input
    [B] Profile JSON textarea (con load desde DB o session)
    [C] Universe multiselect (activo solo para kind=portfolio)
    [D] Universe preview + warnings
    [E] Botón "Save profile"
    [F] Tabla "Profiles"
  with colR:
    [G] "Defaults per environment" — portfolio selectors (paper/backtest)
    [H] "Strategy Defaults" — strategy selectors (paper/backtest)
    [I] "Model presets" — provider, model_name, temperature, save preset
    [J] "Presets preview" dataframe
```

---

## Target layout

```
render(db, environment):
  st.subheader(...)
  st.caption(...)
  session_state init (sin cambios)

  tab_llm, tab_portfolio = st.tabs(["LLM Settings", "Portfolio & Universe"])

  with tab_llm:
    [I] Model presets section (selector, provider, model_name, temperature, save as, save button)
    [J] Presets preview dataframe

  with tab_portfolio:
    colL, colR = st.columns([2, 1])
    with colL:
      [A] Profile kind selector + name input
      [B] Profile JSON textarea
      [C] Universe multiselect (con caption explicativo)
      [D] Universe preview + warnings
      [E] Botón "Save profile"
      [F] Tabla "Profiles"
    with colR:
      [G] "Defaults per environment" — portfolio selectors (paper/backtest) + caption de ayuda
      [H] "Strategy Defaults" — strategy selectors (paper/backtest)
```

---

## Implementación detallada

### 1. Añadir `st.tabs` tras la inicialización de session state

```python
tab_llm, tab_portfolio = st.tabs(["LLM Settings", "Portfolio & Universe"])
```

### 2. Tab "LLM Settings" (`with tab_llm:`)

Mover el bloque actualmente en `colR` a partir de `st.markdown("**Model presets**")` hasta el final de la función (líneas 243–289 del archivo actual):

```python
with tab_llm:
    st.markdown("**Model presets**")
    # ... resto del bloque de presets (sin cambios internos)
```

### 3. Tab "Portfolio & Universe" (`with tab_portfolio:`)

Envolver el bloque `colL, colR = st.columns([2, 1])` existente dentro de este tab, con dos ajustes menores:

**3a. Caption de ayuda en los selectores de default portfolio** (dentro del loop `for env_key in ("paper", "backtest"):` en colR):

Agregar debajo del `st.selectbox(...)`:
```python
st.caption(
    "Select a saved portfolio profile. Create profiles using the Profile editor on the left."
)
```

**3b. Header del tab** (opcional, para claridad visual):

```python
with tab_portfolio:
    st.caption(
        "Manage portfolio profiles, risk profiles, asset universes, and environment defaults."
    )
    colL, colR = st.columns([2, 1])
    ...
```

### 4. No cambiar

- Las funciones `_collect_profiles_from_db` y `_get_profile_json_from_db`.
- El layout interno de `colL` (secciones A-F) ni de `colR` (secciones G-H), salvo el caption añadido en G.
- El bloque de session_state init.
- Los imports.

---

## Archivos a modificar

| Archivo | Tipo de cambio |
|---|---|
| `apps/streamlit/views/configuration.py` | Reorganización visual: `st.tabs`, mover bloques, añadir 2 captions |

---

## Validación post-implementación

```bash
# 1. Syntax check
/home/azureuser/repos/projects/QuantAgent/.venv/bin/python -m compileall -q apps/streamlit/views/configuration.py

# 2. Import check
/home/azureuser/repos/projects/QuantAgent/.venv/bin/python -c "import apps.streamlit.views.configuration"

# 3. Test suite (regression)
/home/azureuser/repos/projects/QuantAgent/.venv/bin/python -m pytest tests/ -x -q --timeout=30 2>&1 | tail -20
```

La validación visual (que los dos tabs aparezcan correctamente en el navegador) es responsabilidad de la fase tester.

---

## Riesgos

| Riesgo | Probabilidad | Mitigación |
|---|---|---|
| Widget key collision si Streamlit reusa keys de session state | Baja | Los widget keys (`model_provider`, `model_name`, etc.) no cambian de contexto tab — Streamlit los mantiene correctamente dentro del tab scope |
| Regresión visual en colR (defaults section) | Muy baja | Sólo se añade un `st.caption`; no se cambia la lógica ni el layout de columns |
| Tests existentes que mockean `st.tabs` | Baja | Revisar `tests/` antes de implementar; si hay tests de `configuration.py`, verificar que acepten la nueva estructura |

---

## Estimación de complejidad

**Baja.** Cambios puramente visuales/estructurales en un solo archivo. No hay lógica nueva. La reorganización es un reordenamiento de bloques existentes con wrapping en tabs.
