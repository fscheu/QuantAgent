# QuantAgent-kkj.5 — PL: Agregar help contextual en Configuration

**Issue:** QuantAgent-kkj.5  
**Phase:** planner  
**Run ID:** 20260529T023634Z-QuantAgent-kkj.5-planner  
**Requirements:** [QuantAgent-kkj.5-RQ-configuration-contextual-help.md](../01_requirements/QuantAgent-kkj.5-RQ-configuration-contextual-help.md)  
**Acceptance Criteria:** [QuantAgent-kkj.5-AC-configuration-contextual-help.md](../05_acceptance_tests/QuantAgent-kkj.5-AC-configuration-contextual-help.md)

---

## Approach

**Un solo archivo cambia:** `apps/streamlit/views/configuration.py`.

Todos los cambios son additive: agregar `help=` parameters a widgets existentes y `st.caption(...)` calls en puntos estratégicos. No se modifica lógica, layout, ni session state.

**Dependencia:** QuantAgent-kkj.4 ya reorganizó la vista en dos tabs (`LLM Settings` / `Portfolio & Universe`). Este plan asume que esa reorganización está mergeada (el código actual en main ya la tiene).

---

## Estado actual de `configuration.py` (post kkj.4)

```
render(db, environment):
  st.subheader(...)
  st.caption(...)                     ← caption genérico existente
  session_state init

  tab_llm, tab_portfolio = st.tabs(["LLM Settings", "Portfolio & Universe"])

  with tab_llm:
    st.markdown("**Model presets**")
    st.selectbox("Preset name", ...)
    st.selectbox("Provider", ...)      ← sin help=
    st.text_input("Model name", ...)   ← sin help=
    st.slider("Temperature", ...)
    st.text_input("Save as (name)", ...)
    st.button("Save preset")
    st.dataframe(...)

  with tab_portfolio:
    st.caption("Manage portfolio profiles...")
    colL, colR = st.columns([2, 1])
    with colL:
      st.selectbox("Profile kind", ...)
      st.text_input("Profile name", ...)
      st.text_area("Profile JSON", ...)    ← sin caption aclaratorio
      st.multiselect("Universe...", ...)   ← sin help=
      ...
    with colR:
      for env_key in ("paper", "backtest"):
        st.selectbox(f"{env_key.title()} default portfolio", ...) ← sin help=
        st.caption("Select a saved portfolio profile...")          ← caption existente genérico
        ...
```

---

## Cambios a implementar

### Cambio 1 — Caption en tab LLM Settings (AC1)

Agregar inmediatamente después de `with tab_llm:` y antes del `st.markdown("**Model presets**")`:

```python
st.caption(
    "These parameters apply only to LLM-based strategies (e.g. LLM Agent). "
    "Deterministic strategies (RSI, 52-week high) do not use these settings."
)
```

### Cambio 2 — help= en selector Provider (AC2)

En `st.selectbox("Provider", provider_options, ...)`:

```python
provider = st.selectbox(
    "Provider",
    provider_options,
    index=provider_index,
    key="model_provider",
    help="LLM API provider. 'openai' uses the OpenAI API, 'anthropic' uses the Anthropic API, 'qwen' uses Alibaba Qwen via compatible endpoint.",
)
```

### Cambio 3 — help= en input Model name (AC3)

En `st.text_input("Model name", ...)`:

```python
model_name = st.text_input(
    "Model name",
    value=preset.get("model_name", "gpt-4o-mini"),
    key="model_name",
    help="Model identifier as expected by the provider API. Examples: 'gpt-4o-mini' (openai), 'claude-3-haiku-20240307' (anthropic), 'qwen-vl-max' (qwen).",
)
```

### Cambio 4 — help= en combos default portfolio (AC4)

En el loop `for env_key in ("paper", "backtest"):`, reemplazar el `st.selectbox` existente agregando `help=`:

```python
chosen = st.selectbox(
    f"{env_key.title()} default portfolio",
    options,
    index=(...),
    key=f"default_{env_key}",
    help=(
        "Pre-select which portfolio profile is used by default when starting a "
        f"{'paper trading' if env_key == 'paper' else 'backtest'} run. "
        "To populate this list, save a portfolio profile using the Profile editor on the left."
    ),
)
```

El `st.caption(...)` que ya existe debajo del selectbox puede **eliminarse** ya que el `help=` cubre esa información, o conservarse. Preferir eliminar para no duplicar.

### Cambio 5 — help= en multiselect Universe (AC5)

En `st.multiselect("Universe (portfolio profiles only)", ...)`:

```python
universe = st.multiselect(
    "Universe (portfolio profiles only)",
    SUPPORTED_UNIVERSE_SYMBOLS,
    default=allowed_universe_default,
    help=(
        "Symbols included in this portfolio profile. "
        "Applies only when Profile kind = 'portfolio'. "
        "Affects which assets are traded in backtesting and paper trading runs that use this profile."
    ),
)
```

### Cambio 6 — Caption en sección Profile JSON (AC6)

Agregar un `st.caption` inmediatamente después del `st.text_area("Profile JSON", ...)`:

```python
st.caption(
    "The JSON above is the source of truth for the profile. "
    "Changes made in the Universe selector below are merged into this JSON when you click 'Save profile' — "
    "they do not replace existing keys."
)
```

---

## Archivos a modificar

| Archivo | Tipo de cambio |
|---|---|
| `apps/streamlit/views/configuration.py` | Añadir `help=` en 4 widgets, añadir 2 `st.caption`, eliminar 1 `st.caption` redundante |

---

## Orden de implementación

1. Abrir `apps/streamlit/views/configuration.py`
2. Aplicar Cambio 1 (caption tab LLM Settings)
3. Aplicar Cambio 2 (help= Provider)
4. Aplicar Cambio 3 (help= Model name)
5. Aplicar Cambio 4 (help= default portfolio combos + eliminar caption redundante)
6. Aplicar Cambio 5 (help= Universe multiselect)
7. Aplicar Cambio 6 (caption Profile JSON)
8. Verificar sintaxis + tests

---

## Validación post-implementación

```bash
# 1. Syntax check
/home/azureuser/repos/projects/QuantAgent/.venv/bin/python -m compileall -q apps/streamlit/views/configuration.py

# 2. Import check
cd /home/azureuser/repos/projects/QuantAgent && \
  PYTHONPATH=. /home/azureuser/repos/projects/QuantAgent/.venv/bin/python \
  -c "import apps.streamlit.views.configuration; print('OK')"

# 3. Test suite (regression)
/home/azureuser/repos/projects/QuantAgent/.venv/bin/python -m pytest tests/ -x -q --timeout=30 2>&1 | tail -10
```

La validación visual (que los tooltips aparezcan en el navegador) es responsabilidad de la fase tester.

---

## Riesgos

| Riesgo | Probabilidad | Mitigación |
|---|---|---|
| Texto de help demasiado largo en tooltips Streamlit | Baja | Streamlit no trunca tooltips, pero mantener mensajes ≤ 2 líneas para UX |
| Caption redundante no eliminado genera texto duplicado | Baja | Verificar visualmente que no queden dos textos equivalentes en el mismo widget |
| Cambio en widget key al agregar `help=` | No aplica | `help=` no afecta el key del widget |
| Tests de `configuration.py` que fallan con nueva firma | Muy baja | Los tests existentes no testean el parámetro `help=` |

---

## Estimación de complejidad

**Muy baja.** 6 cambios aditivos en un solo archivo, todos en la capa de presentación. Cero impacto en lógica, modelos o persistencia.
