# QuantAgent-kkj.4 — AC: Separar pestaña Configuration en LLM Settings y Portfolio & Universe

**Issue:** QuantAgent-kkj.4  
**Requirements:** [QuantAgent-kkj.4-RQ-configuration-split-llm-portfolio.md](../01_requirements/QuantAgent-kkj.4-RQ-configuration-split-llm-portfolio.md)  
**Plan:** [QuantAgent-kkj.4-PL-configuration-split-llm-portfolio.md](../02_planning/QuantAgent-kkj.4-PL-configuration-split-llm-portfolio.md)

---

## AC1 — Dos secciones claramente separadas

**Given** el usuario abre la pestaña Configuration en la UI Streamlit  
**When** la vista carga  
**Then** aparecen dos tabs con labels "LLM Settings" y "Portfolio & Universe"

**Verificación:**
- [ ] `st.tabs(["LLM Settings", "Portfolio & Universe"])` existe en `configuration.py`
- [ ] El tab "LLM Settings" es visible y contiene la sección de model presets
- [ ] El tab "Portfolio & Universe" es visible y contiene los profile editors y los defaults

---

## AC2 — LLM Settings contiene solo configuración de inferencia

**Given** el usuario está en el tab "LLM Settings"  
**When** inspecciona el contenido  
**Then** ve: selector de preset, provider, model name, temperature slider, save as, save preset button, presets dataframe  
**And** NO ve: selectores de portfolio profile, risk profile, universe multiselect, profile JSON editor

**Verificación:**
- [ ] Todos los widgets de model presets (provider, model_name, temperature) están dentro de `with tab_llm:`
- [ ] No hay referencias a portfolio/risk/combined kinds ni a `_collect_profiles_from_db` dentro de `tab_llm`

---

## AC3 — Portfolio & Universe contiene solo configuración de portafolios

**Given** el usuario está en el tab "Portfolio & Universe"  
**When** inspecciona el contenido  
**Then** ve: profile editor (kind/name/JSON), universe multiselect, universe preview, save profile, profiles table, default portfolio selectors, default strategy selectors  
**And** NO ve: provider selector, model name input, temperature slider, model presets dataframe

**Verificación:**
- [ ] Todo el bloque `colL` (profile editor) y `colR` (defaults) está dentro de `with tab_portfolio:`
- [ ] No hay widgets de model presets dentro de `tab_portfolio`

---

## AC4 — Captions de ayuda en los selectores de default portfolio

**Given** el usuario está en el tab "Portfolio & Universe", sección "Defaults per environment"  
**When** ve los selectores "Paper default portfolio" y "Backtest default portfolio"  
**Then** cada selector tiene un caption que explica cómo crear/cargar un portfolio profile

**Verificación:**
- [ ] `st.caption(...)` con texto de ayuda existe inmediatamente después de cada `st.selectbox` de default portfolio
- [ ] El caption menciona cómo se crean los profiles (ej. "Create profiles using the Profile editor")

---

## AC5 — Universe selector tiene contexto claro

**Given** el usuario está en el tab "Portfolio & Universe" con kind = "portfolio"  
**When** ve el universe multiselect  
**Then** el label o caption indica que aplica solo a portfolio profiles

**Given** el usuario selecciona kind ≠ "portfolio"  
**When** inspecciona el area de universe  
**Then** ve un caption "Universe editing is available for portfolio profiles only." (comportamiento existente conservado)

**Verificación:**
- [ ] El comportamiento condicional del universe multiselect es idéntico al original
- [ ] El caption de "portfolio profiles only" sigue presente para kind ≠ portfolio

---

## AC6 — Sin regresión funcional

**Given** la reorganización fue aplicada  
**When** el tester ejecuta el test suite  
**Then** todos los tests existentes pasan

**When** el tester usa la UI (save profile, save preset, set defaults)  
**Then** todas las operaciones funcionan igual que antes de la reorganización

**Verificación:**
- [ ] `pytest tests/ -x -q --timeout=30` — sin nuevas fallas
- [ ] `python -m compileall -q apps/streamlit/views/configuration.py` — sin errores de sintaxis
- [ ] Funcionalidad de save profile: guarda en DB o session correctamente
- [ ] Funcionalidad de save preset: guarda en session state correctamente
- [ ] Selectores de default portfolio/strategy: actualizan session state correctamente

---

## Comandos de validación

```bash
# Syntax
/home/azureuser/repos/projects/QuantAgent/.venv/bin/python -m compileall -q apps/streamlit/views/configuration.py

# Tests de regresión
/home/azureuser/repos/projects/QuantAgent/.venv/bin/python -m pytest tests/ -x -q --timeout=30

# Import check
/home/azureuser/repos/projects/QuantAgent/.venv/bin/python -c "
import sys; sys.path.insert(0, '.')
# Verify tabs exist in source
src = open('apps/streamlit/views/configuration.py').read()
assert 'st.tabs' in src, 'st.tabs not found'
assert 'LLM Settings' in src, 'LLM Settings tab not found'
assert 'Portfolio & Universe' in src, 'Portfolio & Universe tab not found'
print('Structure check: PASS')
"
```
