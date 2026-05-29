# QuantAgent-kkj.5 — AC: Agregar help contextual en Configuration

**Issue:** QuantAgent-kkj.5  
**Requirements:** [QuantAgent-kkj.5-RQ-configuration-contextual-help.md](../01_requirements/QuantAgent-kkj.5-RQ-configuration-contextual-help.md)

---

## AC1 — Caption en tab LLM Settings

**Given** el operador abre la pestaña Configuration y navega al tab "LLM Settings"  
**When** la vista se renderiza  
**Then** aparece un `st.caption` (texto gris descriptivo) que indica que los parámetros de provider/model aplican exclusivamente a LLM-based strategies y no afectan a strategies determinísticas (ej. RSI, 52-week high)

**Cómo verificar:** Inspeccionar `configuration.py` — debe existir un `st.caption(...)` en el bloque `with tab_llm:` con texto que mencione "LLM-based strategies" o equivalente.

---

## AC2 — Tooltip en selector Provider

**Given** el operador ve el selector "Provider" en el tab LLM Settings  
**When** pasa el cursor sobre el ícono "?" del widget  
**Then** aparece un tooltip con descripción del provider (ej. qué API usa)

**Cómo verificar:** En `configuration.py`, el `st.selectbox("Provider", ...)` debe incluir `help="..."` con texto descriptivo no vacío.

---

## AC3 — Tooltip en input Model name

**Given** el operador ve el campo "Model name"  
**When** pasa el cursor sobre el ícono "?" del widget  
**Then** aparece un tooltip con ejemplos de nombres válidos por provider

**Cómo verificar:** En `configuration.py`, el `st.text_input("Model name", ...)` debe incluir `help="..."` con texto que contenga al menos un ejemplo de nombre de modelo.

---

## AC4 — Tooltip en combos de default portfolio

**Given** el operador ve los selectores "Paper default portfolio" y "Backtest default portfolio"  
**When** pasa el cursor sobre el ícono "?" de cada selector  
**Then** aparece un tooltip que explica:
- Qué hace el default (preselecciona el portfolio profile al iniciar una corrida en ese entorno)
- Cómo se popula la lista (guardando un portfolio profile desde el editor de la izquierda)

**Cómo verificar:** En `configuration.py`, ambos `st.selectbox(f"{env_key.title()} default portfolio", ...)` deben incluir `help="..."` con texto que mencione cómo crear profiles y qué implica el default.

---

## AC5 — Tooltip en selector Universe

**Given** el operador ve el multiselect "Universe (portfolio profiles only)"  
**When** pasa el cursor sobre el ícono "?" del widget  
**Then** aparece un tooltip que aclara:
- Solo activo cuando kind = "portfolio"
- Afecta qué símbolos se incluyen en backtesting y paper trading que usen ese profile

**Cómo verificar:** En `configuration.py`, el `st.multiselect("Universe (portfolio profiles only)", ...)` debe incluir `help="..."` con texto que mencione cuándo aplica y qué operaciones afecta.

---

## AC6 — Caption en sección Profile JSON

**Given** el operador ve el textarea "Profile JSON"  
**When** la vista se renderiza  
**Then** aparece un `st.caption` que aclara que el JSON del textarea es la fuente de verdad y que los cambios en el universe multiselect se aplican al JSON al momento de guardar (no reemplaza el form — lo complementa)

**Cómo verificar:** En `configuration.py`, debe existir un `st.caption(...)` cerca del `st.text_area("Profile JSON", ...)` con texto que aclare el comportamiento de merge JSON/form.

---

## AC7 — No regresión en lógica existente

**Given** se aplican los cambios de help text  
**When** se ejecuta el test suite  
**Then** todos los tests existentes pasan sin modificaciones

**Cómo verificar:**
```bash
/home/azureuser/repos/projects/QuantAgent/.venv/bin/python -m pytest tests/ -x -q --timeout=30 2>&1 | tail -10
```

---

## AC8 — Sintaxis Python correcta

**Given** se edita `configuration.py`  
**When** se compila el módulo  
**Then** no hay errores de sintaxis

**Cómo verificar:**
```bash
/home/azureuser/repos/projects/QuantAgent/.venv/bin/python -m compileall -q apps/streamlit/views/configuration.py
```
