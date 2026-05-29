# QuantAgent-kkj.5 — RQ: Agregar help contextual en Configuration (tooltips y texto descriptivo)

**Issue:** QuantAgent-kkj.5  
**Type:** UX / UI enhancement  
**Parent:** QuantAgent-kkj (M2 Milestone)  
**Status:** open

---

## Problema

La pestaña Configuration de Streamlit (`apps/streamlit/views/configuration.py`) carece de texto descriptivo y ayuda contextual. El operador no puede entender qué hace cada campo, combo o selector sin consultar el user manual externo.

Puntos específicos con mayor necesidad de ayuda, identificados durante revisión funcional M2 (2026-05-25):

1. **Selector de provider/modelo LLM**: No queda claro qué afecta. Solo impacta LLM-based strategies; las deterministic (RSI, 52-week high) lo ignoran.
2. **Combos "Paper default portfolio" y "Backtest default portfolio"**: Aparecen vacíos sin explicación de cómo generarlos.
3. **Selector "Universe (for portfolio profiles only)"**: La aclaración existe pero no aclara en qué operación concreta aplica.
4. **Sección de carga de Profile JSON**: No está claro si el JSON reemplaza o complementa los campos del form.

**Nota de contexto:** QuantAgent-kkj.4 ya reorganizó la pestaña en dos tabs (`LLM Settings` y `Portfolio & Universe`). Este ticket agrega help text sobre esa estructura ya implementada.

---

## Cambio requerido

Agregar help contextual en `apps/streamlit/views/configuration.py`:

1. **Tab LLM Settings** — añadir `st.caption` bajo el header de la sección con una línea que aclare que estos parámetros aplican exclusivamente a LLM-based strategies.
2. **Selector provider** — añadir `help=` parameter con descripción concisa del rol del provider.
3. **Selector model_name** — añadir `help=` parameter explicando el nombre esperado (ej. `"gpt-4o-mini"`, `"claude-3-haiku-20240307"`).
4. **Combos "Paper default portfolio" y "Backtest default portfolio"** — añadir `help=` parameter que explique cómo se populan (desde profiles guardados) y qué implica seleccionar uno.
5. **Selector "Universe"** — añadir `help=` parameter que aclare cuándo aplica (solo portfolio profiles) y qué operaciones concretas afecta (backtesting y paper trading que usen ese profile).
6. **Sección Profile JSON (textarea)** — añadir `st.caption` que aclare que el JSON del textarea es la fuente de verdad; los cambios en el form (universe multiselect) se aplican al JSON antes de guardarlo.

---

## Archivos afectados

| Archivo | Cambio |
|---|---|
| `apps/streamlit/views/configuration.py` | Añadir `help=` en selectboxes/text_input críticos; añadir `st.caption` bajo secciones principales |

---

## Fuera de scope

- No cambiar lógica backend ni modelos.
- No rediseñar el layout de la pestaña (QuantAgent-kkj.4 ya lo hizo).
- No implementar onboarding wizard ni tours guiados.
- No agregar validaciones que hoy no existen.
- No modificar las funciones `_collect_profiles_from_db` / `_get_profile_json_from_db`.
