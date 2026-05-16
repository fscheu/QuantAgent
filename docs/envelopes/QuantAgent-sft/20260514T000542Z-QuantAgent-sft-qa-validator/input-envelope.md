# Input envelope

- issue: `QuantAgent-sft`
- run_id: `20260514T000542Z-QuantAgent-sft-qa-validator`
- modalidad: validación manual post-deploy, browser-driven
- target preferido: `http://127.0.0.1:8501`
- target efectivo de validación browser: `http://127.0.0.1:8501/` vía CDP, porque `browser_navigate` bloqueó navegación directa a loopback
- healthcheck observado: `http://127.0.0.1:8501/_stcore/health` -> `200 OK`, body `ok`
- repo: `/home/azureuser/repos/projects/QuantAgent`
- branch esperada: `main`
- HEAD observado: `76dfa0bc84cc02938984190e1bc783d32abe488f`
- merge commit de referencia presente en historia: `2f94af25432ec8828dbd3f311aa48e6419e2bca1` (`yes`)
- objetivo: validar lo observable en UI Streamlit y clasificar honestamente el resultado sin inventar datos

## Documentos revisados

1. `docs/05_acceptance_tests/QuantAgent-sft-AC-paper-runtime-hardening.md`
2. `docs/01_requirements/QuantAgent-sft-RQ-paper-runtime-hardening.md`
3. `docs/03_design/QuantAgent-sft-DS-paper-runtime-hardening.md`
4. `docs/06_implementation/QuantAgent-app-IM-qa-streamlit-cutover.md`
5. `docs/user-manual/paper-trading-automation.md` (existe)

## Oráculos de validación usados

- liveness del runtime Streamlit en `8501/_stcore/health`
- carga real de la UI en browser
- snapshots de pestañas `Dashboard`, `Paper Trading`, `Orders & Positions`, `Logs`
- consola del browser
- consistencia contra AC1..AC6 del ticket

## Restricciones encontradas al iniciar

- la navegación browser a `127.0.0.1` fue rechazada por la capa estándar del browser tool; se resolvió usando CDP sobre una tab ya abierta
- la validación quedó acotada a lo observable en UI/browser y health endpoint local
