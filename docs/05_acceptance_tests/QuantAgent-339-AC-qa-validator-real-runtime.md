# QuantAgent-339 — Acceptance criteria: QA validator on real deployed runtime

## AC1 — Target correctness
**Given** un deploy QA exitoso desde `main`
**When** corre el post-deploy validator
**Then** la validación apunta al runtime QA recién desplegado y no a un target distinto o stale

## AC2 — Canonical artifacts
**Given** el validator finaliza
**When** Tech Lead inspecciona el run
**Then** existen al menos `result.json` y `run-report.md`, más evidencia suficiente para diagnóstico funcional

## AC3 — Paper trading surface covered
**Given** la app QA está accesible
**When** el validator inspecciona la UI relevante para M2
**Then** valida la superficie mínima de paper trading, no sólo un healthcheck técnico

## AC4 — Empty-data behavior
**Given** la app está sana pero todavía no hay actividad de paper trading
**When** corre el validator
**Then** el resultado puede ser `SUCCESS` o `PARTIAL` justificado, pero no `FAIL` por ausencia normal de datos

## AC5 — Workflow interpretation
**Given** el validator devuelve un veredicto canónico
**When** el workflow y Hermes consumen el resultado
**Then** el outcome distingue correctamente `SUCCESS`, `PARTIAL`, `FAIL` y `BLOCKED` para routing y reporting

## AC6 — Drift detection
**Given** existe una discrepancia entre deploy target y validator target
**When** el run se evalúa
**Then** el resultado clasifica explícitamente el caso como drift/bloqueo observable y deja evidencia accionable
