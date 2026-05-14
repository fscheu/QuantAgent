# QuantAgent-339 — Planning: QA validator on real deployed runtime

## Objective
Pasar de un PoC post-deploy a una validación QA funcional y operativamente útil para M2.

## Dependencies
- Baseline de deploy QA en `.github/workflows/main-ci-deploy.yml`
- PoC validado en `QuantAgent-vje`
- Runtime/UI estable por `QuantAgent-s62` y `QuantAgent-sft`

## Task breakdown

### 1. Contract alignment
- Documentar explícitamente el contrato esperado del validator para QuantAgent.
- Definir campos/resultados mínimos que Hermes y Tech Lead necesitan consumir.

### 2. Target verification
- Asegurar que el workflow y la config del validator apunten al runtime QA real.
- Cubrir detección de drift entre deploy target, Docker target y validator target.

### 3. Functional coverage
- Expandir la validación desde health/runtime a la superficie visible de paper trading relevante para M2.
- Permitir empty state sano sin clasificarlo como error funcional.

### 4. Artifact durability
- Garantizar persistencia y lectura de `result.json`, `run-report.md` y evidencia auxiliar.
- Verificar que el workflow eleve el veredicto correcto al webhook de Hermes.

### 5. Verification
- Smoke del workflow/parsing del resultado.
- Ejecución post-deploy en QA con revisión manual de artefactos.

## Risks
- Dependencia externa en `qa-validator-poc` deja parte del cambio fuera del repo.
- El target público puede introducir auth/network noise no atribuible al código.
- Un veredicto `PARTIAL` mal interpretado puede bloquear integración sin necesidad.

## Recommended routing
1. `autodev-implementer`
2. `autodev-tester` si hay parsing/tests de contrato viables
3. Tech Lead integration sólo cuando exista evidencia post-deploy real y artefactos canónicos
