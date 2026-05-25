# QuantAgent-339 — Requirements: QA validator on real deployed runtime

## Objective
Convertir el post-deploy QA validator actual en una validación útil contra el runtime QA realmente desplegado, con artefactos durables y oráculos alineados a la UI de paper trading.

## Context
El workflow `main-ci-deploy.yml` ya:
- despliega Streamlit QA en `127.0.0.1:8501` / `qa.fedes.dev`
- ejecuta el runner interno del skill `autodev-qa-validator`
- lee `result.json`
- envía webhook a Hermes

La validación actual existe, pero todavía opera como PoC y su cobertura funcional para M2 es limitada.

## Scope
1. Formalizar el contrato mínimo del validator para QuantAgent en QA.
2. Validar contra el runtime desplegado real, no sólo contra un target local abstracto.
3. Exigir artefactos durables suficientes para diagnóstico:
   - `result.json`
   - `run-report.md`
   - hallazgos/browser evidence
4. Cubrir al menos el flujo visible de la superficie de paper trading relevante para M2.
5. Hacer que el workflow interprete correctamente el veredicto del validator.

## Out of scope
- Automatización completa de autenticación Cloudflare Access.
- QA de todas las pantallas de Streamlit.
- Reemplazar CI o health checks existentes.
- Validación contra producción.
- Rediseñar el runner del skill más allá de lo necesario para este contrato.

## Constraints
- Mantener el deploy target de QA en Streamlit sobre puerto 8501.
- No depender de secretos impresos ni de credenciales visibles en logs.
- Reusar el patrón de webhook `deploy_finished` ya integrado con Hermes.
- La ausencia de datos de trading no debe falsear una falla funcional del dashboard.

## Edge cases
- Deploy sano pero validator sin artefactos canónicos.
- Validator PASS técnico sobre healthcheck, pero sin validar la UI de paper trading.
- Resultado `PARTIAL` por datos vacíos esperables, con browser/docs correctos.
- Drift entre target desplegado y target inspeccionado por el validator.

## Definition of done
- Existe un contrato explícito y mínimo para el validator de QA en QuantAgent.
- El validator corre contra el runtime QA real y produce artefactos durables.
- El workflow puede distinguir PASS / PARTIAL / FAIL de forma accionable.
- La validación cubre la superficie user-facing mínima de paper trading para M2.
