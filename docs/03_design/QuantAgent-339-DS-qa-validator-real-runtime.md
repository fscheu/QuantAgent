# QuantAgent-339 — Design: QA validator on real deployed runtime

## Design level
STANDARD

## Current baseline
El repo ya tiene un pipeline útil:
- deploy QA a Streamlit en puerto 8501
- healthcheck `_stcore/health`
- ejecución de `qa-validator-poc`
- lectura de `result.json`
- webhook `deploy_finished` hacia Hermes

La mejora requerida no es inventar una fase nueva sino endurecer el contrato entre deploy, validator y Tech Lead.

## Proposed change
Fortalecer tres seams:

1. **Target seam**
   - el validator debe inspeccionar explícitamente el runtime QA real que acaba de desplegar el workflow
   - evitar drift entre target desplegado y target validado

2. **Artifact seam**
   - el validator debe producir artefactos canónicos consumibles por Tech Lead:
     - `result.json`
     - `run-report.md`
     - evidencia browser/findings

3. **Verdict seam**
   - el workflow debe interpretar `SUCCESS | PARTIAL | FAIL | BLOCKED` sin degradar `PARTIAL` útiles a ruido opaco

## Affected components
- `.github/workflows/main-ci-deploy.yml`
- documentación de aceptación/reporte dentro de `docs/envelopes/` cuando corresponda
- configuración de integración con `qa-validator-poc`
- opcionalmente tests o validaciones ligeras sobre parsing del resultado

## Technical decisions
- Mantener Streamlit/8501 como target QA canónico.
- Reusar el webhook actual en lugar de agregar un canal alternativo.
- Considerar “sin datos de trading” como estado funcional posible, no como FAIL por defecto.
- Tratar la integridad de artefactos como parte del contrato del validator, no como detalle opcional.

## External dependency boundary
`qa-validator-poc` sigue siendo dependencia externa. Este ticket debe dejar en QuantAgent el contrato de entrada/salida y la interpretación operativa del resultado; cualquier cambio externo debe ser el mínimo necesario para cumplir ese contrato.
