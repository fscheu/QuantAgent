# Bloqueo QuantAgent-dpr3.1.7

Intenté limpiar las referencias vigentes de `.env.example`, `.vscode/launch.json`, `AGENTS.md` y
`scripts/bootstrap_qa_minimal.py`. El test dirigido `python -m pytest -q tests/test_main_ci_deploy_workflow.py`
pasó (1/1), pero `scripts/loop/smoke.sh` falló al importar `vectorbt`: Numba no puede cachear
`set_seed_nb` desde el `site-packages` del entorno.

El escaneo también encuentra menciones en comentarios antiguos de `.github/workflows/main-ci-deploy.yml`
y en el artefacto binario versionado `.coverage`. Se requiere decidir si se borran esos comentarios y
el artefacto para satisfacer el grep literal. Las menciones de `CHANGELOG.md`, `PLAN-30-DIAS.md`,
`PLAN-CONTINUACION.md` y `TEST_REPORT_UNIVERSE_MANAGEMENT.md` son históricas.

Destraba: corregir el entorno de smoke y definir el tratamiento de los comentarios del workflow y
`.coverage`; después se puede completar la limpieza y repetir el escaneo.
