# Corrida en la nube (carril paralelo del lote)

Sos el implementador de UN ticket de QuantAgent en una sesión cloud. No hay humano: ante una duda real, entrega de bloqueo.
Te revisa el PM en la VM: lee el diff entero y corre tus comandos.

1. `git fetch origin && git checkout -b loop/<ID> origin/lote/fuera-de-muestra`.
2. Leé el ticket: `python3 -c "import json;[print(i['description'],*[c['text'] for c in i.get('comments') or []],sep='\n\n') for i in map(json.loads,open('.beads/issues.jsonl')) if i['id']=='<ID>']"`. Manda el comentario más nuevo. Criterio de aceptación = el del ticket.
3. Entorno: `pip install -e . pytest pytest-xdist` y `export DATABASE_URL=sqlite:///$PWD/nube-test.db`. Sin Postgres. Si un test necesita Postgres, es bloqueo.
4. Solo lo de "Cambio requerido", solo en "Archivos relevantes". Objetivo ≤100 líneas; si estimás más, no escribas código: bloqueo pidiendo partición.
5. No toques `.beads/`, `PLAN-CONTINUACION.md`, `docs/loop/REVIEW-LOG.md` ni `main`. Nunca `--force`. No leas datos desde 2019-01-01 (no hay datos reales en la nube: usá fixtures sintéticos).
6. Corré la línea "Tests a correr:" del ticket más los tests que tocaste: 0 failed.
7. Commit `<tipo>(<ID>): <qué>`, push a `loop/<ID>`, y `gh pr create --base lote/fuera-de-muestra --label nube` (NO uses el label `loop`) con el cuerpo de `.github/pull_request_template.md`. En "Probar" poné la línea `~/repos/projects/QuantAgent/scripts/loop/try.sh loop/<ID> -- '<comando>'`.
8. Bloqueo: rama `loop/<ID>` con solo `docs/loop/bloqueos/<ID>.md` (≤40 líneas: qué intentaste, qué falló, qué destraba), PR con labels `nube,bloqueo`.
9. Terminá con las 3 líneas Cambió / Decidir / Riesgo y la URL del PR.
