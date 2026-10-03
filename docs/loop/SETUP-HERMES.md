# Instalar el loop nocturno en Hermes

Paso 0.10 de `PLAN-CONTINUACION.md`. Toca `~/.hermes`: lo ejecuta Fede. Son 3 bloques para copiar y pegar en la VM.

## Por qué son dos jobs

Hermes corta los scripts de cron a los **120 segundos** (`_DEFAULT_SCRIPT_TIMEOUT` en
`~/.hermes/hermes-agent/cron/scheduler.py`). Una corrida del loop dura hasta 2 horas. En vez de subir ese
límite global en la config de Hermes, el loop usa dos jobs cortos:

| Job | Hora (ART) | Qué hace | Duración |
|---|---|---|---|
| Lanzador | 23:00 dom–jue | Gate de revisión. Si espera, avisa por qué. Si pasa, lanza el loop desacoplado | < 30 s |
| Reporte | 02:30 lun–vie | Manda las 3 líneas de la entrega y la URL del PR, o avisa si falló | < 1 s |

Los dos corren la versión de los scripts que está en `origin/main`, no la del checkout principal,
porque el deploy hace `git reset --hard` de ese checkout en cada merge.

## 1. Instalar los dos shims

```bash
cat > ~/.hermes/scripts/quantagent_loop_launch.sh <<'SH'
#!/usr/bin/env bash
git -C /home/azureuser/repos/projects/QuantAgent fetch -q origin main 2>/dev/null
git -C /home/azureuser/repos/projects/QuantAgent show origin/main:scripts/loop/hermes_launch.sh | bash
SH
cat > ~/.hermes/scripts/quantagent_loop_report.sh <<'SH'
#!/usr/bin/env bash
git -C /home/azureuser/repos/projects/QuantAgent show origin/main:scripts/loop/hermes_report.sh | bash
SH
chmod +x ~/.hermes/scripts/quantagent_loop_*.sh
```

## 2. Probar el lanzador a mano

```bash
bash ~/.hermes/scripts/quantagent_loop_launch.sh
```

Con una entrega sin revisar tiene que imprimir `⏸ esperando revisión: <url>` y terminar en segundos.

## 3. Crear los jobs

```bash
hermes cron create "0 23 * * 0-4" --name "QuantAgent loop — lanzador" \
  --script quantagent_loop_launch.sh --no-agent \
  --deliver telegram:-1003401012237:43
hermes cron create "30 2 * * 1-5" --name "QuantAgent loop — reporte" \
  --script quantagent_loop_report.sh --no-agent \
  --deliver telegram:-1003401012237:43
hermes cron list | grep -A4 "QuantAgent loop"
```

Verificar en `hermes cron list` que "Next run" muestre 23:00 y 02:30 con `-03:00`.
El topic `43` de Telegram es el mismo de "Ideas de Proyectos"; cambiarlo si se quiere otro.

## Base de tests del loop

El wrapper exporta `DATABASE_URL` hacia una base propia en el Postgres de desarrollo y corre
`alembic upgrade head` antes de cada corrida. Sin eso, unos 25 tests que necesitan Postgres fallan en el
worktree. Ya está creada (2026-10-03). Para recrearla:

```bash
docker exec quantagent-dev-db psql -U postgres -c "CREATE ROLE loop_test LOGIN PASSWORD 'loop_test'"
docker exec quantagent-dev-db psql -U postgres -c "CREATE DATABASE quantagent_loop_test OWNER loop_test"
```

## Operación

| Quiero | Cómo |
|---|---|
| Pausar el loop | Crear `loop/PAUSE` en `main` desde GitHub mobile (cualquier contenido) |
| Reanudar | Borrar `loop/PAUSE` |
| Ver logs de una corrida | `ls -t ~/.local/state/quantagent-loop/` |
| Cortar una corrida en curso | `kill $(cat ~/.local/state/quantagent-loop/running.pid)` |
| Desinstalar | `hermes cron list`, después `hermes cron remove <id>` para los dos jobs |
