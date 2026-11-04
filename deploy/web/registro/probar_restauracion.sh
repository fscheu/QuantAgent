#!/usr/bin/env bash
# Restaura el último dump en una base temporal y cuenta filas de las 4 tablas del registro.
# Modo Docker: QUANTAGENT_REGISTRO_CONTAINER=quantagent-dev-db (todo corre dentro del contenedor).
# Modo host: usa createdb/pg_restore/psql/dropdb con las variables PG* del entorno.
set -euo pipefail

backup_dir="${QUANTAGENT_REGISTRO_BACKUP_DIR:-/var/backups/quantagent-registro}"
latest="$(find "$backup_dir" -maxdepth 1 -type f -name 'registro-*.dump' -printf '%T@ %p\n' | sort -rn | head -n1 | cut -d' ' -f2-)"
if [[ -z "$latest" ]]; then echo "No hay dumps en $backup_dir" >&2; exit 2; fi
container="${QUANTAGENT_REGISTRO_CONTAINER:-}"
user="${QUANTAGENT_REGISTRO_PGUSER:-postgres}"
db="quantagent_registro_restore_$(date +%s)_$$"
if [[ -n "$container" ]]; then
  run() { docker exec -i "$container" "$@"; }
  pg=(--username="$user")
else
  run() { "$@"; }
  pg=()
fi
cleanup() { run dropdb "${pg[@]}" --if-exists "$db"; }
trap cleanup EXIT
run createdb "${pg[@]}" "$db"
run pg_restore "${pg[@]}" --no-owner --dbname="$db" < "$latest"
for table in hypotheses hypothesis_events hypothesis_attempts cycles; do
  run psql "${pg[@]}" --dbname="$db" --tuples-only --no-align --command="SELECT '$table', count(*) FROM public.$table;"
done
