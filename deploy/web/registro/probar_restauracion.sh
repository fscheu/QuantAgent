#!/usr/bin/env bash
set -euo pipefail

backup_dir="${QUANTAGENT_REGISTRO_BACKUP_DIR:-/var/backups/quantagent-registro}"
latest="$(find "$backup_dir" -maxdepth 1 -type f -name 'registro-*.dump' -printf '%T@ %p\n' | sort -rn | head -n1 | cut -d' ' -f2-)"
if [[ -z "$latest" ]]; then echo "No hay dumps en $backup_dir" >&2; exit 2; fi
db="quantagent_registro_restore_$(date +%s)_$$"
cleanup() { dropdb --if-exists "$db"; }
trap cleanup EXIT
createdb "$db"
pg_restore --no-owner --dbname="$db" "$latest"
for table in hypotheses hypothesis_events hypothesis_attempts cycles; do
  psql --dbname="$db" --tuples-only --no-align --command="SELECT '$table', count(*) FROM public.$table;"
done
