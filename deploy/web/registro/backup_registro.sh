#!/usr/bin/env bash
# Backup del registro (pg_dump, formato custom).
# Modo Docker (el Postgres de la VM corre en el contenedor quantagent-dev-db, postgres:15): define
#   QUANTAGENT_REGISTRO_CONTAINER=quantagent-dev-db [QUANTAGENT_REGISTRO_DB=quantagent_registro] [QUANTAGENT_REGISTRO_PGUSER=postgres]
# y el dump se hace con `docker exec <contenedor> pg_dump`: usa la misma versión del servidor y no requiere
# instalar postgresql-client en el host.
# Modo host: define QUANTAGENT_REGISTRO_ADMIN_URL (requiere pg_dump en el host).
set -euo pipefail

backup_dir="${QUANTAGENT_REGISTRO_BACKUP_DIR:-/var/backups/quantagent-registro}"
keep="${QUANTAGENT_REGISTRO_BACKUP_KEEP:-14}"
container="${QUANTAGENT_REGISTRO_CONTAINER:-}"
if [[ -n "$container" ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "Falta docker (QUANTAGENT_REGISTRO_CONTAINER está definido)" >&2
    exit 2
  fi
elif [[ -z "${QUANTAGENT_REGISTRO_ADMIN_URL:-}" ]]; then
  echo "Falta QUANTAGENT_REGISTRO_CONTAINER (modo Docker) o QUANTAGENT_REGISTRO_ADMIN_URL (modo host)" >&2
  exit 2
elif ! command -v pg_dump >/dev/null 2>&1; then
  echo "Falta pg_dump en el host; usá el modo Docker (QUANTAGENT_REGISTRO_CONTAINER) o instalá postgresql-client" >&2
  exit 2
fi

mkdir -p "$backup_dir"
dump="$backup_dir/registro-$(date +%Y%m%d-%H%M%S).dump"
if [[ -n "$container" ]]; then
  docker exec "$container" pg_dump --format=custom \
    --username="${QUANTAGENT_REGISTRO_PGUSER:-postgres}" "${QUANTAGENT_REGISTRO_DB:-quantagent_registro}" > "$dump"
else
  pg_dump --format=custom --file="$dump" "$QUANTAGENT_REGISTRO_ADMIN_URL"
fi
if [[ ! -s "$dump" ]]; then rm -f -- "$dump"; echo "El dump quedó vacío" >&2; exit 1; fi
mapfile -t dumps < <(find "$backup_dir" -maxdepth 1 -type f -name 'registro-*.dump' -printf '%T@ %p\n' | sort -rn | awk -v keep="$keep" 'NR>keep {sub(/^[^ ]+ /, ""); print}')
if ((${#dumps[@]})); then rm -- "${dumps[@]}"; fi
echo "Backup creado: $dump"
