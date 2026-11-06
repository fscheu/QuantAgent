#!/usr/bin/env bash
# Backup del registro (pg_dump, formato custom).
# Modo Docker (por defecto): el registro de producción vive en el contenedor quantagent-prod-db (postgres:15), base
# quantagent_registro; el dump se hace con `docker exec <contenedor> pg_dump`: misma versión del servidor y sin
# instalar postgresql-client en el host. Variables (todas opcionales):
#   QUANTAGENT_REGISTRO_CONTAINER (default quantagent-prod-db), QUANTAGENT_REGISTRO_DB (default quantagent_registro),
#   QUANTAGENT_REGISTRO_PGUSER (usuario con permiso de dump; default postgres: ajustar al definido por el hilo del registro).
# quantagent-dev-db (5432) es solo desarrollo y tests: NO contiene el registro real.
# Modo host: QUANTAGENT_REGISTRO_CONTAINER="" y QUANTAGENT_REGISTRO_ADMIN_URL (requiere pg_dump en el host).
set -euo pipefail

backup_dir="${QUANTAGENT_REGISTRO_BACKUP_DIR:-/var/backups/quantagent-registro}"
keep="${QUANTAGENT_REGISTRO_BACKUP_KEEP:-14}"
container="${QUANTAGENT_REGISTRO_CONTAINER-quantagent-prod-db}"
if [[ -n "$container" ]]; then
  if ! command -v docker >/dev/null 2>&1; then
    echo "Falta docker (QUANTAGENT_REGISTRO_CONTAINER está definido)" >&2
    exit 2
  fi
elif [[ -z "${QUANTAGENT_REGISTRO_ADMIN_URL:-}" ]]; then
  echo "Falta QUANTAGENT_REGISTRO_ADMIN_URL (modo host, QUANTAGENT_REGISTRO_CONTAINER vacío); por defecto se usa el contenedor quantagent-prod-db" >&2
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
