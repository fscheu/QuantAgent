#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${QUANTAGENT_REGISTRO_ADMIN_URL:-}" ]]; then
  echo "Falta QUANTAGENT_REGISTRO_ADMIN_URL" >&2
  exit 2
fi
if ! command -v pg_dump >/dev/null 2>&1; then
  echo "Falta pg_dump; instalalo con: sudo apt install postgresql-client" >&2
  exit 2
fi

backup_dir="${QUANTAGENT_REGISTRO_BACKUP_DIR:-/var/backups/quantagent-registro}"
keep="${QUANTAGENT_REGISTRO_BACKUP_KEEP:-14}"
mkdir -p "$backup_dir"
dump="$backup_dir/registro-$(date +%Y%m%d-%H%M%S).dump"
pg_dump --format=custom --file="$dump" "$QUANTAGENT_REGISTRO_ADMIN_URL"
mapfile -t dumps < <(find "$backup_dir" -maxdepth 1 -type f -name 'registro-*.dump' -printf '%T@ %p\n' | sort -rn | awk -v keep="$keep" 'NR>keep {sub(/^[^ ]+ /, ""); print}')
if ((${#dumps[@]})); then rm -- "${dumps[@]}"; fi
echo "Backup creado: $dump"
