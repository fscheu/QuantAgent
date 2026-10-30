#!/usr/bin/env bash
set -euo pipefail

DRY_RUN="${DRY_RUN:-1}"
PAPER_DIR="/home/azureuser/paper-ops/QuantAgent"
VENV="/home/azureuser/paper-ops/venv"
DATA_DIR="${PAPER_DIR}/local-data"
MASTER_DB="/home/azureuser/web-ops/maestro.db"
USER_UNITS="/home/azureuser/.config/systemd/user"
DATE="$(date -u +%Y%m%dT%H%M%SZ)"
BACKUP_DIR="/home/azureuser/paper-ops/backups/${DATE}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"

run() {
    if [[ "${DRY_RUN}" == "1" ]]; then
        printf '+ '
        printf '%q ' "$@"
        printf '\n'
    else
        "$@"
    fi
}

printf '1. Back up paper and master databases to %s\n' "${BACKUP_DIR}"
run mkdir -p "${BACKUP_DIR}"
for portfolio in cartera-nucleo-curva cartera-transversal; do
    run sqlite3 "${DATA_DIR}/${portfolio}.db" ".backup '${BACKUP_DIR}/${portfolio}.db'"
done
run sqlite3 "${MASTER_DB}" ".backup '${BACKUP_DIR}/maestro.db'"

printf '2. Build the web application\n'
run npm --prefix "${PAPER_DIR}/web" ci
run npm --prefix "${PAPER_DIR}/web" run build

printf '3. Apply paper database migrations\n'
for portfolio in cartera-nucleo-curva cartera-transversal; do
    run "${VENV}/bin/quantagent" api migrate-paper-db --database-url "sqlite:////home/azureuser/paper-ops/QuantAgent/local-data/${portfolio}.db" --apply
done

printf '4. Install API unit and digest snapshot drop-in\n'
run mkdir -p "${USER_UNITS}/paper-digest@.service.d"
run install -m 0644 "${SCRIPT_DIR}/quantagent-api.service" "${USER_UNITS}/quantagent-api.service"
run install -m 0644 "${SCRIPT_DIR}/paper-digest-snapshot.conf" "${USER_UNITS}/paper-digest@.service.d/snapshot.conf"

printf '5. Reload user units and enable the API\n'
run systemctl --user daemon-reload
run systemctl --user enable --now quantagent-api.service
