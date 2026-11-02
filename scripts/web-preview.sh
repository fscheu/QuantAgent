#!/usr/bin/env bash
set -euo pipefail

TARGET="${1:?Uso: $0 <rama|PR>}"
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

PREVIEW_DIR="${QUANTAGENT_PREVIEW_DIR:-/tmp/quantagent-web-preview}"
WORKTREE_DIR="${QUANTAGENT_PREVIEW_WORKTREE:-${PREVIEW_DIR}/worktree}"
DATA_DIR="${QUANTAGENT_PREVIEW_DATA:-${PREVIEW_DIR}/data}"
SOURCE_MAESTRO="${QUANTAGENT_PREVIEW_SOURCE_MAESTRO:-${SOURCE_MAESTRO:-/home/azureuser/web-ops/maestro.db}}"

COMMIT=""
if [[ "$TARGET" =~ ^#?([0-9]+)$ || "$TARGET" =~ ^pr/([0-9]+)$ || "$TARGET" =~ ^pull/([0-9]+)/head$ ]]; then
    PR="${BASH_REMATCH[1]}"
    if git -C "$REPO_DIR" rev-parse --verify --quiet "refs/pull/${PR}/head" >/dev/null; then
        COMMIT="$(git -C "$REPO_DIR" rev-parse "refs/pull/${PR}/head")"
    else
        git -C "$REPO_DIR" fetch -q origin "pull/${PR}/head" 2>/dev/null || true
        COMMIT="$(git -C "$REPO_DIR" rev-parse --verify --quiet FETCH_HEAD || true)"
    fi
fi

if [[ -z "$COMMIT" ]]; then
    if git -C "$REPO_DIR" rev-parse --verify --quiet "origin/${TARGET}" >/dev/null; then
        COMMIT="$(git -C "$REPO_DIR" rev-parse "origin/${TARGET}")"
    elif git -C "$REPO_DIR" rev-parse --verify --quiet "${TARGET}" >/dev/null; then
        COMMIT="$(git -C "$REPO_DIR" rev-parse "${TARGET}")"
    else
        git -C "$REPO_DIR" fetch -q origin "${TARGET}" 2>/dev/null || true
        if git -C "$REPO_DIR" rev-parse --verify --quiet "origin/${TARGET}" >/dev/null; then
            COMMIT="$(git -C "$REPO_DIR" rev-parse "origin/${TARGET}")"
        elif git -C "$REPO_DIR" rev-parse --verify --quiet "${TARGET}" >/dev/null; then
            COMMIT="$(git -C "$REPO_DIR" rev-parse "${TARGET}")"
        fi
    fi
fi

if [[ -z "$COMMIT" ]]; then
    echo "Error: no se pudo resolver la rama o PR '${TARGET}'" >&2
    exit 1
fi

mkdir -p "$PREVIEW_DIR"
if [[ -d "$WORKTREE_DIR" ]]; then
    git -C "$REPO_DIR" worktree remove --force "$WORKTREE_DIR" 2>/dev/null || true
    rm -rf "$WORKTREE_DIR"
fi
git -C "$REPO_DIR" worktree add --force --detach "$WORKTREE_DIR" "$COMMIT" >/dev/null

if [[ -d "${WORKTREE_DIR}/web" && -f "${WORKTREE_DIR}/web/package.json" && "${QUANTAGENT_PREVIEW_SKIP_BUILD:-0}" != "1" ]]; then
    npm --prefix "${WORKTREE_DIR}/web" ci >/dev/null 2>&1
    npm --prefix "${WORKTREE_DIR}/web" run build >/dev/null 2>&1
fi

python3 "${SCRIPT_DIR}/web_preview_data.py" "$DATA_DIR" --source-maestro "$SOURCE_MAESTRO"

printf "QUANTAGENT_DATABASE_URL=sqlite:///%s/maestro.db\nQUANTAGENT_PREVIEW_BRANCH=%s\nQUANTAGENT_WEB_DIST=%s/web/dist\nPORT=8012\n" \
    "$DATA_DIR" "$TARGET" "$WORKTREE_DIR" > "${PREVIEW_DIR}/preview.env"

echo "Preview preparado para ${TARGET} en ${PREVIEW_DIR} (commit ${COMMIT:0:8})"
