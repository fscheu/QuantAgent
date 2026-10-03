#!/usr/bin/env bash
# Smoke run of the deterministic backtest CLI against an isolated, throwaway SQLite DB.
# Usage: scripts/loop/smoke.sh [strategy] [fixture] [extra backtest-run args...]
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
export DATABASE_URL="sqlite:///$TMP/smoke.db"
# The engine builds an LLM client even for deterministic strategies (QuantAgent-fdi); the key is never used.
export OPENAI_API_KEY="${OPENAI_API_KEY:-dummy-not-used}"
python -c "from quantagent.database import init_db; init_db()"
python -m quantagent.cli backtest run --strategy "${1:-rsi}" --fixture "${2:-spy-smoke}" "${@:3}"
