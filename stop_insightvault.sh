#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$PROJECT_ROOT/.run"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
FRONTEND_BIN="$FRONTEND_DIR/node_modules/.bin/vite"
FRONTEND_PID_FILE="$RUN_DIR/frontend.pid"
SILENT="${1:-}"

log() {
  if [[ "$SILENT" != "--silent" ]]; then
    echo "$1"
  fi
}

cd "$PROJECT_ROOT"

echo "stopping insightvault services..."

if [[ -f "$FRONTEND_PID_FILE" ]]; then
  pid=""
  pid="$(cat "$FRONTEND_PID_FILE" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    sleep 1
    kill -9 "$pid" 2>/dev/null || true
    log "[ok] stopped frontend (pid=$pid)"
  fi
  rm -f "$FRONTEND_PID_FILE"
fi

if command -v pgrep >/dev/null 2>&1; then
  while IFS= read -r pid; do
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
      log "[ok] stopped lingering frontend (pid=$pid)"
    fi
  done < <(pgrep -f "$FRONTEND_BIN" 2>/dev/null || true)
fi

echo "stopping docker containers..."
docker compose down >/dev/null 2>&1 || true

log "insightvault stopped"
