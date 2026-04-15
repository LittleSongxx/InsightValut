#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$PROJECT_ROOT/.run"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
FRONTEND_BIN="$FRONTEND_DIR/node_modules/.bin/vite"
FRONTEND_PID_FILE="$RUN_DIR/frontend.pid"
SILENT=0
STOP_DOCKER=0

usage() {
  cat <<'EOF'
Usage: ./stop_insightvault.sh [--docker] [--silent] [--help]

Options:
  --docker   stop Docker containers with docker compose down
  --silent   suppress non-essential output
  --help     show this help message

Default behavior only cleans up local helper processes and keeps containers running.
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --docker|--all)
        STOP_DOCKER=1
        ;;
      --silent)
        SILENT=1
        ;;
      --help|-h)
        usage
        exit 0
        ;;
      *)
        echo "[error] unknown option: $1"
        usage
        exit 1
        ;;
    esac
    shift
  done
}

log() {
  if (( ! SILENT )); then
    echo "$1"
  fi
}

parse_args "$@"

cd "$PROJECT_ROOT"

log "stopping insightvault local helpers..."

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

if (( STOP_DOCKER )); then
  log "stopping docker containers..."
  docker compose down >/dev/null 2>&1 || true
  log "insightvault containers stopped"
else
  log "docker containers left running"
  log "use ./stop_insightvault.sh --docker to stop containers"
fi

log "insightvault stop completed"
