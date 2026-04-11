#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$PROJECT_ROOT/.run"
IMPORT_PID_FILE="$RUN_DIR/import_service.pid"
QUERY_PID_FILE="$RUN_DIR/query_service.pid"
SILENT="${1:-}"

log() {
  if [[ "$SILENT" != "--silent" ]]; then
    echo "$1"
  fi
}

kill_by_pid_file() {
  local name="$1"
  local pid_file="$2"

  if [[ ! -f "$pid_file" ]]; then
    return 0
  fi

  local pid
  pid="$(cat "$pid_file" 2>/dev/null || true)"
  if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
    kill "$pid" 2>/dev/null || true
    sleep 1
    if kill -0 "$pid" 2>/dev/null; then
      kill -9 "$pid" 2>/dev/null || true
    fi
    log "[ok] stopped $name (pid=$pid)"
  fi

  rm -f "$pid_file"
}

kill_by_pattern() {
  local name="$1"
  local pattern="$2"

  local pids
  pids="$(pgrep -f "$pattern" || true)"
  if [[ -n "$pids" ]]; then
    pkill -f "$pattern" || true
    log "[ok] stopped $name by pattern"
  fi
}

kill_by_port() {
  local name="$1"
  local port="$2"

  if command -v fuser >/dev/null 2>&1; then
    if fuser "${port}/tcp" >/dev/null 2>&1; then
      fuser -k "${port}/tcp" >/dev/null 2>&1 || true
      log "[ok] stopped $name by port $port"
    fi
  fi
}

kill_by_pid_file "import-service" "$IMPORT_PID_FILE"
kill_by_pid_file "query-service" "$QUERY_PID_FILE"

kill_by_pattern "import-service" "app.import_process.api.file_import_service"
kill_by_pattern "query-service" "app.query_process.api.query_service"
kill_by_pattern "import-service" "app/import_process/api/file_import_service.py"
kill_by_pattern "query-service" "app/query_process/api/query_service.py"
kill_by_port "import-service" 8000
kill_by_port "query-service" 8001

log "insightvault API processes stopped (docker containers kept running)"
