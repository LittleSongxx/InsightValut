#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$PROJECT_ROOT/.run"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
FRONTEND_BIN="$FRONTEND_DIR/node_modules/.bin/vite"
FRONTEND_PID_FILE="$RUN_DIR/frontend.pid"
DOCKER_CONFIG_FALLBACK_DIR="$RUN_DIR/docker-config"
APP_SERVICES=(app-query app-import app-frontend)
SILENT=0
STOP_ALL=0

usage() {
  cat <<'EOF'
Usage: ./stop_insightvault.sh [--all|--docker] [--silent] [--help]

Options:
  --all      stop all Docker services with docker compose down
  --docker   alias of --all
  --silent   suppress non-essential output
  --help     show this help message

Default behavior stops app-query, app-import, and app-frontend while keeping data services running.
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --docker|--all)
        STOP_ALL=1
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

prepare_docker_config() {
  local docker_config_dir config_path creds_store helper

  docker_config_dir="${DOCKER_CONFIG:-$HOME/.docker}"
  config_path="$docker_config_dir/config.json"
  if [[ ! -r "$config_path" ]]; then
    return 0
  fi

  creds_store="$(tr -d '\n' < "$config_path" | sed -nE 's/.*"credsStore"[[:space:]]*:[[:space:]]*"([^"]+)".*/\1/p')"
  if [[ -z "$creds_store" ]]; then
    return 0
  fi

  helper="docker-credential-$creds_store"
  if command -v "$helper" >/dev/null 2>&1; then
    return 0
  fi

  mkdir -p "$DOCKER_CONFIG_FALLBACK_DIR"
  printf '{\n  "auths": {}\n}\n' > "$DOCKER_CONFIG_FALLBACK_DIR/config.json"
  export DOCKER_CONFIG="$DOCKER_CONFIG_FALLBACK_DIR"
  log "[warn] docker credsStore '$creds_store' is configured but helper '$helper' is unavailable"
  log "[warn] using temporary anonymous docker config for this project"
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

if command -v docker >/dev/null 2>&1; then
  prepare_docker_config
  if (( STOP_ALL )); then
    log "stopping all docker services..."
    docker compose down >/dev/null 2>&1 || true
    log "insightvault containers stopped"
  else
    log "stopping application containers..."
    docker compose stop "${APP_SERVICES[@]}" >/dev/null 2>&1 || true
    log "insightvault app containers stopped (if running)"
    log "data containers left running"
    log "use ./stop_insightvault.sh --all to stop everything"
  fi
else
  log "[warn] docker command not found; skipped container stop"
fi

log "insightvault stop completed"
