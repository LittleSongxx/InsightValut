#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$PROJECT_ROOT/.run"
LOG_DIR="$PROJECT_ROOT/logs"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
START_LOCK_FILE="$RUN_DIR/start.lock"
FRONTEND_PID_FILE="$RUN_DIR/frontend.pid"
FRONTEND_PORT="${FRONTEND_PORT:-3002}"
FRONTEND_HOST="127.0.0.1"
FRONTEND_URL="http://$FRONTEND_HOST:$FRONTEND_PORT"
FRONTEND_BIN="$FRONTEND_DIR/node_modules/.bin/vite"
IMPORT_CONTAINER_URL="http://127.0.0.1:8000"
QUERY_CONTAINER_URL="http://127.0.0.1:8001"
IMPORT_HOST_URL="http://127.0.0.1:${IMPORT_PORT:-18000}"
QUERY_HOST_URL="http://127.0.0.1:${QUERY_PORT:-8001}"
ENV_FILE="$PROJECT_ROOT/.env"
ENV_EXAMPLE_FILE="$PROJECT_ROOT/.env.example"
DO_BUILD=0
DO_WARMUP=0
CURRENT_STEP=1
TOTAL_STEPS=4

wait_http() {
  local name="$1"
  local url="$2"
  local max_retry="${3:-90}"
  local pid="${4:-}"
  local log_file="${5:-}"

  local i
  for ((i=1; i<=max_retry; i++)); do
    local code
    code="$(curl --noproxy '*' -s -o /dev/null -m 3 -w '%{http_code}' "$url" 2>/dev/null || true)"
    if [[ "$code" == "200" ]]; then
      echo "[ok] $name ready: $url"
      return 0
    fi
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "[error] $name exited before becoming ready: $url"
      if [[ -n "$log_file" ]] && [[ -f "$log_file" ]]; then
        tail -n 20 "$log_file"
      fi
      return 1
    fi
    sleep 1
  done

  echo "[error] $name health check failed: $url"
  return 1
}

service_running() {
  local service="$1"
  docker compose ps --status running --services 2>/dev/null | grep -qx "$service"
}

print_service_logs() {
  local service="$1"
  echo "[info] recent logs for $service:"
  docker compose logs --tail=40 "$service" || true
}

wait_container_http() {
  local name="$1"
  local service="$2"
  local url="$3"
  local max_retry="${4:-90}"

  local i
  for ((i=1; i<=max_retry; i++)); do
    local code
    code="$(docker compose exec -T "$service" sh -lc "curl --noproxy '*' -s -o /dev/null -m 3 -w '%{http_code}' \"$url\"" 2>/dev/null || true)"
    if [[ "$code" == "200" ]]; then
      echo "[ok] $name ready: $service -> $url"
      return 0
    fi

    if ! service_running "$service"; then
      echo "[error] $name is not running: $service"
      print_service_logs "$service"
      return 1
    fi

    sleep 1
  done

  echo "[error] $name health check failed: $service -> $url"
  print_service_logs "$service"
  return 1
}

usage() {
  cat <<'EOF'
Usage: ./start_insightvault.sh [--build] [--warmup] [--help]

Options:
  --build    rebuild Docker images before starting services
  --warmup   warm up backend embedding models after health checks
  --help     show this help message

Default behavior is quick startup without rebuild or warmup.
EOF
}

parse_args() {
  while [[ $# -gt 0 ]]; do
    case "$1" in
      --build)
        DO_BUILD=1
        ;;
      --warmup)
        DO_WARMUP=1
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

acquire_start_lock() {
  if ! command -v flock >/dev/null 2>&1; then
    return 0
  fi

  exec 9>"$START_LOCK_FILE"
  if ! flock -n 9; then
    echo "[error] another start_insightvault.sh process is already running"
    return 1
  fi
}

log_step() {
  local message="$1"
  echo "[$CURRENT_STEP/$TOTAL_STEPS] $message"
  CURRENT_STEP=$((CURRENT_STEP + 1))
}

start_backend_services() {
  if (( DO_BUILD )); then
    log_step "building docker images..."
    docker compose --progress plain build
  fi

  log_step "starting docker services via docker-compose..."
  if (( DO_BUILD )); then
    docker compose up -d >/dev/null
    return 0
  fi

  if docker compose up -d --no-build >/dev/null; then
    return 0
  fi

  echo "[error] failed to start existing backend services without rebuild"
  echo "[hint] try: ./start_insightvault.sh --build"
  return 1
}

post_json() {
  local url="$1"
  local max_retry="${2:-1}"

  local i
  for ((i=1; i<=max_retry; i++)); do
    local code
    code="$(curl --noproxy '*' -s -o /dev/null -m 180 -w '%{http_code}' -X POST -H 'Content-Type: application/json' "$url" 2>/dev/null || true)"
    if [[ "$code" == "200" ]]; then
      return 0
    fi
    sleep 1
  done

  return 1
}

post_container_json() {
  local service="$1"
  local url="$2"
  local max_retry="${3:-1}"

  local i
  for ((i=1; i<=max_retry; i++)); do
    local code
    code="$(docker compose exec -T "$service" sh -lc "curl --noproxy '*' -s -o /dev/null -m 180 -w '%{http_code}' -X POST -H 'Content-Type: application/json' \"$url\"" 2>/dev/null || true)"
    if [[ "$code" == "200" ]]; then
      return 0
    fi
    sleep 1
  done

  return 1
}

stop_frontend_processes() {
  if [[ -f "$FRONTEND_PID_FILE" ]]; then
    local pid
    pid="$(cat "$FRONTEND_PID_FILE" 2>/dev/null || true)"
    if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
      kill "$pid" 2>/dev/null || true
      sleep 1
      kill -9 "$pid" 2>/dev/null || true
    fi
    rm -f "$FRONTEND_PID_FILE"
  fi

  if command -v pgrep >/dev/null 2>&1; then
    local pid
    while IFS= read -r pid; do
      if [[ -n "$pid" ]] && kill -0 "$pid" 2>/dev/null; then
        kill "$pid" 2>/dev/null || true
        sleep 1
        kill -9 "$pid" 2>/dev/null || true
      fi
    done < <(pgrep -f "$FRONTEND_BIN" 2>/dev/null || true)
  fi
}

start_frontend() {
  wait_http "frontend" "$FRONTEND_URL" 120
}

ensure_env_file() {
  if [[ -f "$ENV_FILE" ]]; then
    return 0
  fi

  if [[ -f "$ENV_EXAMPLE_FILE" ]]; then
    cp "$ENV_EXAMPLE_FILE" "$ENV_FILE"
    echo "[info] local .env missing, created from .env.example"
    return 0
  fi

  echo "[error] .env missing and .env.example not found"
  return 1
}

warmup_backend_models() {
  if post_container_json "app-import" "$IMPORT_CONTAINER_URL/warmup/embeddings"; then
    echo "[ok] import-service embedding warmup completed"
  else
    echo "[warn] import-service embedding warmup failed or timed out"
    print_service_logs "app-import"
  fi

  if post_container_json "app-query" "$QUERY_CONTAINER_URL/warmup/embeddings"; then
    echo "[ok] query-service embedding warmup completed"
  else
    echo "[warn] query-service embedding warmup failed or timed out"
    print_service_logs "app-query"
  fi
}

parse_args "$@"

if (( DO_BUILD )); then
  TOTAL_STEPS=$((TOTAL_STEPS + 1))
fi

if (( DO_WARMUP )); then
  TOTAL_STEPS=$((TOTAL_STEPS + 1))
fi

mkdir -p "$RUN_DIR" "$LOG_DIR"

acquire_start_lock

cd "$PROJECT_ROOT"

if ! command -v docker >/dev/null 2>&1; then
  echo "[error] docker command not found"
  exit 1
fi

ensure_env_file

echo "[info] startup mode: build=$DO_BUILD warmup=$DO_WARMUP"

log_step "stopping stale local frontend dev server..."
stop_frontend_processes

start_backend_services

log_step "waiting for API health checks..."
wait_container_http "import-service" "app-import" "$IMPORT_CONTAINER_URL/health" 120
wait_container_http "query-service" "app-query" "$QUERY_CONTAINER_URL/health" 120

if (( DO_WARMUP )); then
  log_step "warming up backend embedding models..."
  warmup_backend_models
fi

log_step "waiting for frontend dev server (docker)..."
start_frontend

echo ""
echo "insightvault started successfully"
echo "- docker logs  : docker compose logs -f"
echo "- frontend log: docker compose logs -f app-frontend"
echo ""
echo "- fast start  : ./start_insightvault.sh"
echo "- with build  : ./start_insightvault.sh --build"
echo "- with warmup : ./start_insightvault.sh --warmup"
echo "- full start  : ./start_insightvault.sh --build --warmup"
echo "- frontend    : $FRONTEND_URL"
if curl --noproxy '*' -s -o /dev/null -m 3 "$IMPORT_HOST_URL/health" 2>/dev/null; then
  echo "- import API  : $IMPORT_HOST_URL/health"
else
  echo "- import API  : container check via docker compose exec -T app-import curl $IMPORT_CONTAINER_URL/health"
fi
if curl --noproxy '*' -s -o /dev/null -m 3 "$QUERY_HOST_URL/health" 2>/dev/null; then
  echo "- query  API  : $QUERY_HOST_URL/health"
else
  echo "- query  API  : container check via docker compose exec -T app-query curl $QUERY_CONTAINER_URL/health"
fi
echo "- local stop  : ./stop_insightvault.sh"
echo "- stop all    : ./stop_insightvault.sh --docker"
