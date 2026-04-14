#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$PROJECT_ROOT/.run"
LOG_DIR="$PROJECT_ROOT/logs"
FRONTEND_PID_FILE="$RUN_DIR/frontend.pid"
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

  local i
  for ((i=1; i<=max_retry; i++)); do
    local code
    code="$(curl --noproxy '*' -s -o /dev/null -m 3 -w '%{http_code}' "$url" 2>/dev/null || true)"
    if [[ "$code" == "200" ]]; then
      echo "[ok] $name ready: $url"
      return 0
    fi
    sleep 1
  done

  echo "[error] $name health check failed: $url"
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

log_step() {
  local message="$1"
  echo "[$CURRENT_STEP/$TOTAL_STEPS] $message"
  CURRENT_STEP=$((CURRENT_STEP + 1))
}

start_backend_services() {
  if (( DO_BUILD )); then
    log_step "building docker images..."
    docker compose build --progress=plain
  fi

  log_step "starting backend services via docker-compose..."
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

port_in_use() {
  local port="$1"
  ss -ltn "( sport = :$port )" 2>/dev/null | awk 'NR>1 {print $0}' | grep -q .
}

ensure_port_free() {
  local port="$1"
  if port_in_use "$port"; then
    echo "[warn] port $port already in use, trying to release..."
    if command -v fuser >/dev/null 2>&1; then
      fuser -k "${port}/tcp" >/dev/null 2>&1 || true
      sleep 1
    fi
  fi

  if port_in_use "$port"; then
    echo "[error] port $port is still occupied, please free it manually"
    return 1
  fi
}

ensure_frontend_deps() {
  local frontend_dir="$PROJECT_ROOT/frontend"

  if ! command -v npm >/dev/null 2>&1; then
    echo "[error] npm command not found"
    return 1
  fi

  if [[ -x "$frontend_dir/node_modules/.bin/vite" ]]; then
    return 0
  fi

  echo "[info] frontend dependencies missing, installing..."
  (
    cd "$frontend_dir"
    if [[ -f package-lock.json ]]; then
      npm ci
    else
      npm install
    fi
  )
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
  if post_json "http://127.0.0.1:8000/warmup/embeddings"; then
    echo "[ok] import-service embedding warmup completed"
  else
    echo "[warn] import-service embedding warmup failed or timed out"
  fi

  if post_json "http://127.0.0.1:8001/warmup/embeddings"; then
    echo "[ok] query-service embedding warmup completed"
  else
    echo "[warn] query-service embedding warmup failed or timed out"
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

cd "$PROJECT_ROOT"

if ! command -v docker >/dev/null 2>&1; then
  echo "[error] docker command not found"
  exit 1
fi

ensure_env_file

echo "[info] startup mode: build=$DO_BUILD warmup=$DO_WARMUP"

start_backend_services

log_step "waiting for API health checks..."
wait_http "import-service" "http://127.0.0.1:8000/health" 120
wait_http "query-service" "http://127.0.0.1:8001/health" 120

if (( DO_WARMUP )); then
  log_step "warming up backend embedding models..."
  warmup_backend_models
fi

log_step "stopping stale frontend dev server..."
if [[ -f "$FRONTEND_PID_FILE" ]]; then
  kill "$(cat "$FRONTEND_PID_FILE")" 2>/dev/null || true
  rm -f "$FRONTEND_PID_FILE"
fi

log_step "preparing and starting frontend dev server..."
ensure_port_free 3002
ensure_frontend_deps
cd "$PROJECT_ROOT/frontend"
nohup npm run dev >"$LOG_DIR/frontend.log" 2>&1 &
echo $! > "$FRONTEND_PID_FILE"
cd "$PROJECT_ROOT"
wait_http "frontend" "http://127.0.0.1:3002" 30

echo ""
echo "insightvault started successfully"
echo "- docker logs  : docker compose logs -f"
echo "- frontend log: $LOG_DIR/frontend.log"
echo ""
echo "- fast start  : ./start_insightvault.sh"
echo "- with build  : ./start_insightvault.sh --build"
echo "- with warmup : ./start_insightvault.sh --warmup"
echo "- full start  : ./start_insightvault.sh --build --warmup"
echo "- frontend    : http://127.0.0.1:3002"
echo "- import API  : http://127.0.0.1:8000/health"
echo "- query  API  : http://127.0.0.1:8001/health"
echo "- stop command: ./stop_insightvault.sh"
