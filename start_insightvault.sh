#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$PROJECT_ROOT/.run"
LOG_DIR="$PROJECT_ROOT/logs"
FRONTEND_DIR="$PROJECT_ROOT/frontend"
START_LOCK_FILE="$RUN_DIR/start.lock"
FRONTEND_PID_FILE="$RUN_DIR/frontend.pid"
FRONTEND_PREFERRED_PORT="${FRONTEND_PORT:-3002}"
FRONTEND_PORT="$FRONTEND_PREFERRED_PORT"
FRONTEND_HOST="127.0.0.1"
FRONTEND_URL="http://$FRONTEND_HOST:$FRONTEND_PORT"
FRONTEND_LOG_FILE="$LOG_DIR/frontend.log"
FRONTEND_BIN="$FRONTEND_DIR/node_modules/.bin/vite"
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
  if command -v fuser >/dev/null 2>&1 && fuser -n tcp "$port" >/dev/null 2>&1; then
    return 0
  fi

  if command -v ss >/dev/null 2>&1 && ss -ltnH 2>/dev/null | awk '{print $4}' | grep -Eq "(^|:)$port$"; then
    return 0
  fi

  return 1
}

port_bindable() {
  local port="$1"

  if ! command -v node >/dev/null 2>&1; then
    if port_in_use "$port"; then
      return 1
    fi
    return 0
  fi

  node -e 'const net = require("node:net"); const port = Number(process.argv[1]); const hosts = ["0.0.0.0", "127.0.0.1", "::"]; (async () => { for (const host of hosts) { const ok = await new Promise((resolve) => { const server = net.createServer(); server.once("error", (err) => { server.close(() => resolve(err.code !== "EADDRINUSE")); }); server.listen(port, host, () => { server.close(() => resolve(true)); }); }); if (!ok) process.exit(1); } process.exit(0); })().catch(() => process.exit(1));' "$port" >/dev/null 2>&1
}

pick_frontend_port() {
  local start_port="$1"
  local max_offset="${2:-20}"
  local port

  for ((port=start_port; port<=start_port+max_offset; port++)); do
    if port_bindable "$port"; then
      FRONTEND_PORT="$port"
      FRONTEND_URL="http://$FRONTEND_HOST:$FRONTEND_PORT"
      if [[ "$FRONTEND_PORT" != "$FRONTEND_PREFERRED_PORT" ]]; then
        echo "[warn] frontend port $FRONTEND_PREFERRED_PORT unavailable, using $FRONTEND_PORT"
      fi
      return 0
    fi
  done

  echo "[error] no available frontend port found in range $start_port-$((start_port + max_offset))"
  return 1
}

ensure_port_free() {
  local port="$1"
  local max_retry="${2:-10}"
  local i

  for ((i=1; i<=max_retry; i++)); do
    if ! port_in_use "$port"; then
      return 0
    fi

    echo "[warn] port $port already in use, trying to release..."
    if command -v fuser >/dev/null 2>&1; then
      fuser -k "${port}/tcp" >/dev/null 2>&1 || true
    fi
    sleep 1
  done

  echo "[error] port $port is still occupied, please free it manually"
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

ensure_frontend_deps() {
  if ! command -v npm >/dev/null 2>&1; then
    echo "[error] npm command not found"
    return 1
  fi

  if [[ -x "$FRONTEND_BIN" ]]; then
    return 0
  fi

  echo "[info] frontend dependencies missing, installing..."
  (
    cd "$FRONTEND_DIR"
    if [[ -f package-lock.json ]]; then
      npm ci
    else
      npm install
    fi
  )
}

start_frontend() {
  local frontend_pid=""
  local attempt
  local max_attempts=3
  local search_port="$FRONTEND_PREFERRED_PORT"

  ensure_frontend_deps

  for ((attempt=1; attempt<=max_attempts; attempt++)); do
    stop_frontend_processes
    pick_frontend_port "$search_port" 20
    : > "$FRONTEND_LOG_FILE"

    (
      cd "$FRONTEND_DIR"
      nohup "$FRONTEND_BIN" --host "$FRONTEND_HOST" --port "$FRONTEND_PORT" --strictPort >"$FRONTEND_LOG_FILE" 2>&1 &
      echo "$!" > "$FRONTEND_PID_FILE"
    )

    frontend_pid="$(cat "$FRONTEND_PID_FILE" 2>/dev/null || true)"
    if [[ -z "$frontend_pid" ]]; then
      echo "[error] failed to capture frontend pid"
      return 1
    fi

    if wait_http "frontend" "$FRONTEND_URL" 30 "$frontend_pid" "$FRONTEND_LOG_FILE"; then
      return 0
    fi

    if [[ "$attempt" -lt "$max_attempts" ]] && [[ -f "$FRONTEND_LOG_FILE" ]] && grep -Fq "Port $FRONTEND_PORT is already in use" "$FRONTEND_LOG_FILE"; then
      echo "[warn] frontend port $FRONTEND_PORT became busy during startup, retrying..."
      search_port=$((FRONTEND_PORT + 1))
      sleep 1
      continue
    fi

    return 1
  done

  return 1
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

acquire_start_lock

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
stop_frontend_processes

log_step "preparing and starting frontend dev server..."
start_frontend

echo ""
echo "insightvault started successfully"
echo "- docker logs  : docker compose logs -f"
echo "- frontend log: $FRONTEND_LOG_FILE"
echo ""
echo "- fast start  : ./start_insightvault.sh"
echo "- with build  : ./start_insightvault.sh --build"
echo "- with warmup : ./start_insightvault.sh --warmup"
echo "- full start  : ./start_insightvault.sh --build --warmup"
echo "- frontend    : $FRONTEND_URL"
echo "- import API  : http://127.0.0.1:8000/health"
echo "- query  API  : http://127.0.0.1:8001/health"
echo "- stop command: ./stop_insightvault.sh"
