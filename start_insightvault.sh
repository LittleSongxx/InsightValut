#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$PROJECT_ROOT/.run"
LOG_DIR="$PROJECT_ROOT/logs"
FRONTEND_PID_FILE="$RUN_DIR/frontend.pid"

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

mkdir -p "$RUN_DIR" "$LOG_DIR"

cd "$PROJECT_ROOT"

if ! command -v docker >/dev/null 2>&1; then
  echo "[error] docker command not found"
  exit 1
fi

echo "[1/4] building and starting all services via docker-compose..."
docker compose up --build -d >/dev/null

echo "[2/4] waiting for API health checks..."
wait_http "import-service" "http://127.0.0.1:8000/health" 120
wait_http "query-service" "http://127.0.0.1:8001/health" 120

echo "[3/4] stopping stale frontend dev server..."
if [[ -f "$FRONTEND_PID_FILE" ]]; then
  kill "$(cat "$FRONTEND_PID_FILE")" 2>/dev/null || true
  rm -f "$FRONTEND_PID_FILE"
fi

echo "[4/4] starting frontend dev server..."
ensure_port_free 3002
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
echo "- frontend    : http://127.0.0.1:3002"
echo "- import API  : http://127.0.0.1:8000/health"
echo "- query  API  : http://127.0.0.1:8001/health"
echo "- stop command: docker compose down"
