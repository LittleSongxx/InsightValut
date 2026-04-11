#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR="$PROJECT_ROOT/.run"
LOG_DIR="$PROJECT_ROOT/logs"
IMPORT_PID_FILE="$RUN_DIR/import_service.pid"
QUERY_PID_FILE="$RUN_DIR/query_service.pid"

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

mkdir -p "$RUN_DIR" "$LOG_DIR" "$PROJECT_ROOT/.cache/models" "$PROJECT_ROOT/.cache/huggingface" "$PROJECT_ROOT/.cache/modelscope"

cd "$PROJECT_ROOT"

if ! command -v docker >/dev/null 2>&1; then
  echo "[error] docker command not found"
  exit 1
fi

if ! command -v conda >/dev/null 2>&1; then
  echo "[error] conda command not found"
  exit 1
fi

echo "[1/6] starting docker infra containers..."
docker compose up -d mongodb etcd minio minio-init milvus neo4j >/dev/null

echo "[2/6] ensuring BGE-M3 model exists in local docker volume..."
if [[ ! -f "$PROJECT_ROOT/.cache/models/bge-m3/config.json" ]]; then
  docker run --rm \
    -v "$PROJECT_ROOT/.cache/models:/models" \
    -e HF_HOME=/models/hf \
    python:3.11-slim \
    bash -lc "set -e; pip install --quiet --no-cache-dir huggingface_hub && python - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download(
    repo_id='BAAI/bge-m3',
    local_dir='/models/bge-m3',
    local_dir_use_symlinks=False,
    resume_download=True,
)
print('BGE-M3 model prepared at /models/bge-m3')
PY"
else
  echo "[ok] BGE-M3 already present, skip download"
fi

echo "[3/6] stopping stale API processes..."
"$PROJECT_ROOT/stop_insightvault.sh" --silent || true
ensure_port_free 8000
ensure_port_free 8001

echo "[4/6] starting import service..."
ensure_port_free 8000
nohup conda run -n langchain --no-capture-output python -m app.import_process.api.file_import_service >"$LOG_DIR/import_service.log" 2>&1 &
echo $! > "$IMPORT_PID_FILE"

echo "[5/6] starting query service..."
ensure_port_free 8001
nohup conda run -n langchain --no-capture-output python -m app.query_process.api.query_service >"$LOG_DIR/query_service.log" 2>&1 &
echo $! > "$QUERY_PID_FILE"

echo "[6/6] waiting for API health checks..."
wait_http "import-service" "http://127.0.0.1:8000/import.html" 120
wait_http "query-service" "http://127.0.0.1:8001/health" 120

echo ""
echo "insightvault started successfully"
echo "- import log: $LOG_DIR/import_service.log"
echo "- query  log: $LOG_DIR/query_service.log"
echo "- import page: http://127.0.0.1:8000/import.html"
echo "- chat   page: http://127.0.0.1:8001/chat.html"
echo "- query health: http://127.0.0.1:8001/health"
echo "- stop command: ./stop_insightvault.sh"
