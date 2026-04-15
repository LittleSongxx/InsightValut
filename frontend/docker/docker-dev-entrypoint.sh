#!/bin/sh
set -eu

APP_DIR="/app"
LOCKFILE="$APP_DIR/package-lock.json"
NODE_MODULES_DIR="$APP_DIR/node_modules"
STAMP_FILE="$NODE_MODULES_DIR/.package-lock.sha256"
FRONTEND_PORT="${FRONTEND_PORT:-3002}"

CURRENT_HASH=""
INSTALLED_HASH=""

if [ -f "$LOCKFILE" ]; then
  CURRENT_HASH="$(sha256sum "$LOCKFILE" | awk '{print $1}')"
fi

if [ -f "$STAMP_FILE" ]; then
  INSTALLED_HASH="$(cat "$STAMP_FILE" 2>/dev/null || true)"
fi

if [ ! -x "$NODE_MODULES_DIR/.bin/vite" ] || [ "$CURRENT_HASH" != "$INSTALLED_HASH" ]; then
  echo "[frontend] syncing dependencies with npm ci..."
  cd "$APP_DIR"
  npm ci
  mkdir -p "$NODE_MODULES_DIR"
  echo "$CURRENT_HASH" > "$STAMP_FILE"
fi

cd "$APP_DIR"
exec npm run dev -- --host 0.0.0.0 --port "$FRONTEND_PORT" --strictPort
