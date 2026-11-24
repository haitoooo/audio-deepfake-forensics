#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
FRONTEND_PORT="${FRONTEND_PORT:-5173}"
PYTHON_BIN="${PYTHON_BIN:-$(which python)}"

echo "Starting standalone Frontend server at http://0.0.0.0:${FRONTEND_PORT}"
cd "$PROJECT_ROOT/frontend"
"$PYTHON_BIN" -m http.server "$FRONTEND_PORT" --bind 0.0.0.0
