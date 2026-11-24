#!/usr/bin/env bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
export ASV_FORENSICS_ROOT="$PROJECT_ROOT"

PYTHON_BIN="${PYTHON_BIN:-$(which python)}"
SITE_PKGS="$($PYTHON_BIN -c 'import site; print(site.getsitepackages()[0])')"
if [ -d "$SITE_PKGS/nvidia" ]; then
  export LD_LIBRARY_PATH="$(echo "$SITE_PKGS"/nvidia/*/lib | tr ' ' ':'):${LD_LIBRARY_PATH:-}"
fi

GPU_ID="${GPU_ID:-2}"
BIND_HOST="${BIND_HOST:-0.0.0.0}"
PORT="${PORT:-8000}"

echo "Starting Audio Deepfake Forensics Web Demo on GPU=${GPU_ID}, http://${BIND_HOST}:${PORT}"
cd "$PROJECT_ROOT"
CUDA_VISIBLE_DEVICES="$GPU_ID" "$PYTHON_BIN" -m uvicorn backend.app:app --host "$BIND_HOST" --port "$PORT"
