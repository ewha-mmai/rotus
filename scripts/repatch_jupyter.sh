#!/usr/bin/env bash
set -euo pipefail

CONTAINER_NAME="${1:-deepeyes-v2}"
HOST_PATCH_PATH="${HOST_PATCH_PATH:-/tmp/patch_jupyter.py}"
CONTAINER_PATCH_PATH="${CONTAINER_PATCH_PATH:-/tmp/patch_jupyter.py}"
SANDBOX_URL="${SANDBOX_URL:-http://127.0.0.1:8000/run_jupyter}"
TARGET_FILE="/workspace/jupyter_sandbox/local_jupyter_session.py"

echo "[INFO] container: ${CONTAINER_NAME}"
echo "[INFO] patch file: ${HOST_PATCH_PATH}"
echo "[INFO] sandbox url: ${SANDBOX_URL}"

if ! command -v docker >/dev/null 2>&1; then
  echo "[ERROR] docker command not found"
  exit 1
fi

if ! docker ps --format '{{.Names}}' | grep -qx "${CONTAINER_NAME}"; then
  echo "[ERROR] container '${CONTAINER_NAME}' is not running"
  echo "[HINT] docker ps"
  exit 1
fi

if [[ ! -f "${HOST_PATCH_PATH}" ]]; then
  echo "[ERROR] host patch script not found: ${HOST_PATCH_PATH}"
  exit 1
fi

echo "[INFO] copying patch script to container"
docker cp "${HOST_PATCH_PATH}" "${CONTAINER_NAME}:${CONTAINER_PATCH_PATH}"

echo "[INFO] checking whether patch is already applied"
if docker exec "${CONTAINER_NAME}" sh -lc "grep -q 'session state not persisted' '${TARGET_FILE}'"; then
  echo "[INFO] patch markers already found, skipping patch execution"
else
  echo "[INFO] applying patch"
  docker exec "${CONTAINER_NAME}" python "${CONTAINER_PATCH_PATH}"
fi

echo "[INFO] running sandbox health check"
SESSION_ID="patch_check_$(date +%s)"
RESP="$(curl -sS -m 8 -X POST "${SANDBOX_URL}" \
  -H 'Content-Type: application/json' \
  -d "{\"session_id\":\"${SESSION_ID}\",\"code\":\"print(1)\",\"timeout\":5}")"

if echo "${RESP}" | grep -q '"status":"success"'; then
  echo "[OK] patch ready and sandbox healthy"
  echo "${RESP}"
else
  echo "[ERROR] health check failed"
  echo "${RESP}"
  exit 1
fi
