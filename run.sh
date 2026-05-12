#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# 백엔드(FastAPI) + 프론트(Streamlit) 동시 실행 스크립트 (Mac/Linux)
#
# 플랫 구조: run.sh 는 저장소 루트에 있으며, main.py / app.py 를 같은 디렉터리에서 실행한다.
#
# 요구사항:
# - 백엔드는 백그라운드로 실행 (uvicorn --reload)
# - 프론트는 포그라운드로 실행 (streamlit run)
# - Ctrl+C 등으로 스크립트 종료 시 백엔드도 같이 깔끔하게 종료 (trap 사용)
# -------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${SCRIPT_DIR}"
cd "${REPO_ROOT}"

BACKEND_HOST="${BACKEND_HOST:-127.0.0.1}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-8501}"
BACKEND_READY_TIMEOUT_S="${BACKEND_READY_TIMEOUT_S:-120}"

export BACKEND_HOST BACKEND_PORT

# 가상환경(.venv)이 있으면 그 Python을 우선 사용 (PEP 668 환경에서 필수)
PYTHON_BIN="${PYTHON_BIN:-}"
if [[ -z "${PYTHON_BIN}" ]]; then
  if [[ -x ".venv/bin/python" ]]; then
    PYTHON_BIN=".venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

BACKEND_CMD=(
  "${PYTHON_BIN}" -m uvicorn main:app --reload
  --host "${BACKEND_HOST}"
  --port "${BACKEND_PORT}"
)
FRONTEND_CMD=(
  "${PYTHON_BIN}" -m streamlit run app.py
  --server.port "${FRONTEND_PORT}"
)

backend_pid=""

wait_for_backend() {
  # 백엔드가 실제로 포트를 열었는지 확인한 뒤 프론트를 실행한다.
  local deadline_s="${1:-${BACKEND_READY_TIMEOUT_S}}"
  local start_ts
  local last_msg_ts
  start_ts="$(date +%s)"
  last_msg_ts="${start_ts}"

  while true; do
    if ! kill -0 "${backend_pid}" 2>/dev/null; then
      echo "[오류] 백엔드(uvicorn)가 실행 중이 아닙니다. 설치/환경/로그를 확인해 주세요."
      return 1
    fi

    if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import os, socket
bind_host = os.environ.get("BACKEND_HOST", "127.0.0.1")
host = "127.0.0.1" if bind_host == "0.0.0.0" else bind_host
port = int(os.environ.get("BACKEND_PORT", "8000"))
with socket.create_connection((host, port), timeout=0.2):
    pass
PY
    then
      return 0
    fi

    local now_ts
    now_ts="$(date +%s)"
    if (( now_ts - start_ts >= deadline_s )); then
      echo "[오류] 백엔드 포트(${BACKEND_HOST}:${BACKEND_PORT})가 ${deadline_s}초 안에 열리지 않았습니다."
      echo "[오류] 의존성 설치(uvicorn/fastapi), 포트 충돌, 환경 변수를 점검해 주세요. 필요 시 BACKEND_READY_TIMEOUT_S 를 늘리세요."
      return 1
    fi

    if (( now_ts - last_msg_ts >= 15 )); then
      echo "[안내] 백엔드 기동 대기 중... (경과 약 $((now_ts - start_ts))초 / 한도 ${deadline_s}초)"
      last_msg_ts="${now_ts}"
    fi

    sleep 0.2
  done
}

cleanup() {
  if [[ -n "${backend_pid}" ]] && kill -0 "${backend_pid}" 2>/dev/null; then
    echo ""
    echo "[종료] 백엔드(uvicorn) 프로세스 정리 중... (pid=${backend_pid})"
    kill "${backend_pid}" 2>/dev/null || true

    for _ in {1..20}; do
      if kill -0 "${backend_pid}" 2>/dev/null; then
        sleep 0.1
      else
        break
      fi
    done

    if kill -0 "${backend_pid}" 2>/dev/null; then
      echo "[종료] 백엔드가 종료되지 않아 강제 종료합니다. (pid=${backend_pid})"
      kill -9 "${backend_pid}" 2>/dev/null || true
    fi
  fi
}

trap cleanup EXIT INT TERM

if ! "${PYTHON_BIN}" - "${BACKEND_PORT}" <<'PY'
import socket
import sys

port = int(sys.argv[1])
sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
try:
    sock.bind(("127.0.0.1", port))
except OSError as e:
    if e.errno == 98:  # EADDRINUSE
        sys.exit(1)
    raise
finally:
    sock.close()
sys.exit(0)
PY
then
  echo "[오류] 포트 ${BACKEND_PORT}이(가) 이미 사용 중입니다. (Address already in use)"
  echo "[오류] 다른 터미널의 uvicorn 을 종료하거나, 예: fuser -k ${BACKEND_PORT}/tcp  후 다시 실행하세요."
  echo "[오류] 또는 환경 변수로 포트 변경: export BACKEND_PORT=8001"
  exit 1
fi

echo "[시작] 백엔드 실행: ${BACKEND_CMD[*]}"
"${BACKEND_CMD[@]}" &
backend_pid="$!"

if ! wait_for_backend "${BACKEND_READY_TIMEOUT_S}"; then
  exit 1
fi

echo "[시작] 프론트 실행: ${FRONTEND_CMD[*]}"
echo "[안내] 프론트가 종료되면 백엔드도 함께 종료됩니다."
echo "[안내] 사용 Python: ${PYTHON_BIN}"

"${FRONTEND_CMD[@]}"
