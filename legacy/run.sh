#!/usr/bin/env bash
set -euo pipefail

# -------------------------
# 백엔드(FastAPI) + 프론트(Streamlit) 동시 실행 스크립트 (Mac/Linux)
#
# 요구사항:
# - 백엔드는 백그라운드로 실행 (uvicorn --reload)
# - 프론트는 포그라운드로 실행 (streamlit run)
# - Ctrl+C 등으로 스크립트 종료 시 백엔드도 같이 깔끔하게 종료 (trap 사용)
# - 스크립트는 legacy/ 에 두고, 실행 시 저장소 루트로 이동해 .venv·경로를 맞춘다.
# -------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

BACKEND_HOST="${BACKEND_HOST:-0.0.0.0}"
BACKEND_PORT="${BACKEND_PORT:-8000}"
FRONTEND_PORT="${FRONTEND_PORT:-8501}"
# 첫 기동 시 HF 모델(Bi-Encoder, Cross-Encoder) 다운로드로 수 분 걸릴 수 있음
BACKEND_READY_TIMEOUT_S="${BACKEND_READY_TIMEOUT_S:-300}"

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
  --host "${BACKEND_HOST}" --port "${BACKEND_PORT}"
  --app-dir "${SCRIPT_DIR}"
)
FRONTEND_CMD=("${PYTHON_BIN}" -m streamlit run "${SCRIPT_DIR}/frontend.py" --server.port "${FRONTEND_PORT}")

backend_pid=""

wait_for_backend() {
  # -------------------------
  # 백엔드가 실제로 포트를 열었는지 확인한 뒤 프론트를 실행한다.
  # - 백엔드는 백그라운드 실행이라, 모듈 누락/환경 문제로 즉시 죽어도 스크립트가 계속 진행될 수 있음
  # - curl/wget 의존 없이 Python 소켓으로 로컬 포트 오픈 여부 확인
  # - startup 에서 대용량 모델 로드 시 기본 8초는 부족하므로, 기본 대기는 BACKEND_READY_TIMEOUT_S 로 조절
  # -------------------------

  local deadline_s="${1:-${BACKEND_READY_TIMEOUT_S}}"
  local start_ts
  local last_msg_ts
  start_ts="$(date +%s)"
  last_msg_ts="${start_ts}"

  while true; do
    # 백엔드 프로세스가 이미 죽었으면 즉시 실패
    if ! kill -0 "${backend_pid}" 2>/dev/null; then
      echo "[오류] 백엔드(uvicorn)가 실행 중이 아닙니다. 설치/환경/로그를 확인해 주세요."
      return 1
    fi

    # 포트가 열렸는지 확인
    if "${PYTHON_BIN}" - <<'PY' >/dev/null 2>&1
import os, socket
# 서버가 0.0.0.0 으로 바인딩되면, 클라이언트는 127.0.0.1 로 접속해야 한다.
bind_host = os.environ.get("BACKEND_HOST", "127.0.0.1")
host = "127.0.0.1" if bind_host == "0.0.0.0" else bind_host
port = int(os.environ.get("BACKEND_PORT", "8000"))
with socket.create_connection((host, port), timeout=0.2):
    pass
PY
    then
      return 0
    fi

    # 타임아웃 체크
    local now_ts
    now_ts="$(date +%s)"
    if (( now_ts - start_ts >= deadline_s )); then
      echo "[오류] 백엔드 포트(${BACKEND_HOST}:${BACKEND_PORT})가 ${deadline_s}초 안에 열리지 않았습니다."
      echo "[오류] 첫 실행은 모델 다운로드로 더 걸릴 수 있습니다. BACKEND_READY_TIMEOUT_S 를 늘리거나, HF_TOKEN 설정 후 재시도하세요."
      echo "[오류] 그 외 OPENAI_API_KEY, legacy/data 아티팩트(피클·인덱스 등) 경로, 의존성 설치를 점검해 주세요."
      return 1
    fi

    # 장시간 대기 시 안내 (약 15초마다)
    if (( now_ts - last_msg_ts >= 15 )); then
      echo "[안내] 백엔드 기동 대기 중... (경과 약 $((now_ts - start_ts))초 / 한도 ${deadline_s}초, 첫 실행은 모델 다운로드로 수 분 걸릴 수 있음)"
      last_msg_ts="${now_ts}"
    fi

    sleep 0.2
  done
}

cleanup() {
  # 종료 시 백그라운드 백엔드 프로세스 정리
  if [[ -n "${backend_pid}" ]] && kill -0 "${backend_pid}" 2>/dev/null; then
    echo ""
    echo "[종료] 백엔드(uvicorn) 프로세스 정리 중... (pid=${backend_pid})"
    kill "${backend_pid}" 2>/dev/null || true

    # 정상 종료를 잠깐 기다린 뒤, 남아있으면 강제 종료
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

# 백엔드 포트 선점 여부 확인 (EADDRINUSE 시 uvicorn만 뜨고 실제 API는 죽는 혼란 방지)
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
  echo "[오류] 또는 환경 변수로 포트 변경: export BACKEND_PORT=8001  (프론트 기본 API URL도 맞춰 주세요)"
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

# 프론트는 포그라운드로 실행 (Ctrl+C는 여기로 들어오며 trap이 정리 수행)
"${FRONTEND_CMD[@]}"

