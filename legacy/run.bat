@echo off
setlocal

REM 저장소 루트에서 실행 (legacy\run.bat 위치 기준)
cd /d "%~dp0.."

REM -------------------------
REM 백엔드(FastAPI) + 프론트(Streamlit) 동시 실행 스크립트 (Windows / CMD)
REM
REM 요구사항:
REM - 백엔드는 백그라운드로 실행 (uvicorn --reload)
REM - 프론트는 포그라운드로 실행 (streamlit run)
REM - Ctrl+C 등으로 종료 시 백엔드도 같이 종료
REM
REM 주의:
REM - CMD만으로 Ctrl+C 트랩 처리에 제약이 있어, PowerShell의 try/finally로
REM   "프론트 실행이 끝나면(또는 Ctrl+C로 중단되면) 백엔드 종료"를 보장한다.
REM -------------------------

set "BACKEND_HOST=127.0.0.1"
set "BACKEND_PORT=8000"
set "FRONTEND_PORT=8501"

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference='Stop';" ^
  "$backendArgs=@(' -m',' uvicorn',' main:app',' --reload',' --host',' %BACKEND_HOST%',' --port',' %BACKEND_PORT%',' --app-dir',' legacy');" ^
  "$frontendArgs=@(' -m',' streamlit',' run',' legacy/frontend.py',' --server.port',' %FRONTEND_PORT%');" ^
  "Write-Host ('[시작] 백엔드 실행: python' + ($backendArgs -join '')); " ^
  "$p = Start-Process -FilePath 'python' -ArgumentList ($backendArgs -join '') -PassThru -WindowStyle Hidden; " ^
  "try { " ^
  "  Write-Host ('[시작] 프론트 실행: python' + ($frontendArgs -join '')); " ^
  "  Write-Host '[안내] 프론트가 종료되면 백엔드도 함께 종료됩니다.'; " ^
  "  & python @frontendArgs; " ^
  "} finally { " ^
  "  if ($p -and -not $p.HasExited) { " ^
  "    Write-Host ('[종료] 백엔드(uvicorn) 프로세스 정리 중... (pid=' + $p.Id + ')'); " ^
  "    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue; " ^
  "  } " ^
  "}"

endlocal

