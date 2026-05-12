@echo off
setlocal

REM 플랫 구조: run.bat 이 있는 폴더(저장소 루트)에서 main.py / app.py 실행
cd /d "%~dp0"

if exist ".venv\Scripts\python.exe" (
  set "PYTHON_EXE=%~dp0.venv\Scripts\python.exe"
) else (
  set "PYTHON_EXE=python"
)

set "BACKEND_HOST=127.0.0.1"
set "BACKEND_PORT=8000"
set "FRONTEND_PORT=8501"

REM CMD만으로 Ctrl+C 트랩 처리에 제약이 있어, PowerShell try/finally 로
REM 프론트 종료·중단 시 백엔드 프로세스를 정리한다.

powershell -NoProfile -ExecutionPolicy Bypass -Command ^
  "$ErrorActionPreference = 'Stop';" ^
  "$py = $env:PYTHON_EXE;" ^
  "$wd = (Get-Location).Path;" ^
  "$hostAddr = $env:BACKEND_HOST;" ^
  "$bp = $env:BACKEND_PORT;" ^
  "$fp = $env:FRONTEND_PORT;" ^
  "$backendArgs = @('-m','uvicorn','main:app','--reload','--host', $hostAddr, '--port', $bp);" ^
  "$frontendArgs = @('-m','streamlit','run','app.py','--server.port', $fp);" ^
  "Write-Host ('[시작] 백엔드 실행: ' + $py + ' ' + ($backendArgs -join ' '));" ^
  "$p = Start-Process -FilePath $py -ArgumentList $backendArgs -WorkingDirectory $wd -PassThru -WindowStyle Hidden;" ^
  "try {" ^
  "  Write-Host ('[시작] 프론트 실행: ' + $py + ' ' + ($frontendArgs -join ' '));" ^
  "  Write-Host '[안내] 프론트가 종료되면 백엔드도 함께 종료됩니다.';" ^
  "  & $py @frontendArgs;" ^
  "} finally {" ^
  "  if ($null -ne $p -and -not $p.HasExited) {" ^
  "    Write-Host ('[종료] 백엔드(uvicorn) 프로세스 정리 중... (pid=' + $p.Id + ')');" ^
  "    Stop-Process -Id $p.Id -Force -ErrorAction SilentlyContinue;" ^
  "  }" ^
  "}"

endlocal
