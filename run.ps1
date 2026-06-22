# 백엔드(FastAPI)와 프론트(Streamlit)를 한 번에 실행한다 (Windows PowerShell).
# 사용: 프로젝트 루트에서  .\run.ps1
# 백엔드는 별도 창에서 뜨고, 이 창에서는 Streamlit이 실행된다.

$ErrorActionPreference = "Stop"

$BackendHost = "127.0.0.1"
$BackendPort = 8000
$env:BACKEND_URL = "http://${BackendHost}:${BackendPort}"

# 가상환경의 파이썬/스트림릿 경로 (활성화 없이 직접 호출)
$Py = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$Streamlit = Join-Path $PSScriptRoot ".venv\Scripts\streamlit.exe"
if (-not (Test-Path $Py)) {
    Write-Error "가상환경이 없습니다. 먼저 'python -m venv .venv'로 만들고 의존성을 설치하세요."
}

# 1) 백엔드 기동 (새 창)
Write-Host "[run] 백엔드 기동: $($env:BACKEND_URL)"
Start-Process -FilePath $Py `
    -ArgumentList "-m", "uvicorn", "backend.app:app", "--host", $BackendHost, "--port", "$BackendPort" `
    -WorkingDirectory $PSScriptRoot

# 2) 백엔드 헬스 대기 (최대 60초)
Write-Host "[run] 백엔드 준비 대기..."
$ready = $false
for ($i = 0; $i -lt 60; $i++) {
    try {
        $r = Invoke-WebRequest -Uri "$($env:BACKEND_URL)/health" -UseBasicParsing -TimeoutSec 2
        if ($r.StatusCode -eq 200) { $ready = $true; break }
    } catch { Start-Sleep -Seconds 1 }
}
if ($ready) { Write-Host "[run] 백엔드 준비 완료" }
else { Write-Host "[run] 백엔드 준비 확인 실패(계속 진행). 백엔드 창의 로그를 확인하세요." }

# 3) 프론트 실행 (이 창에서 포그라운드)
Write-Host "[run] Streamlit 실행 (기본 http://localhost:8501)"
& $Streamlit run (Join-Path $PSScriptRoot "src\streamlit\app.py")
