# 딥러닝의 정석 with 파이토치 - Windows 가상환경 셋업
# 사용법: PowerShell에서 이 파일이 있는 폴더로 이동한 뒤
#   .\setup_env.ps1
# GPU 없이 CPU만 쓰려면:
#   .\setup_env.ps1 -Cpu

param(
    [switch]$Cpu
)

$ErrorActionPreference = "Stop"

$PythonExe = "C:\Users\KDS23\AppData\Local\Programs\Python\Python310\python.exe"
$VenvDir = Join-Path $PSScriptRoot ".venv"

if (-not (Test-Path $PythonExe)) {
    throw "Python 3.10 을 찾을 수 없습니다: $PythonExe"
}

Write-Host "[1/6] 가상환경 생성: $VenvDir" -ForegroundColor Cyan
& $PythonExe -m venv $VenvDir

$VenvPython = Join-Path $VenvDir "Scripts\python.exe"

Write-Host "[2/6] pip 업그레이드" -ForegroundColor Cyan
& $VenvPython -m pip install --upgrade pip

Write-Host "[3/6] gym==0.25.2 legacy 빌드를 위한 setuptools/wheel 고정" -ForegroundColor Cyan
& $VenvPython -m pip install "setuptools==65.5.0" "wheel==0.38.4"

if ($Cpu) {
    Write-Host "[4/6] PyTorch 설치 (CPU 전용)" -ForegroundColor Cyan
    & $VenvPython -m pip install torch torchvision torchaudio
} else {
    Write-Host "[4/6] PyTorch 설치 (CUDA 13.0 / RTX 50 시리즈 Blackwell 지원)" -ForegroundColor Cyan
    & $VenvPython -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
}

Write-Host "[5/6] 나머지 패키지 설치 (requirements-windows.txt)" -ForegroundColor Cyan
& $VenvPython -m pip install -r (Join-Path $PSScriptRoot "requirements-windows.txt")

Write-Host "[6/6] Jupyter 커널 등록" -ForegroundColor Cyan
& $VenvPython -m ipykernel install --user --name "17-deep-learning" --display-name "17-deep-learning (.venv)"

Write-Host ""
Write-Host "완료. 가상환경 활성화:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "GPU 인식 확인:"
Write-Host '  python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"'

