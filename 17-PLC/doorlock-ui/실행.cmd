@echo off
chcp 65001 >nul
cd /d "%~dp0"
echo 손가락 인식 도어락 UI를 시작합니다.
echo 브라우저 주소: http://127.0.0.1:8765
echo 종료하려면 이 창에서 Ctrl+C를 누르세요.
python server.py --open
pause
