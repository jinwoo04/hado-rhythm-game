#!/usr/bin/env bash
# HADO 리듬 게임 — Raspberry Pi 설치 스크립트
#
# 사용법 (Pi에서 한 번만 실행):
#   git clone https://github.com/jinwoo04/hado-rhythm-game.git ~/hado-rhythm-game
#   cd ~/hado-rhythm-game
#   bash setup_pi.sh
#
# 재설치:
#   bash setup_pi.sh --clean

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/hado_venv"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}  $1"; }
warn() { echo -e "${YELLOW}[!!]${NC}  $1"; }

echo "======================================================"
echo "  HADO 리듬 게임 — Raspberry Pi 설치"
echo "======================================================"
echo "  설치 경로: $REPO_DIR"
echo ""

# ── 0. 클린 설치 ─────────────────────────────────────────────
if [[ "$1" == "--clean" ]]; then
    warn "기존 가상환경 삭제"
    rm -rf "$VENV_DIR"
fi

# ── 1. 최신 코드 ─────────────────────────────────────────────
echo "[1] 최신 코드 업데이트"
cd "$REPO_DIR"
git fetch origin
git reset --hard origin/main
ok "코드 최신화 (main 브랜치)"

# ── 2. 시스템 패키지 ──────────────────────────────────────────
echo ""
echo "[2] 시스템 패키지 (sudo 필요)"
sudo apt-get update -q
# libatlas-base-dev 는 Trixie 에서 제거됨 → libopenblas-dev 로 대체
sudo apt-get install -y -q \
    python3-pip \
    python3-venv \
    libopenblas-dev \
    libgl1 \
    fonts-nanum \
    || sudo apt-get install -y -q python3-pip python3-venv fonts-nanum
ok "시스템 패키지 완료 (한글 폰트: fonts-nanum)"

# ── 3. 가상환경 ───────────────────────────────────────────────
echo ""
echo "[3] Python 가상환경"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    ok "가상환경 생성: $VENV_DIR"
else
    ok "기존 가상환경 사용: $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# ── 4. Python 패키지 ──────────────────────────────────────────
echo ""
echo "[4] Python 패키지 설치 (시간이 걸릴 수 있음)"
pip install --upgrade pip -q
pip install \
    numpy \
    opencv-python-headless \
    "scikit-learn>=1.3.0" \
    "Pillow>=10.0.0" \
    pyyaml \
    -q
pip install "ultralytics>=8.2.0" -q
ok "Python 패키지 설치 완료"

# ── 5. 모델 파일 확인 ─────────────────────────────────────────
echo ""
echo "[5] 모델 파일 확인"
if [ -d "$REPO_DIR/yolov8n-pose_ncnn_model" ]; then
    ok "NCNN 모델 있음 (Pi4 최적화)"
elif [ -f "$REPO_DIR/yolov8n-pose.onnx" ]; then
    warn "ONNX 모델로 대체 사용"
else
    warn "포즈 모델 없음!"
fi

if [ -f "$REPO_DIR/models/hado_movement_clf.pkl" ]; then
    ok "동작 분류기 (sklearn pkl) 있음"
else
    warn "분류기 없음 — 규칙 기반으로 동작"
fi

# ── 6. 실행 권한 ──────────────────────────────────────────────
echo ""
echo "[6] 실행 권한"
chmod +x "$REPO_DIR/run.sh"
ok "run.sh 실행 권한 설정"

# ── 7. 동작 확인 ──────────────────────────────────────────────
echo ""
echo "[7] 패키지 동작 확인"
python3 -c "
import cv2, numpy, sklearn
from PIL import Image
print('  cv2     :', cv2.__version__)
print('  numpy   :', numpy.__version__)
print('  sklearn :', sklearn.__version__)
"
ok "모든 패키지 import 성공"

# ── 완료 ─────────────────────────────────────────────────────
echo ""
echo "======================================================"
echo -e "${GREEN}  설치 완료!${NC}"
echo "======================================================"
echo ""
echo "실행 방법:"
echo "  cd $REPO_DIR"
echo "  source hado_venv/bin/activate"
echo ""
echo "  # 리듬 게임 (헤드리스 + 웹 스트리밍)"
echo "  python3 -m src.rhythm_game_demo --headless --stream --autostart"
echo ""
echo "  # 동작 인식 데모 (헤드리스 + 웹 스트리밍)"
echo "  python3 -m src.movement_demo --headless --stream"
echo ""
echo "  브라우저에서 http://<Pi-IP>:8080 으로 접속"
echo "======================================================"
