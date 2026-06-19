#!/usr/bin/env bash
# HADO Smart Court IoT — Raspberry Pi 설치 스크립트
#
# 사용법 (Pi에서 실행):
#   git clone https://github.com/jinwoo04/HADO_SMART_COURT_IOT.git ~/hado
#   cd ~/hado
#   bash setup_pi.sh
#
# 재설치 (기존 내용 삭제 후 새로 설치):
#   bash setup_pi.sh --clean

set -e

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$REPO_DIR/hado_venv"
BRANCH="presentation/demo-finalization"

GREEN='\033[0;32m'; YELLOW='\033[1;33m'; RED='\033[0;31m'; NC='\033[0m'
ok()   { echo -e "${GREEN}[OK]${NC}  $1"; }
warn() { echo -e "${YELLOW}[!!]${NC}  $1"; }
info() { echo -e "      $1"; }

echo "======================================================"
echo "  HADO Smart Court IoT — Raspberry Pi 설치"
echo "======================================================"

# ── 0. 클린 설치 옵션 ─────────────────────────────────────────
if [[ "$1" == "--clean" ]]; then
    warn "기존 가상환경 삭제 후 재설치"
    rm -rf "$VENV_DIR"
fi

# ── 1. git pull (최신 코드) ───────────────────────────────────
echo ""
echo "[1] 최신 코드 업데이트"
cd "$REPO_DIR"
git fetch origin
git checkout "$BRANCH" 2>/dev/null || git checkout -b "$BRANCH" "origin/$BRANCH"
git reset --hard "origin/$BRANCH"
ok "코드 업데이트 완료 (branch: $BRANCH)"

# ── 2. 시스템 패키지 ──────────────────────────────────────────
echo ""
echo "[2] 시스템 패키지 설치 (sudo 필요)"
sudo apt-get update -q
sudo apt-get install -y -q \
    python3-pip \
    python3-venv \
    libopenblas-dev \
    libatlas-base-dev \
    fonts-nanum \
    espeak-ng \
    libcamera-apps
ok "시스템 패키지 설치 완료"
info "한글 폰트: fonts-nanum"
info "TTS: espeak-ng"

# ── 3. Python 가상환경 ────────────────────────────────────────
echo ""
echo "[3] Python 가상환경 생성"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
    ok "가상환경 생성: $VENV_DIR"
else
    ok "가상환경 기존 사용: $VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

# ── 4. Python 패키지 ──────────────────────────────────────────
echo ""
echo "[4] Python 패키지 설치 (시간이 걸릴 수 있음)"
pip install --upgrade pip -q
# Pi에서는 opencv-python 대신 headless 버전 사용
pip install \
    numpy \
    opencv-python-headless \
    "scikit-learn>=1.3.0" \
    "Pillow>=10.0.0" \
    pyyaml \
    -q
# ultralytics (ncnn 지원 포함)
pip install "ultralytics>=8.2.0" -q
ok "Python 패키지 설치 완료"

# ── 5. NCNN 모델 확인 ─────────────────────────────────────────
echo ""
echo "[5] 모델 파일 확인"
NCNN_DIR="$REPO_DIR/yolov8n-pose_ncnn_model"
ONNX_FILE="$REPO_DIR/yolov8n-pose.onnx"
PKL_FILE="$REPO_DIR/models/hado_movement_clf.pkl"

if [ -d "$NCNN_DIR" ]; then
    ok "NCNN 모델: $NCNN_DIR (Pi4 최적)"
elif [ -f "$ONNX_FILE" ]; then
    warn "NCNN 없음 — ONNX 모델로 대체 (속도 약간 느림)"
else
    warn "모델 없음! 아래 명령으로 복사:"
    info "  scp -r <맥주소>:\"<프로젝트경로>/yolov8n-pose_ncnn_model\" $REPO_DIR/"
fi

if [ -f "$PKL_FILE" ]; then
    ok "ML 분류기: $PKL_FILE"
else
    warn "ML 분류기 없음 — 규칙 기반 분류기로 동작"
    info "  라벨링 후 학습하거나 Mac에서 scp로 복사:"
    info "  scp <맥주소>:\"<프로젝트경로>/models/hado_movement_clf.pkl\" $REPO_DIR/models/"
fi

# ── 6. 실행 권한 ──────────────────────────────────────────────
echo ""
echo "[6] 실행 권한 설정"
chmod +x "$REPO_DIR/run.sh"
ok "run.sh 실행 권한 설정"

# ── 7. 불필요한 파일 정리 ─────────────────────────────────────
echo ""
echo "[7] 불필요한 파일 정리 (발표 관련 없는 파일)"
# 대용량 데이터 폴더만 제거 (src 코드는 유지)
rm -rf \
    "$REPO_DIR/data/reference_actions/raw_jpg" \
    "$REPO_DIR/data/drive_imports" \
    "$REPO_DIR/data/matches" \
    "$REPO_DIR/data/track_b_batch_review" \
    "$REPO_DIR/data/track_b_pose_review" \
    "$REPO_DIR/data/track_b_reviews" \
    "$REPO_DIR/outputs" \
    2>/dev/null || true
ok "불필요한 대용량 데이터 폴더 제거"

# ── 8. 동작 확인 ──────────────────────────────────────────────
echo ""
echo "[8] 설치 확인"
python3 -c "
import cv2, numpy, sklearn, PIL, ultralytics
print('  cv2:', cv2.__version__)
print('  numpy:', numpy.__version__)
print('  sklearn:', sklearn.__version__)
print('  PIL:', PIL.__version__)
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
echo "  # 웹캠으로 데모"
echo "  ./run.sh movement --source 0"
echo ""
echo "  # Pi 카메라로 데모 (threaded 캡처)"
echo "  ./run.sh movement_live"
echo ""
echo "  # 녹화"
echo "  ./run.sh movement --source 0 --record demo.mp4"
echo "======================================================"
