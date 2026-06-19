#!/usr/bin/env bash
# HADO Smart Court 원클릭 실행 스크립트
#
# 사용법:
#   ./run.sh              # main 실행 (Level 1)
#   ./run.sh calibrate    # 캘리브레이션 도구
#   ./run.sh bench        # FPS 벤치마크
#   ./run.sh test         # 단위 테스트

set -e

# 스크립트 위치를 기준으로 절대 경로
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# 가상환경 자동 활성화
if [ -d "$SCRIPT_DIR/hado_venv" ]; then
    source "$SCRIPT_DIR/hado_venv/bin/activate"
elif [ -d "../hado_venv" ]; then
    source ../hado_venv/bin/activate
elif [ -d "$HOME/hado_venv" ]; then
    source "$HOME/hado_venv/bin/activate"
fi

CMD="${1:-main}"

case "$CMD" in
    main)
        python -m src.main "${@:2}"
        ;;
    calibrate|cal)
        python -m src.calibrate "${@:2}"
        ;;
    calibrate-aruco|aruco)
        python -m src.calibrate --aruco "${@:2}"
        ;;
    calibrate-intrinsic|intrinsic)
        python -m src.calibrate --intrinsic "${@:2}"
        ;;
    gen-checkerboard|checkerboard)
        python -m src.calibrate --gen-checkerboard
        ;;
    gen-markers|markers)
        python -m src.aruco_calibrate --gen-markers "${@:2}"
        ;;
    bench|benchmark)
        python -m src.benchmark "${@:2}"
        ;;
    detect)
        python -m src.detector "${@:2}"
        ;;
    demo)
        python -m src.demo "${@:2}"
        ;;
    demo_pose|pose)
        python -m src.demo_pose "${@:2}"
        ;;
    demo_pose_live|pose_live)
        # Pi4 라이브 카메라 전체 파이프라인 (NCNN 자동 선택, threaded 캡처)
        python -m src.demo_pose --threaded "${@:2}"
        ;;
    movement|movement_demo)
        # HADO 8가지 기본동작 실시간 인식 (게임 모드 없는 단순 데모)
        python -m src.movement_demo "${@:2}"
        ;;
    movement_live)
        # Pi4 라이브 카메라 기본동작 인식 (NCNN 자동 선택, threaded 캡처)
        python -m src.movement_demo --threaded "${@:2}"
        ;;
    game|rhythm|rhythm_game)
        # [최종발표] HADO 리듬 게임 — 80초 안에 추천 동작을 따라하는 Just Dance 형식
        python -m src.rhythm_game_demo "${@:2}"
        ;;
    game_live|rhythm_live)
        # Pi4 라이브 게임 (threaded 캡처)
        python -m src.rhythm_game_demo --threaded "${@:2}"
        ;;
    game_record)
        # 발표용 데모 영상 녹화 (autostart + 80초)
        python -m src.rhythm_game_demo --autostart --record data/rhythm_game_demo.mp4 "${@:2}"
        ;;
    action|action_demo)
        python -m src.action_demo "${@:2}"
        ;;
    action_live)
        # Pi4 라이브 카메라 동작 인식 (NCNN 자동 선택, threaded 캡처)
        python -m src.action_demo --threaded "${@:2}"
        ;;
    preflight|check|check-pi)
        # Pi4 현장 테스트 사전 점검 (의존성/모델/TTS/카메라/RAM/캘리브레이션)
        python -m src.pi_preflight "${@:2}"
        ;;
    measure-error|measure)
        python -m src.measure_error "${@:2}"
        ;;
    w5_measure|w5)
        # Pi4 W5 실측: NCNN/ONNX FPS + RAM + CPU온도 + TTS 지연 → data/w5_measurements.md
        python -m src.measure_w5 "${@:2}"
        ;;
    match)
        python -m src.main --match "${@:2}"
        ;;
    match-upload)
        python -m src.main --match --upload "${@:2}"
        ;;
    analyze)
        python -m src.analyzer "${@:2}"
        ;;
    upload)
        python -m src.upload "${@:2}"
        ;;
    upload-setup)
        python -m src.upload --setup
        ;;
    annotate|ann)
        python -m src.annotate "${@:2}"
        ;;
    label)
        python -m src.label_intents "${@:2}"
        ;;
    label-movements|label_movements|label-mv)
        # 207장 참조 사진 라벨링 도구 (data/reference_actions/labels.csv 생성)
        python -m tools.label_movements "${@:2}"
        ;;
    train-model|train_model|train)
        # 라벨된 사진으로 ML 동작 분류기 학습 (models/hado_movement_clf.pkl 생성)
        python -m tools.train_movement_model "${@:2}"
        ;;
    test)
        python -m pytest tests/ -q
        python -m src.homography
        ;;
    *)
        echo "사용법: ./run.sh [game|movement|label-movements|train-model|test]"
        echo "  game             — HADO 리듬 게임 (최종발표용)"
        echo "  game_live        — Pi4 라이브 게임"
        echo "  game_record      — 80초 데모 영상 녹화"
        echo "  movement         — 8동작 인식 단순 데모"
        echo "  label-movements  — 사진 라벨링 도구"
        echo "  train-model      — ML 분류기 재학습"
        echo "  test             — 단위 테스트 실행"
        exit 1
        ;;
esac
