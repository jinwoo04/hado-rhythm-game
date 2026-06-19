"""HADO 기본동작 실시간 인식 데모 (최종 발표용).

YOLOv8n-pose로 17개 keypoint를 추출하고 HADO 8가지 기본동작을 실시간 분류한다.

기본동작 (하도리듬운동트레이닝-1편 PPT 기반):
  ⬇ 스쿼트        — 양발 넓게, 무릎 굽혀 낮춤
  ↗ 런지           — 한발 앞, 앞무릎 90도
  ↙ 백런지         — 한발 뒤, 뒷무릎 낮춤
  ↔ 슬라이드       — 측면 넓게, 한쪽 무릎 깊게 굽힘
  ⚡ 런닝슬라이드   — 동적 측면이동 + 지면 터치
  → 사이드스텝     — 측면 소폭 이동, 직립
  ▼ 버피테스트     — 바닥 플랭크 ↔ 기립 전환
  ★ 하도리듬박스   — 무릎 높이 킥 + 팔 동작 콤보

실행:
    python -m src.movement_demo                       # 웹캠
    python -m src.movement_demo --source 0            # 카메라 인덱스 0
    python -m src.movement_demo --source video.mp4   # 비디오
    python -m src.movement_demo --record demo.mp4    # 녹화
    python -m src.movement_demo --headless --record demo.mp4  # Pi 헤드리스
"""
from __future__ import annotations

import argparse
import collections
import time
from pathlib import Path

import cv2
import numpy as np

from src.camera import Camera
from src.detector import PersonDetector
from src.hado_movement import (
    MOVEMENT_COLOR, MOVEMENT_EMOJI, MOVEMENT_KO,
    MovementResult, classify_hado_movement,
)
from src.pose import draw_keypoint_ids, draw_skeleton
from src.stream_server import MJPEGStreamServer

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ── 상수 ──────────────────────────────────────────────────────────
_SMOOTH_N   = 5      # N프레임 최다 동작으로 확정
_HISTORY_N  = 40     # 히스토리 타임라인 길이
_PANEL_H    = 90     # 하단 패널 높이


# ── 한글 텍스트 ──────────────────────────────────────────────────
def _put_kr(img: np.ndarray, text: str, xy: tuple[int, int],
            size: int, color: tuple[int, int, int]) -> None:
    try:
        from src.annotate import put_text_kr
        put_text_kr(img, text, xy, size, color)
    except Exception:
        cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                    size / 30, color, 2, cv2.LINE_AA)


# ── 오버레이 드로잉 ────────────────────────────────────────────────
def _draw_panel(
    img: np.ndarray,
    result: MovementResult,
    smoothed: str,
    fps: float,
) -> None:
    """하단 패널: 현재 동작 + 신뢰도 바."""
    h, w = img.shape[:2]
    y0 = h - _PANEL_H

    # 배경
    overlay = img.copy()
    cv2.rectangle(overlay, (0, y0), (w, h), (12, 12, 12), -1)
    cv2.addWeighted(overlay, 0.80, img, 0.20, 0, img)

    color = MOVEMENT_COLOR.get(smoothed, (180, 180, 180))
    emoji = MOVEMENT_EMOJI.get(smoothed, "?")
    label = MOVEMENT_KO.get(smoothed, smoothed)
    conf  = result.confidence

    # 동작 레이블 (대형)
    _put_kr(img, f"{emoji}  {label}", (20, y0 + 10), 32, color)

    # FPS
    cv2.putText(img, f"FPS {fps:.1f}", (w - 110, y0 + 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (120, 120, 120), 1, cv2.LINE_AA)

    # 신뢰도 바
    bar_y = y0 + 54
    bar_w = max(0, int((w - 200) * min(1.0, conf)))
    cv2.rectangle(img, (20, bar_y), (w - 180, bar_y + 8), (45, 45, 45), -1)
    cv2.rectangle(img, (20, bar_y), (20 + bar_w, bar_y + 8), color, -1)
    cv2.putText(img, f"{conf:.0%}", (w - 165, bar_y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)

    # 구분선
    sep_y = y0 + 74
    cv2.line(img, (20, sep_y), (w - 20, sep_y), (45, 45, 45), 1)
    _put_kr(img, "HADO 기본동작 인식 | 하도리듬운동트레이닝",
            (22, sep_y + 4), 11, (80, 80, 80))


def _draw_history(img: np.ndarray, history: "collections.deque[str]") -> None:
    """좌측 상단 타임라인 — 최근 N프레임 동작 색상."""
    if not history:
        return
    cell_w = max(4, min(18, (img.shape[1] // 4) // max(1, len(history))))
    x0, y0, bar_h = 10, 10, 14
    for i, mv in enumerate(history):
        color = MOVEMENT_COLOR.get(mv, (70, 70, 70))
        cv2.rectangle(img, (x0 + i * cell_w, y0),
                      (x0 + (i + 1) * cell_w - 1, y0 + bar_h), color, -1)


def _draw_legend(img: np.ndarray) -> None:
    """우측 상단 — 8가지 동작 범례."""
    h, w = img.shape[:2]
    x0 = w - 160
    y0 = 30
    lh = 20

    overlay = img.copy()
    cv2.rectangle(overlay, (x0 - 6, y0 - 4),
                  (w - 4, y0 + len(MOVEMENT_KO) * lh + 4), (18, 18, 18), -1)
    cv2.addWeighted(overlay, 0.70, img, 0.30, 0, img)

    for i, (key, ko) in enumerate(MOVEMENT_KO.items()):
        color  = MOVEMENT_COLOR.get(key, (140, 140, 140))
        emoji  = MOVEMENT_EMOJI.get(key, "?")
        y      = y0 + i * lh
        _put_kr(img, f"{emoji} {ko}", (x0, y), 11, color)


# ── 메인 루프 ─────────────────────────────────────────────────────
def run(args) -> int:
    # 모델 선택: NCNN > ONNX > PT
    ncnn_path = PROJECT_ROOT / "yolov8n-pose_ncnn_model"
    onnx_path = PROJECT_ROOT / "yolov8n-pose.onnx"
    if ncnn_path.exists() and not args.pt and not args.onnx:
        model_path = str(ncnn_path)
        print(f"[MovementDemo] NCNN 모델: {model_path}")
    elif onnx_path.exists() and not args.pt:
        model_path = str(onnx_path)
        print(f"[MovementDemo] ONNX 모델: {model_path}")
    else:
        model_path = "yolov8n-pose.pt"
        print(f"[MovementDemo] PT 모델: {model_path}")

    detector = PersonDetector(
        model_path=model_path,
        imgsz=args.imgsz,
        conf_threshold=args.conf,
        device=args.device,
    )

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    cam = Camera(source=source, width=args.width, height=args.height,
                 fps=args.fps, threaded=args.threaded)
    cam.open()

    writer = None
    if not args.headless:
        cv2.namedWindow("HADO Movement Demo", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("HADO Movement Demo", args.width, args.height)

    smooth_buf: collections.deque[str] = collections.deque(maxlen=_SMOOTH_N)
    history:    collections.deque[str] = collections.deque(maxlen=_HISTORY_N)
    smoothed    = "ready"
    fps         = 0.0
    fps_alpha   = 0.9
    last_t      = time.time()
    frame_idx   = 0
    last_result: MovementResult | None = None

    streamer: MJPEGStreamServer | None = None
    if args.stream:
        streamer = MJPEGStreamServer(port=args.stream_port)
        streamer.start()

    print("[MovementDemo] 시작 — 카메라 앞에서 하도 기본동작을 취하세요. ESC로 종료.")

    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("[MovementDemo] 프레임 읽기 실패")
                break

            dets   = detector.detect(frame)
            # 가장 큰 바운딩 박스 = 주 선수
            target = max(dets, key=lambda d: d.area) if dets else None

            if target is not None:
                draw_skeleton(frame, target)
                if args.show_kp_ids:
                    draw_keypoint_ids(frame, target)
                res = classify_hado_movement(target, frame=frame)
                if res is not None:
                    last_result = res
                    smooth_buf.append(res.movement)
                    smoothed = max(set(smooth_buf), key=list(smooth_buf).count)
                    history.append(smoothed)

            # HUD
            _draw_history(frame, history)
            if not args.no_legend:
                _draw_legend(frame)
            if last_result is not None:
                _draw_panel(frame, last_result, smoothed, fps)
            else:
                h, w = frame.shape[:2]
                _put_kr(frame, "카메라 앞에 서주세요",
                        (w // 2 - 100, h // 2), 20, (180, 180, 180))

            _put_kr(frame, f"감지: {len(dets)}명",
                    (10, frame.shape[0] - _PANEL_H - 6), 12, (100, 100, 100))

            # 녹화
            if writer is None and args.record:
                h_out, w_out = frame.shape[:2]
                rec_path = Path(args.record)
                rec_path.parent.mkdir(parents=True, exist_ok=True)
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(args.record, fourcc, 20.0, (w_out, h_out))
                print(f"[MovementDemo] 녹화: {args.record}")
            if writer:
                writer.write(frame)

            if streamer:
                streamer.push(frame)

            if not args.headless:
                cv2.imshow("HADO Movement Demo", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:   # ESC
                    break

            # FPS
            now   = time.time()
            inst  = 1.0 / max(0.001, now - last_t)
            fps   = fps_alpha * fps + (1 - fps_alpha) * inst if fps > 0 else inst
            last_t = now
            frame_idx += 1

            if args.max_frames and frame_idx >= args.max_frames:
                break

    except KeyboardInterrupt:
        print("\n[MovementDemo] 중단")
    finally:
        if writer:
            writer.release()
            print(f"[MovementDemo] 저장 완료: {args.record}")
        if streamer:
            streamer.stop()
        cam.close()
        if not args.headless:
            cv2.destroyAllWindows()

    print(f"[MovementDemo] 종료 — {frame_idx}프레임, 평균 FPS {fps:.1f}")
    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="HADO 기본동작 실시간 인식 (최종발표)")
    parser.add_argument("--source",     default="0",   help="카메라 인덱스 또는 영상 경로")
    parser.add_argument("--pt",         action="store_true", help=".pt 모델 강제 사용")
    parser.add_argument("--onnx",       action="store_true", help="ONNX 강제 사용")
    parser.add_argument("--imgsz",      type=int,   default=320)
    parser.add_argument("--conf",       type=float, default=0.40)
    parser.add_argument("--device",     default="cpu")
    parser.add_argument("--width",      type=int,   default=640)
    parser.add_argument("--height",     type=int,   default=480)
    parser.add_argument("--fps",        type=int,   default=30)
    parser.add_argument("--record",     default="",    help="출력 mp4 경로")
    parser.add_argument("--max-frames", type=int,   default=0, dest="max_frames",
                        help="최대 프레임 수 (0=무제한)")
    parser.add_argument("--headless",   action="store_true")
    parser.add_argument("--threaded",   action="store_true", help="스레드 캡처 (Pi4 FPS 향상)")
    parser.add_argument("--show-kp-ids", action="store_true", dest="show_kp_ids",
                        help="키포인트 번호 표시 (발표·디버그용)")
    parser.add_argument("--no-legend",   action="store_true", dest="no_legend",
                        help="우측 동작 범례 숨기기")
    parser.add_argument("--stream",      action="store_true",
                        help="MJPEG 웹 스트리밍 활성화 (VNC 없이 브라우저로 시청)")
    parser.add_argument("--stream-port", type=int, default=8080,
                        dest="stream_port",
                        help="스트리밍 서버 포트 (기본 5000)")
    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
