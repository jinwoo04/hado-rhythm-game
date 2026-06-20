"""HADO 리듬 게임 — 실시간 데모 메인.

실행:
    python -m src.rhythm_game_demo                                     # 웹캠, NORMAL, 80초
    python -m src.rhythm_game_demo --source 0
    python -m src.rhythm_game_demo --source video.mp4
    python -m src.rhythm_game_demo --level EASY
    python -m src.rhythm_game_demo --level HARD --duration 90
    python -m src.rhythm_game_demo --record game_clip.mp4              # 녹화 (발표용)
    python -m src.rhythm_game_demo --headless --record game_clip.mp4   # 헤드리스 (Pi)

키:
    SPACE  — 시작 (IDLE 상태에서)
    R      — 결과 화면에서 재시작
    ESC    — 종료
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
from src.hado_movement import classify_hado_movement
from src.hado_movement_levels import MovementLevel
from src.pose import draw_skeleton
from src.rhythm_game import GameConfig, Phase, RhythmGame
from src.rhythm_game_ui import draw_hud, draw_intro, draw_results
from src.stream_server import MJPEGStreamServer

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def _largest_det(dets):
    if not dets:
        return None
    return max(dets, key=lambda d: d.area)


def run(args) -> int:
    # 모델 우선순위: NCNN(Pi) > ONNX > PT
    ncnn_path = PROJECT_ROOT / "yolov8n-pose_ncnn_model"
    onnx_path = PROJECT_ROOT / "yolov8n-pose.onnx"
    if ncnn_path.exists() and not args.pt and not args.onnx:
        model_path = str(ncnn_path)
        print(f"[RhythmGame] NCNN 모델: {model_path}")
    elif onnx_path.exists() and not args.pt:
        model_path = str(onnx_path)
        print(f"[RhythmGame] ONNX 모델: {model_path}")
    else:
        model_path = "yolov8n-pose.pt"
        print(f"[RhythmGame] PT 모델: {model_path}")

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
    cam = Camera(
        source=source,
        width=args.width,
        height=args.height,
        fps=args.fps,
        threaded=args.threaded,
    )
    cam.open()

    # 게임 인스턴스
    level = MovementLevel(args.level)
    cfg = GameConfig(
        duration_sec=args.duration,
        hold_frames=args.hold_frames,
        min_confidence=args.min_conf,
        level=level,
    )
    game = RhythmGame(cfg)
    print(f"[RhythmGame] Level={level.value}  Duration={cfg.duration_sec}s  "
          f"HoldFrames={cfg.hold_frames}  Pool={len(game.pool)}동작")

    # 웹 스트리밍 (--stream 플래그)
    streamer: MJPEGStreamServer | None = None
    if args.stream:
        streamer = MJPEGStreamServer(port=args.stream_port)
        streamer.start()

    # 분류 안정화 (최근 N프레임 최다 동작)
    smooth_buf: collections.deque[str] = collections.deque(maxlen=args.smooth_n)

    writer = None
    if not args.headless:
        cv2.namedWindow("HADO Rhythm Game", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("HADO Rhythm Game", args.width, args.height)

    if args.autostart:
        game.start()

    fps = 0.0
    fps_alpha = 0.9
    last_t = time.time()
    frame_idx = 0

    print("[RhythmGame] SPACE=시작  R=재시작  ESC=종료")
    try:
        while True:
            ok, frame = cam.read()
            if not ok:
                print("[RhythmGame] 프레임 읽기 실패")
                break

            dets = detector.detect(frame)
            target_det = _largest_det(dets)

            recognized = None
            confidence = 0.0
            if target_det is not None:
                draw_skeleton(frame, target_det)
                result = classify_hado_movement(target_det, frame=frame)
                if result is not None:
                    smooth_buf.append(result.movement)
                    most = max(set(smooth_buf), key=list(smooth_buf).count)
                    recognized = most
                    confidence = result.confidence

            # 게임 진행
            state = game.update(recognized, confidence)

            # HUD
            if state.phase == Phase.IDLE:
                draw_intro(frame, level.value, cfg.duration_sec)
            elif state.phase == Phase.ENDED:
                summary = game.end()
                draw_results(frame, summary)
            else:
                draw_hud(frame, state, fps)

            # 녹화
            if args.record:
                if writer is None:
                    rec_path = Path(args.record)
                    rec_path.parent.mkdir(parents=True, exist_ok=True)
                    h_out, w_out = frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    writer = cv2.VideoWriter(args.record, fourcc, 20.0, (w_out, h_out))
                    print(f"[RhythmGame] 녹화 시작: {args.record}")
                writer.write(frame)

            if streamer:
                streamer.push(frame)

            if not args.headless:
                cv2.imshow("HADO Rhythm Game", frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC
                    break
                elif key == ord(' ') and game.phase == Phase.IDLE:
                    game.start()
                    print("[RhythmGame] START")
                elif key == ord('r') and game.phase == Phase.ENDED:
                    game = RhythmGame(cfg)
                    game.start()
                    smooth_buf.clear()
                    print("[RhythmGame] RESTART")

            # FPS
            now = time.time()
            inst = 1.0 / max(0.001, now - last_t)
            fps = fps_alpha * fps + (1 - fps_alpha) * inst if fps > 0 else inst
            last_t = now
            frame_idx += 1

            if args.max_frames and frame_idx >= args.max_frames:
                break

    except KeyboardInterrupt:
        print("\n[RhythmGame] 중단")
    finally:
        if writer:
            writer.release()
            print(f"[RhythmGame] 녹화 저장: {args.record}")
        if streamer:
            streamer.stop()
        cam.close()
        if not args.headless:
            cv2.destroyAllWindows()

    # 최종 결과 출력
    if game.phase == Phase.ENDED:
        summary = game.end()
        print("\n========== GAME SUMMARY ==========")
        print(summary.to_text())
        print("==================================\n")

    return 0


def main() -> None:
    parser = argparse.ArgumentParser(description="HADO 리듬 게임")
    parser.add_argument("--source",      default="0")
    parser.add_argument("--pt",          action="store_true")
    parser.add_argument("--onnx",        action="store_true")
    parser.add_argument("--imgsz",       type=int,   default=320)
    parser.add_argument("--conf",        type=float, default=0.40)
    parser.add_argument("--device",      default="cpu")
    parser.add_argument("--width",       type=int,   default=960)
    parser.add_argument("--height",      type=int,   default=540)
    parser.add_argument("--fps",         type=int,   default=30)
    parser.add_argument("--threaded",    action="store_true")
    parser.add_argument("--headless",    action="store_true")
    parser.add_argument("--record",      default="")
    parser.add_argument("--max-frames",  type=int,   default=0, dest="max_frames")

    # 게임 파라미터
    parser.add_argument("--level",       choices=["EASY", "NORMAL", "HARD"], default="NORMAL")
    parser.add_argument("--duration",    type=float, default=80.0,
                        help="게임 제한 시간(초). 기본 80초")
    parser.add_argument("--hold-frames", type=int,   default=6,
                        dest="hold_frames",
                        help="N프레임 연속 매치하면 성공으로 인정")
    parser.add_argument("--min-conf",    type=float, default=0.40,
                        dest="min_conf",
                        help="분류 신뢰도 하한")
    parser.add_argument("--smooth-n",    type=int,   default=5,
                        dest="smooth_n",
                        help="동작 안정화용 sliding window")
    parser.add_argument("--autostart",    action="store_true",
                        help="SPACE 안 눌러도 즉시 시작 (녹화/헤드리스용)")
    parser.add_argument("--stream",       action="store_true",
                        help="MJPEG 웹 스트리밍 활성화 (VNC 없이 브라우저로 시청)")
    parser.add_argument("--stream-port",  type=int, default=8080,
                        dest="stream_port",
                        help="스트리밍 서버 포트 (기본 5000)")

    raise SystemExit(run(parser.parse_args()))


if __name__ == "__main__":
    main()
