"""카메라 추상화.

Pi Camera (picamera2)와 일반 USB 웹캠(cv2.VideoCapture)을 동일한 인터페이스로 사용.
노트북에서도 동작하도록 자동 fallback.

threaded=True 로 생성하면 백그라운드 스레드가 항상 최신 프레임을 유지하므로
추론 루프에서 카메라 I/O 대기 없이 즉시 프레임을 얻는다 (Pi4 FPS 향상).
"""
from __future__ import annotations

import threading
import time
from typing import Optional, Tuple

import cv2
import numpy as np


class Camera:
    """카메라 통합 인터페이스.

    사용 예시
    ---------
    >>> cam = Camera(width=640, height=480, fps=30, threaded=True)
    >>> cam.open()
    >>> while True:
    ...     ok, frame = cam.read()
    ...     if not ok: break
    >>> cam.close()
    """

    def __init__(
        self,
        source: int | str = 0,
        width: int = 1280,
        height: int = 720,
        fps: int = 30,
        prefer_picamera: bool = True,
        threaded: bool = False,
    ):
        self.source = source
        self.width = width
        self.height = height
        self.fps = fps
        self.prefer_picamera = prefer_picamera
        self._threaded = threaded

        self._backend: str = "none"
        self._cap: Optional[cv2.VideoCapture] = None
        self._picam = None  # picamera2 인스턴스

        # 스레드 캡처 관련
        self._thread: Optional[threading.Thread] = None
        self._latest_frame: Optional[np.ndarray] = None
        self._frame_lock = threading.Lock()
        self._thread_running = False

    # ---------- 라이프사이클 ----------
    def open(self) -> str:
        """카메라 오픈. picamera2 → cv2.VideoCapture 순서로 시도.

        Returns
        -------
        str : "picamera2" | "opencv" — 실제 사용된 백엔드
        """
        if self.prefer_picamera and self.source == 0:
            try:
                from picamera2 import Picamera2  # type: ignore
                self._picam = Picamera2()
                config = self._picam.create_preview_configuration(
                    main={"size": (self.width, self.height), "format": "RGB888"}
                )
                self._picam.configure(config)
                self._picam.start()
                time.sleep(0.5)  # 노출 안정화
                self._backend = "picamera2"
                print(f"[Camera] picamera2 백엔드 사용 ({self.width}x{self.height} @{self.fps}fps)")
                return self._backend
            except (ImportError, Exception) as e:
                print(f"[Camera] picamera2 사용 불가 ({e.__class__.__name__}), OpenCV로 fallback")

        # OpenCV fallback (USB 웹캠, 노트북 내장)
        self._cap = cv2.VideoCapture(self.source)
        if not self._cap.isOpened():
            raise RuntimeError(f"카메라 열기 실패: source={self.source}")
        self._cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._cap.set(cv2.CAP_PROP_FPS, self.fps)
        self._cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # 최신 프레임만 유지 → 지연 제거

        actual_w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = self._cap.get(cv2.CAP_PROP_FPS)
        print(f"[Camera] OpenCV 백엔드 사용 (요청 {self.width}x{self.height}, "
              f"실제 {actual_w}x{actual_h} @{actual_fps:.0f}fps)")
        self._backend = "opencv"

        if self._threaded:
            self._start_capture_thread()
            print("[Camera] 스레드 캡처 활성화")

        return self._backend

    # ---------- 스레드 캡처 ----------
    def _start_capture_thread(self) -> None:
        """백그라운드 스레드로 카메라에서 최신 프레임을 지속 획득."""
        self._thread_running = True

        def _loop():
            while self._thread_running:
                if self._cap is not None:
                    ret, frame = self._cap.read()
                    if ret:
                        with self._frame_lock:
                            self._latest_frame = frame
                    else:
                        time.sleep(0.005)
                else:
                    time.sleep(0.01)

        self._thread = threading.Thread(target=_loop, daemon=True)
        self._thread.start()

    def close(self):
        self._thread_running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._picam is not None:
            try:
                self._picam.stop()
                self._picam.close()
            except Exception:
                pass
            self._picam = None
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        self._backend = "none"

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    # ---------- 프레임 읽기 ----------
    def read(self) -> Tuple[bool, np.ndarray]:
        """단일 프레임 캡처. BGR ndarray 반환.

        threaded=True 모드에서는 백그라운드 스레드가 유지한 최신 프레임을 즉시 반환한다.
        """
        if self._backend == "picamera2":
            frame = self._picam.capture_array()
            frame_bgr = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            return True, frame_bgr
        elif self._backend == "opencv":
            if self._threaded:
                with self._frame_lock:
                    if self._latest_frame is None:
                        # 스레드가 아직 첫 프레임을 캡처하지 못한 경우:
                        # False 반환 시 호출부 루프가 종료되므로 True + zeros 반환
                        return True, np.zeros((self.height, self.width, 3), dtype=np.uint8)
                    return True, self._latest_frame.copy()
            return self._cap.read()
        else:
            raise RuntimeError("카메라가 열려있지 않습니다. open()을 먼저 호출하세요.")

    @property
    def backend(self) -> str:
        return self._backend


def main():
    """단독 실행: 카메라 미리보기 + fps 측정."""
    import argparse
    parser = argparse.ArgumentParser(description="카메라 동작 확인")
    parser.add_argument("--source", default=0, help="카메라 소스 (정수 또는 비디오 파일 경로)")
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--no-display", action="store_true", help="헤드리스 모드 (fps만 측정)")
    args = parser.parse_args()

    try:
        source = int(args.source)
    except ValueError:
        source = args.source

    with Camera(source=source, width=args.width, height=args.height) as cam:
        frame_count = 0
        t_start = time.time()
        t_last_print = t_start

        while True:
            ok, frame = cam.read()
            if not ok:
                print("[Camera] 프레임 읽기 실패")
                break
            frame_count += 1

            now = time.time()
            if now - t_last_print >= 2.0:
                fps = frame_count / (now - t_start)
                print(f"[Camera] {frame_count} frames, {fps:.1f} fps "
                      f"(shape={frame.shape}, backend={cam.backend})")
                t_last_print = now

            if not args.no_display:
                cv2.putText(frame, f"backend: {cam.backend}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.imshow("Camera Test (ESC to exit)", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # ESC
                    break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
