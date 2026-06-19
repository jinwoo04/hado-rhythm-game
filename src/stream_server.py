"""MJPEG 웹 스트리밍 서버 — 브라우저에서 게임 화면 실시간 시청.

VNC 없이 Raspberry Pi 화면을 같은 WiFi의 모든 기기에서 볼 수 있음.
rhythm_game_demo.py / movement_demo.py 에서 --stream 플래그로 활성화.

접속 방법:
    Pi 터미널에서 출력되는 URL을 브라우저에 입력
    예) http://192.168.0.42:5000
"""
from __future__ import annotations

import socket
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Optional

import cv2
import numpy as np


class _FrameBuffer:
    """스레드 안전한 최신 프레임 버퍼."""

    def __init__(self) -> None:
        self._lock  = threading.Lock()
        self._jpeg: Optional[bytes] = None
        self._event = threading.Event()

    def put(self, frame: np.ndarray, quality: int = 70) -> None:
        ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, quality])
        if not ok:
            return
        with self._lock:
            self._jpeg = buf.tobytes()
        self._event.set()

    def get(self, timeout: float = 1.0) -> Optional[bytes]:
        self._event.wait(timeout=timeout)
        self._event.clear()
        with self._lock:
            return self._jpeg


class MJPEGStreamServer:
    """별도 스레드에서 동작하는 MJPEG HTTP 서버.

    사용법:
        stream = MJPEGStreamServer(port=5000)
        stream.start()
        # 매 프레임 렌더링 후:
        stream.push(frame)
        # 종료 시:
        stream.stop()
    """

    _HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>HADO Rhythm Game — Live</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body {
  background:#0a0a0a;
  display:flex; flex-direction:column;
  justify-content:center; align-items:center;
  min-height:100vh;
}
img { max-width:100vw; max-height:95vh; object-fit:contain; }
p { color:#fff; font-family:sans-serif; font-size:12px;
    opacity:0.35; margin-top:8px; letter-spacing:.05em; }
</style>
</head>
<body>
<img src="/video_feed" alt="HADO Live">
<p>HADO Rhythm Game &mdash; Live Stream</p>
</body>
</html>"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        self._buf    = _FrameBuffer()
        self._host   = host
        self._port   = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None

    # ── public API ─────────────────────────────────────────────

    def push(self, frame: np.ndarray) -> None:
        """렌더링된 BGR 프레임을 스트림에 전달."""
        self._buf.put(frame)

    def start(self) -> None:
        """백그라운드 스레드에서 HTTP 서버 시작."""
        buf = self._buf

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass  # 접속 로그 숨김

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = MJPEGStreamServer._HTML.encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                elif self.path == "/video_feed":
                    self.send_response(200)
                    self.send_header(
                        "Content-Type",
                        "multipart/x-mixed-replace; boundary=frame",
                    )
                    self.end_headers()
                    try:
                        while True:
                            jpeg = buf.get()
                            if jpeg is None:
                                continue
                            self.wfile.write(
                                b"--frame\r\n"
                                b"Content-Type: image/jpeg\r\n\r\n"
                                + jpeg
                                + b"\r\n"
                            )
                            self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError, OSError):
                        pass
                else:
                    self.send_response(404)
                    self.end_headers()

        class _ReusableHTTPServer(HTTPServer):
            allow_reuse_address = True

        self._server = _ReusableHTTPServer((self._host, self._port), _Handler)
        self._thread = threading.Thread(
            target=self._server.serve_forever, daemon=True,
        )
        self._thread.start()

        ip = self._local_ip()
        print(f"[Stream] ▶  http://{ip}:{self._port}  ← 브라우저에서 접속")
        print(f"[Stream]    같은 WiFi의 스마트폰·노트북 모두 접속 가능")

    def stop(self) -> None:
        if self._server:
            self._server.shutdown()

    # ── internal ───────────────────────────────────────────────

    @staticmethod
    def _local_ip() -> str:
        try:
            s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            s.connect(("8.8.8.8", 80))
            ip = s.getsockname()[0]
            s.close()
            return ip
        except Exception:
            return "localhost"
