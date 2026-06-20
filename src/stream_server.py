"""웹 서버 — 브라우저에서 게임 상태 실시간 시청.

VNC 없이 Raspberry Pi 게임 상태를 같은 WiFi의 모든 기기에서 볼 수 있음.
rhythm_game_demo.py / movement_demo.py 에서 --stream 플래그로 활성화.

엔드포인트:
    /              — 텍스트 게임 상태 페이지 (카메라 불필요, 자동 새로고침)
    /status        — JSON 게임 상태 (AJAX 폴링용)
    /video_feed    — MJPEG 영상 스트림 (카메라 필요)

접속 방법:
    Pi 터미널에서 출력되는 URL을 브라우저에 입력
    예) http://172.20.10.4:8080
"""
from __future__ import annotations

import json
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


_STATUS_HTML = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="utf-8">
<title>HADO 리듬 게임</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
* { margin:0; padding:0; box-sizing:border-box; }
body {
  background:#0a0a1a;
  font-family:'Segoe UI', 'Apple SD Gothic Neo', sans-serif;
  color:#fff;
  min-height:100vh;
  display:flex;
  flex-direction:column;
  align-items:center;
  justify-content:center;
  padding:20px;
  text-align:center;
}
#phase {
  font-size:13px;
  color:#666;
  letter-spacing:3px;
  text-transform:uppercase;
  margin-bottom:10px;
}
#target {
  font-size:80px;
  font-weight:900;
  color:#00e5ff;
  line-height:1;
  margin:16px 0;
  transition:color 0.15s;
  min-height:100px;
}
#target.success { color:#00ff88; }
#target.ended   { color:#ffbb00; }
.row {
  display:flex;
  gap:40px;
  margin:10px 0;
}
.stat {
  font-size:20px;
  color:#888;
}
.stat span {
  color:#fff;
  font-weight:700;
  font-size:28px;
}
#hold-wrap {
  width:280px;
  height:14px;
  background:#1c1c2e;
  border-radius:7px;
  overflow:hidden;
  margin:18px auto 0;
}
#hold-bar {
  height:100%;
  background:#00e5ff;
  border-radius:7px;
  transition:width 0.1s;
}
#hold-bar.success { background:#00ff88; }
#recognized {
  margin-top:14px;
  font-size:13px;
  color:#444;
}
#fps { font-size:11px; color:#333; margin-top:6px; }
</style>
</head>
<body>
<div id="phase">연결 중...</div>
<div id="target">HADO</div>
<div class="row">
  <div class="stat">점수<br><span id="score">0</span></div>
  <div class="stat">남은 시간<br><span id="time">--</span>초</div>
</div>
<div id="hold-wrap">
  <div id="hold-bar" style="width:0%"></div>
</div>
<div id="recognized">인식: --</div>
<div id="fps">FPS: --</div>
<script>
const PHASE_KO = {
  idle:'대기 중', preparing:'준비 자세', playing:'⚡ 플레이',
  flash:'✅ 성공!', cooldown:'다음 준비', ended:'🏆 게임 종료',
};
const MV_KO = {
  squat:'스쿼트', lunge:'런지', back_lunge:'백런지',
  slide:'슬라이드', next_direction:'넥스트', weaving:'위빙',
  burpee:'버피', ready:'준비',
};
function update() {
  fetch('/status')
    .then(r => r.json())
    .then(d => {
      document.getElementById('phase').textContent = PHASE_KO[d.phase] || d.phase || '--';
      const t = document.getElementById('target');
      t.textContent = MV_KO[d.target] || d.target || '--';
      t.className = d.phase === 'flash' ? 'success' : d.phase === 'ended' ? 'ended' : '';
      document.getElementById('score').textContent = d.score ?? 0;
      document.getElementById('time').textContent =
        d.time_left != null ? d.time_left.toFixed(1) : '--';
      const pct = ((d.hold_progress || 0) * 100).toFixed(0) + '%';
      const bar = document.getElementById('hold-bar');
      bar.style.width = pct;
      bar.className = d.phase === 'flash' ? 'success' : '';
      document.getElementById('recognized').textContent =
        '인식: ' + (MV_KO[d.last_recognized] || d.last_recognized || '--');
      document.getElementById('fps').textContent =
        'FPS: ' + (d.fps ? d.fps.toFixed(0) : '--');
    })
    .catch(() => {
      document.getElementById('phase').textContent = '서버 연결 끊김';
    });
}
setInterval(update, 300);
update();
</script>
</body>
</html>"""


class MJPEGStreamServer:
    """별도 스레드에서 동작하는 HTTP 서버.

    - /            텍스트 게임 상태 페이지 (카메라 불필요)
    - /status      JSON 게임 상태 (300ms 폴링)
    - /video_feed  MJPEG 영상 스트림 (push() 호출 시 활성화)

    사용법:
        stream = MJPEGStreamServer(port=8080)
        stream.start()
        # 매 프레임 렌더링 후:
        stream.push(frame)                   # 영상 (optional)
        stream.push_status(state_dict)       # 게임 상태 (필수)
        # 종료 시:
        stream.stop()
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8080) -> None:
        self._buf    = _FrameBuffer()
        self._host   = host
        self._port   = port
        self._server: Optional[HTTPServer] = None
        self._thread: Optional[threading.Thread] = None
        self._status: dict = {}
        self._status_lock = threading.Lock()

    # ── public API ─────────────────────────────────────────────

    def push(self, frame: np.ndarray) -> None:
        """렌더링된 BGR 프레임을 MJPEG 스트림에 전달."""
        self._buf.put(frame)

    def push_status(self, status: dict) -> None:
        """게임 상태 딕셔너리를 /status 엔드포인트에 반영."""
        with self._status_lock:
            self._status = status

    def start(self) -> None:
        """백그라운드 스레드에서 HTTP 서버 시작."""
        buf          = self._buf
        status_ref   = self._status
        status_lock  = self._status_lock

        class _Handler(BaseHTTPRequestHandler):
            def log_message(self, *_):
                pass  # 접속 로그 숨김

            def do_GET(self):
                if self.path in ("/", "/index.html"):
                    body = _STATUS_HTML.encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)

                elif self.path == "/status":
                    with status_lock:
                        data = dict(status_ref)
                    body = json.dumps(data, ensure_ascii=False).encode("utf-8")
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Cache-Control", "no-cache")
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
        print(f"[Stream]    /video_feed 는 영상 스트림 (카메라 필요)")

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
