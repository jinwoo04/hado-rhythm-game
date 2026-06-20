"""TTS 헬퍼 — espeak-ng로 게임 이벤트를 소리로 알림.

headless Pi 환경에서 화면 없이 동작을 소리로 확인할 수 있음.
espeak-ng가 없으면 조용히 실패 (예외 없음).

설치: sudo apt-get install -y espeak-ng
"""
from __future__ import annotations

import subprocess

_MOVEMENT_KO: dict[str, str] = {
    "squat":          "스쿼트",
    "lunge":          "런지",
    "back_lunge":     "백런지",
    "slide":          "슬라이드",
    "next_direction": "넥스트",
    "weaving":        "위빙",
    "burpee":         "버피",
    "ready":          "준비",
}

_current_proc: subprocess.Popen | None = None  # 이전 발화 프로세스 (중복 방지)


def speak(text: str) -> None:
    """Non-blocking espeak-ng TTS. 설치 안 됐으면 조용히 무시."""
    global _current_proc
    try:
        if _current_proc is not None and _current_proc.poll() is None:
            _current_proc.terminate()
        _current_proc = subprocess.Popen(
            ["espeak-ng", "-v", "ko", "-s", "150", "-a", "200", text],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except (FileNotFoundError, OSError):
        pass


def speak_target(movement: str) -> None:
    """타겟 동작 이름 발화."""
    speak(_MOVEMENT_KO.get(movement, movement))


def speak_success() -> None:
    """성공 발화."""
    speak("성공!")


def speak_game_over(score: int) -> None:
    """게임 종료 발화."""
    speak(f"게임 끝. 점수 {score}점")
