"""HADO 리듬 게임 — HUD 렌더링.

매 프레임 영상 위에 게임 상태를 그린다:
- 좌상: SCORE
- 우상: 남은 시간 (mm:ss)
- 중앙 상단: 다음 타겟 동작 (이모지 + 한글)
- 하단: 현재 인식 + 신뢰도 + hold 진행바
- 성공 순간: 화면 0.4초 초록 플래시
- 종료 후: 결과 요약 화면
"""
from __future__ import annotations

import cv2
import numpy as np

from src.hado_movement import MOVEMENT_COLOR, MOVEMENT_EMOJI, MOVEMENT_KO
from src.rhythm_game import GameState, GameSummary, Phase

_PREP_COLOR = (100, 220, 255)   # 준비 단계 강조색 (밝은 하늘)


# ── 색상 ──────────────────────────────────────────────────────────
_BG_DIM   = (15, 15, 18)
_FG_TEXT  = (230, 230, 235)
_FG_MUTE  = (150, 150, 155)
_OK       = (60, 220, 100)
_FAIL     = (60, 60, 220)
_BAR_BG   = (45, 45, 50)


def _put_kr(img: np.ndarray, text: str, xy: tuple[int, int],
            size: int, color: tuple[int, int, int]) -> None:
    """한글 텍스트 폴백 (PIL 없으면 cv2.putText 사용)."""
    try:
        from src.annotate import put_text_kr
        put_text_kr(img, text, xy, size, color)
    except Exception:
        cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                    size / 30, color, 2, cv2.LINE_AA)


def _format_time(sec: float) -> str:
    m, s = divmod(int(max(0, sec)), 60)
    return f"{m:02d}:{s:02d}"


def draw_hud(img: np.ndarray, state: GameState, fps: float) -> np.ndarray:
    """매 프레임 HUD 렌더링. in-place 수정."""
    h, w = img.shape[:2]

    # ── 상단 바 ────────────────────────────────────────────────
    cv2.rectangle(img, (0, 0), (w, 70), _BG_DIM, -1)

    # SCORE
    cv2.putText(img, "SCORE", (24, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _FG_MUTE, 1, cv2.LINE_AA)
    cv2.putText(img, str(state.score), (24, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 1.4, _FG_TEXT, 2, cv2.LINE_AA)

    # TIME
    cv2.putText(img, "TIME", (w - 140, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _FG_MUTE, 1, cv2.LINE_AA)
    time_color = _FAIL if state.time_left < 10 else _FG_TEXT
    cv2.putText(img, _format_time(state.time_left), (w - 140, 58),
                cv2.FONT_HERSHEY_SIMPLEX, 1.3, time_color, 2, cv2.LINE_AA)

    # FPS
    cv2.putText(img, f"{fps:.0f} fps", (w // 2 - 30, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _FG_MUTE, 1, cv2.LINE_AA)

    # ── 타겟 동작 박스 (중앙 상단) ──────────────────────────────
    show_target_phases = (
        Phase.PREPARING, Phase.PLAYING, Phase.SUCCESS_FLASH, Phase.COOLDOWN
    )
    if state.target and state.phase in show_target_phases:
        color = MOVEMENT_COLOR.get(state.target, _FG_TEXT)
        emoji = MOVEMENT_EMOJI.get(state.target, "")
        label = MOVEMENT_KO.get(state.target, state.target)
        is_prep = state.phase == Phase.PREPARING

        box_x, box_y, box_w, box_h = w // 2 - 220, 90, 440, 110
        overlay = img.copy()
        cv2.rectangle(overlay, (box_x, box_y),
                      (box_x + box_w, box_y + box_h), _BG_DIM, -1)
        cv2.addWeighted(overlay, 0.78, img, 0.22, 0, img)
        border_color = _PREP_COLOR if is_prep else color
        cv2.rectangle(img, (box_x, box_y),
                      (box_x + box_w, box_y + box_h), border_color, 2)

        # 헤더: PREPARING 단계면 "READY?" 표시
        header = "READY ?" if is_prep else "NEXT MOVE"
        header_color = _PREP_COLOR if is_prep else _FG_MUTE
        cv2.putText(img, header, (box_x + 16, box_y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, header_color, 1, cv2.LINE_AA)

        # 타겟 동작 라벨
        _put_kr(img, f"{emoji}  {label}",
                (box_x + 16, box_y + 30), 36, color)

        # PREPARING 단계: 준비 자세 안내 + 프로그레스 바
        if is_prep:
            _put_kr(img, "● 준비 자세를 취하세요",
                    (box_x + 16, box_y + 82), 13, _PREP_COLOR)
            bar_x2 = box_x + 16
            bar_y2 = box_y + 100
            bar_w2 = box_w - 32
            cv2.rectangle(img, (bar_x2, bar_y2),
                          (bar_x2 + bar_w2, bar_y2 + 6), _BAR_BG, -1)
            fill = int(bar_w2 * min(1.0, state.prep_progress))
            if fill > 0:
                cv2.rectangle(img, (bar_x2, bar_y2),
                              (bar_x2 + fill, bar_y2 + 6), _PREP_COLOR, -1)

    # ── 하단 패널: 현재 인식 + 진행바 ───────────────────────────
    panel_h = 70
    py = h - panel_h
    overlay = img.copy()
    cv2.rectangle(overlay, (0, py), (w, h), _BG_DIM, -1)
    cv2.addWeighted(overlay, 0.80, img, 0.20, 0, img)

    if state.last_recognized:
        rec_color = MOVEMENT_COLOR.get(state.last_recognized, _FG_MUTE)
        rec_emoji = MOVEMENT_EMOJI.get(state.last_recognized, "?")
        rec_label = MOVEMENT_KO.get(state.last_recognized, state.last_recognized)
        cv2.putText(img, "DETECTED", (24, py + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, _FG_MUTE, 1, cv2.LINE_AA)
        _put_kr(img, f"{rec_emoji} {rec_label}",
                (24, py + 30), 20, rec_color)
        cv2.putText(img, f"{state.last_confidence:.0%}",
                    (180, py + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, rec_color, 2, cv2.LINE_AA)

    # 진행바: PLAYING이면 hold, PREPARING이면 숨김
    if state.phase == Phase.PLAYING:
        bar_x, bar_y = w // 2 - 200, py + 38
        bar_w, bar_h = 400, 16
        cv2.rectangle(img, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h), _BAR_BG, -1)
        if state.target:
            target_color = MOVEMENT_COLOR.get(state.target, _OK)
            fill_w = int(bar_w * min(1.0, state.hold_progress))
            if fill_w > 0:
                cv2.rectangle(img, (bar_x, bar_y),
                              (bar_x + fill_w, bar_y + bar_h), target_color, -1)
        cv2.putText(img, "HOLD",
                    (bar_x - 56, bar_y + 13),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, _FG_MUTE, 1, cv2.LINE_AA)

    # ── 성공 순간 플래시 ───────────────────────────────────────
    if state.phase == Phase.SUCCESS_FLASH:
        flash = img.copy()
        cv2.rectangle(flash, (0, 0), (w, h), _OK, -1)
        cv2.addWeighted(flash, 0.25, img, 0.75, 0, img)
        cv2.putText(img, "+1", (w // 2 - 40, h // 2 + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 3.0, _OK, 6, cv2.LINE_AA)

    return img


def draw_intro(img: np.ndarray, level_name: str, duration_sec: float) -> np.ndarray:
    """시작 전 안내 화면."""
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), _BG_DIM, -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)

    _put_kr(img, "HADO 리듬 게임",
            (w // 2 - 180, h // 2 - 120), 44, _FG_TEXT)
    cv2.putText(img, f"Level: {level_name}    Duration: {int(duration_sec)}s",
                (w // 2 - 200, h // 2 - 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, _FG_MUTE, 2, cv2.LINE_AA)
    cv2.putText(img, "Press SPACE to start  |  Press ESC to quit",
                (w // 2 - 240, h // 2 + 40),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _OK, 1, cv2.LINE_AA)
    return img


def draw_results(img: np.ndarray, summary: GameSummary) -> np.ndarray:
    """게임 종료 후 결과 화면."""
    h, w = img.shape[:2]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, h), _BG_DIM, -1)
    cv2.addWeighted(overlay, 0.92, img, 0.08, 0, img)

    _put_kr(img, "GAME RESULT",
            (w // 2 - 140, 80), 32, _FG_MUTE)

    # 총 점수 (큰 숫자)
    score_str = str(summary.total_score)
    cv2.putText(img, score_str,
                (w // 2 - 60, 220),
                cv2.FONT_HERSHEY_SIMPLEX, 5.0, _OK, 10, cv2.LINE_AA)
    _put_kr(img, "수행한 동작 수",
            (w // 2 - 90, 250), 16, _FG_MUTE)

    # 부가 통계
    y = 320
    cv2.putText(img, f"Duration:      {summary.duration_sec:.1f} s",
                (w // 2 - 200, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _FG_TEXT, 1, cv2.LINE_AA)
    y += 35
    cv2.putText(img, f"Avg per move:  {summary.avg_seconds_per_move:.2f} s",
                (w // 2 - 200, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, _FG_TEXT, 1, cv2.LINE_AA)
    y += 50

    # 동작별 카운트
    cv2.putText(img, "BY MOVEMENT",
                (w // 2 - 200, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, _FG_MUTE, 1, cv2.LINE_AA)
    y += 30
    for mv, cnt in sorted(summary.per_movement_count.items(), key=lambda x: -x[1]):
        label = MOVEMENT_KO.get(mv, mv)
        emoji = MOVEMENT_EMOJI.get(mv, "")
        _put_kr(img, f"{emoji} {label}",
                (w // 2 - 200, y), 16, _FG_TEXT)
        cv2.putText(img, f"x{cnt}",
                    (w // 2 + 80, y + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, _OK, 2, cv2.LINE_AA)
        y += 32

    cv2.putText(img, "Press R to restart  |  Press ESC to quit",
                (w // 2 - 240, h - 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, _FG_MUTE, 1, cv2.LINE_AA)
    return img
