"""HADO 리듬 게임 — 게임 로직 (타이머 / 점수 / 타겟 동작 선정).

Just Dance × HADO 리듬 컨셉.
- 화면에 "수행할 다음 동작" 표시
- 사용자가 그 동작을 N프레임 연속 수행하면 성공 → 점수 +1
- 새 타겟 동작 표시, 80초 동안 반복
- 끝나면 총 점수 (수행한 동작 수) + 동작별 통계

게임 로직만 담당. UI/렌더링은 rhythm_game_ui.py, 실행은 rhythm_game_demo.py.

핵심 동작 어휘는 hado_movement_levels.py 에서 가져옴. 내일 중급 동작 7개가
추가되면 INTERMEDIATE_MOVEMENTS만 채우면 자동으로 HARD 모드에서 사용됨.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from src.hado_movement_levels import LEVEL_POOLS, MovementLevel


class Phase(str, Enum):
    """게임 단계."""
    IDLE = "idle"            # 시작 전
    PREPARING = "preparing"  # 준비 자세 확인 중 (연결동작)
    PLAYING = "playing"      # 타겟 동작 수행 대기 중
    SUCCESS_FLASH = "flash"  # 성공 직후 짧은 피드백 (시각/음성 효과용)
    COOLDOWN = "cooldown"    # 다음 타겟 표시 전 짧은 대기
    ENDED = "ended"          # 80초 종료


@dataclass
class GameConfig:
    """게임 설정. CLI에서 override 가능."""
    duration_sec: float = 80.0
    hold_frames: int = 6              # N프레임 연속 매치 → 성공
    prep_frames: int = 10             # 준비 자세 유지 프레임 수 (≈0.6초 @15fps)
    min_confidence: float = 0.40      # 분류 신뢰도 하한
    flash_sec: float = 0.4            # 성공 직후 시각 피드백 시간
    cooldown_sec: float = 0.8         # 다음 타겟 표시 전 쿨다운
    level: MovementLevel = MovementLevel.NORMAL
    seed: Optional[int] = None        # 재현 가능한 게임 시드 (테스트용)


@dataclass
class GameState:
    """매 프레임 외부로 노출되는 상태."""
    phase: Phase
    target: Optional[str]             # 현재 수행해야 할 동작
    score: int                        # 수행 완료한 동작 수
    time_left: float                  # 남은 시간 (초)
    hold_progress: float              # 0~1, 본 동작 hold 진행도
    prep_progress: float              # 0~1, 준비 자세 hold 진행도
    last_recognized: Optional[str]    # 최근 인식된 동작 (HUD 표시용)
    last_confidence: float            # 최근 인식 신뢰도


@dataclass
class MovementRecord:
    """완료된 한 동작 기록 — 종료 후 통계용."""
    movement: str
    completed_at: float               # 게임 시작 후 경과 초
    time_to_complete: float           # 타겟 표시 ~ 성공까지 걸린 시간 (초)


@dataclass
class GameSummary:
    """게임 종료 시 결과 요약."""
    total_score: int                  # 총 수행 동작 수
    duration_sec: float               # 실제 경과 시간
    records: list[MovementRecord]     # 시간순 동작 기록
    per_movement_count: dict[str, int]   # 동작별 횟수
    avg_seconds_per_move: float       # 동작당 평균 소요 시간

    def to_text(self) -> str:
        """사람이 읽을 수 있는 요약."""
        lines = [
            f"총 점수: {self.total_score}",
            f"플레이 시간: {self.duration_sec:.1f}초",
            f"동작당 평균: {self.avg_seconds_per_move:.2f}초",
            "",
            "동작별 횟수:",
        ]
        for mv, cnt in sorted(self.per_movement_count.items(), key=lambda x: -x[1]):
            lines.append(f"  {mv:18s}  {cnt}회")
        return "\n".join(lines)


class RhythmGame:
    """HADO 리듬 게임 엔진. 매 프레임 update()를 호출하여 상태를 진행."""

    def __init__(self, cfg: Optional[GameConfig] = None) -> None:
        self.cfg = cfg or GameConfig()
        self.pool: list[str] = list(LEVEL_POOLS[self.cfg.level])
        if not self.pool:
            raise ValueError(f"Empty movement pool for level {self.cfg.level}")
        self._rng = random.Random(self.cfg.seed)

        # state
        self.phase: Phase = Phase.IDLE
        self.score: int = 0
        self.target: Optional[str] = None
        self._hold_count: int = 0
        self._prep_count: int = 0      # 준비 자세 연속 프레임
        self._start_time: float = 0.0
        self._target_show_time: float = 0.0
        self._flash_until: float = 0.0
        self._cooldown_until: float = 0.0
        self._last_recognized: Optional[str] = None
        self._last_confidence: float = 0.0
        self._records: list[MovementRecord] = []

    # ── public API ──────────────────────────────────────────────

    def start(self, now: Optional[float] = None) -> None:
        """게임 시작."""
        now = now if now is not None else time.time()
        self.phase = Phase.PREPARING
        self.score = 0
        self._hold_count = 0
        self._prep_count = 0
        self._records.clear()
        self._start_time = now
        self.target = self._pick_target(exclude=None)
        self._target_show_time = now
        self._flash_until = 0.0
        self._cooldown_until = 0.0

    def update(
        self,
        recognized: Optional[str],
        confidence: float,
        now: Optional[float] = None,
    ) -> GameState:
        """매 프레임 호출. 인식 결과를 받아 상태를 진행하고 GameState 반환."""
        now = now if now is not None else time.time()
        self._last_recognized = recognized
        self._last_confidence = confidence

        if self.phase == Phase.IDLE:
            return self._snapshot(now)

        # 시간 초과 확인
        if now - self._start_time >= self.cfg.duration_sec:
            self.phase = Phase.ENDED
            return self._snapshot(now)

        # 단계별 처리
        if self.phase == Phase.SUCCESS_FLASH and now >= self._flash_until:
            self.phase = Phase.COOLDOWN
            self._cooldown_until = now + self.cfg.cooldown_sec

        if self.phase == Phase.COOLDOWN and now >= self._cooldown_until:
            self.target = self._pick_target(exclude=self.target)
            self._target_show_time = now
            self._hold_count = 0
            self._prep_count = 0
            self.phase = Phase.PREPARING

        if self.phase == Phase.PREPARING:
            self._handle_preparing(recognized, confidence)

        if self.phase == Phase.PLAYING:
            self._handle_playing(recognized, confidence, now)

        return self._snapshot(now)

    def end(self, now: Optional[float] = None) -> GameSummary:
        """게임 강제 종료 또는 시간 만료 후 결과 요약."""
        now = now if now is not None else time.time()
        if self.phase != Phase.ENDED:
            self.phase = Phase.ENDED
        duration = max(0.001, now - self._start_time)
        per_count: dict[str, int] = {}
        for r in self._records:
            per_count[r.movement] = per_count.get(r.movement, 0) + 1
        avg = (
            sum(r.time_to_complete for r in self._records) / len(self._records)
            if self._records else 0.0
        )
        return GameSummary(
            total_score=self.score,
            duration_sec=duration,
            records=list(self._records),
            per_movement_count=per_count,
            avg_seconds_per_move=avg,
        )

    # ── internal ───────────────────────────────────────────────

    def _handle_preparing(
        self, recognized: Optional[str], confidence: float,
    ) -> None:
        """PREPARING 단계: 준비 자세(ready) 감지 → PLAYING으로 전환."""
        if recognized == "ready" and confidence >= self.cfg.min_confidence:
            self._prep_count += 1
        else:
            self._prep_count = max(0, self._prep_count - 1)  # 부드럽게 감소

        if self._prep_count >= self.cfg.prep_frames:
            self._prep_count = 0
            self._hold_count = 0
            self._target_show_time = time.time()  # 본 동작 시작 시점 갱신
            self.phase = Phase.PLAYING

    def _handle_playing(
        self, recognized: Optional[str], confidence: float, now: float,
    ) -> None:
        """PLAYING 단계의 hold count 처리."""
        if (
            recognized == self.target
            and confidence >= self.cfg.min_confidence
        ):
            self._hold_count += 1
        else:
            # 끊김. hold count 리셋 (반드시 연속 매치)
            self._hold_count = 0

        if self._hold_count >= self.cfg.hold_frames:
            # 성공!
            assert self.target is not None
            elapsed_target = now - self._target_show_time
            self._records.append(MovementRecord(
                movement=self.target,
                completed_at=now - self._start_time,
                time_to_complete=elapsed_target,
            ))
            self.score += 1
            self._hold_count = 0
            self.phase = Phase.SUCCESS_FLASH
            self._flash_until = now + self.cfg.flash_sec

    def _pick_target(self, exclude: Optional[str]) -> str:
        """다음 타겟 동작 선정. 직전과 같은 동작 반복은 피함."""
        candidates = [m for m in self.pool if m != exclude]
        if not candidates:
            candidates = list(self.pool)
        return self._rng.choice(candidates)

    def _snapshot(self, now: float) -> GameState:
        elapsed = max(0.0, now - self._start_time) if self._start_time else 0.0
        time_left = max(0.0, self.cfg.duration_sec - elapsed)
        hold_progress = (
            min(1.0, self._hold_count / max(1, self.cfg.hold_frames))
            if self.phase == Phase.PLAYING else 0.0
        )
        prep_progress = (
            min(1.0, self._prep_count / max(1, self.cfg.prep_frames))
            if self.phase == Phase.PREPARING else 0.0
        )
        return GameState(
            phase=self.phase,
            target=self.target,
            score=self.score,
            time_left=time_left,
            hold_progress=hold_progress,
            prep_progress=prep_progress,
            last_recognized=self._last_recognized,
            last_confidence=self._last_confidence,
        )
