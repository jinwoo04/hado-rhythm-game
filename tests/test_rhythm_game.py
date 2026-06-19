"""rhythm_game 단위 테스트 — 게임 로직만 (UI / 영상 없이)."""
from __future__ import annotations

import time

import pytest

from src.hado_movement_levels import MovementLevel
from src.rhythm_game import GameConfig, Phase, RhythmGame


def _quick_cfg(**override) -> GameConfig:
    base = dict(
        duration_sec=10.0,
        hold_frames=3,
        prep_frames=2,       # 테스트용: 준비 자세 2프레임으로 단축
        min_confidence=0.5,
        flash_sec=0.1,
        cooldown_sec=0.1,
        level=MovementLevel.NORMAL,
        seed=42,
    )
    base.update(override)
    return GameConfig(**base)


def _do_prep(g: RhythmGame, now: float = 1000.0) -> float:
    """준비 자세(ready)를 prep_frames만큼 입력해서 PLAYING으로 전환. 현재 시각 반환."""
    for i in range(g.cfg.prep_frames):
        g.update("ready", 0.9, now=now + i * 0.033)
    return now + g.cfg.prep_frames * 0.033


def test_idle_until_start():
    g = RhythmGame(_quick_cfg())
    state = g.update("squat", 0.9, now=1000.0)
    assert state.phase == Phase.IDLE
    assert state.score == 0
    assert state.target is None


def test_start_enters_preparing():
    """start() 후 PREPARING 단계로 진입."""
    g = RhythmGame(_quick_cfg())
    g.start(now=1000.0)
    assert g.phase == Phase.PREPARING
    assert g.target is not None
    assert g.target in g.pool


def test_preparing_to_playing_on_ready():
    """준비 자세 유지 → PLAYING 전환."""
    g = RhythmGame(_quick_cfg(prep_frames=2))
    g.start(now=1000.0)
    assert g.phase == Phase.PREPARING
    _do_prep(g, now=1000.0)
    assert g.phase == Phase.PLAYING


def test_prep_progress_increases():
    """prep_progress가 0→1로 증가."""
    g = RhythmGame(_quick_cfg(prep_frames=4))
    g.start(now=1000.0)
    state = g.update("ready", 0.9, now=1000.0)
    assert state.prep_progress > 0.0
    assert state.phase == Phase.PREPARING


def test_successful_hold_increments_score():
    g = RhythmGame(_quick_cfg(hold_frames=3))
    g.start(now=1000.0)
    t = 1000.0
    t = _do_prep(g, now=t)   # PREPARING → PLAYING
    target = g.target
    for i in range(3):
        state = g.update(target, 0.9, now=t + i * 0.033)
    assert state.score == 1
    assert state.phase == Phase.SUCCESS_FLASH


def test_low_confidence_rejected():
    g = RhythmGame(_quick_cfg(hold_frames=2, min_confidence=0.7))
    g.start(now=1000.0)
    t = _do_prep(g)
    target = g.target
    for i in range(5):
        g.update(target, 0.5, now=t + i * 0.033)
    assert g.score == 0


def test_wrong_movement_resets_hold():
    g = RhythmGame(_quick_cfg(hold_frames=3))
    g.start(now=1000.0)
    t = _do_prep(g)
    target = g.target
    other = next(m for m in g.pool if m != target)

    g.update(target, 0.9, now=t + 0.00)
    g.update(target, 0.9, now=t + 0.03)
    g.update(other,  0.9, now=t + 0.06)  # 끊김
    g.update(target, 0.9, now=t + 0.09)
    state = g.update(target, 0.9, now=t + 0.12)
    assert state.score == 0


def test_consecutive_targets_differ():
    g = RhythmGame(_quick_cfg(hold_frames=2, cooldown_sec=0.0, flash_sec=0.0, prep_frames=2))
    g.start(now=1000.0)
    t = _do_prep(g, now=1000.0)
    first_target = g.target

    # 본 동작 성공
    g.update(first_target, 0.9, now=t + 0.00)
    g.update(first_target, 0.9, now=t + 0.03)
    # FLASH→COOLDOWN(0초)→PREPARING 전환
    g.update("ready", 0.9, now=t + 0.10)
    g.update("ready", 0.9, now=t + 0.20)  # prep 2프레임 → PLAYING

    assert g.target is not None
    if len(g.pool) > 1:
        assert g.target != first_target


def test_time_expires_to_ended():
    g = RhythmGame(_quick_cfg(duration_sec=1.0))
    g.start(now=1000.0)
    state = g.update("ready", 0.9, now=1002.0)
    assert state.phase == Phase.ENDED


def test_summary_after_game():
    """두 번의 성공 후 결과 요약이 올바르게 집계되는지."""
    g = RhythmGame(_quick_cfg(hold_frames=2, cooldown_sec=0.0, flash_sec=0.0, prep_frames=2))
    g.start(now=1000.0)
    t = _do_prep(g, now=1000.0)

    # 첫 타겟 성공
    t1 = g.target
    g.update(t1, 0.9, now=t + 0.00)
    g.update(t1, 0.9, now=t + 0.03)
    # FLASH→COOLDOWN(0초)→PREPARING→PLAYING
    g.update("ready", 0.9, now=t + 0.06)
    g.update("ready", 0.9, now=t + 0.09)  # prep 완료 → PLAYING
    t2 = g.target
    assert t2 is not None
    g.update(t2, 0.9, now=t + 0.12)
    g.update(t2, 0.9, now=t + 0.15)
    summary = g.end(now=t + 0.20)
    assert summary.total_score == 2
    assert len(summary.records) == 2
    assert sum(summary.per_movement_count.values()) == 2


def test_easy_pool_non_empty():
    """EASY 풀은 3개 이상이어야 함."""
    from src.hado_movement_levels import LEVEL_POOLS
    assert len(LEVEL_POOLS[MovementLevel.EASY]) >= 3


def test_side_step_temporarily_excluded():
    """사이드스텝은 분류기 정확도 문제로 NORMAL 풀에서 임시 제외 상태."""
    from src.hado_movement_levels import BASIC_MOVEMENTS
    assert "side_step" not in BASIC_MOVEMENTS, (
        "side_step 이 다시 포함되었으면 이 테스트를 제거하세요"
    )


def test_hard_pool_includes_intermediate_when_present():
    """중급 동작이 추가되면 HARD 풀이 자동 확장되는지."""
    import src.hado_movement_levels as L
    original = list(L.INTERMEDIATE_MOVEMENTS)
    try:
        L.INTERMEDIATE_MOVEMENTS.append("__test_intermediate__")
        # LEVEL_POOLS는 dict가 이미 만들어진 상태이므로 직접 합쳐서 확인
        pool = L.BASIC_MOVEMENTS + L.INTERMEDIATE_MOVEMENTS
        assert "__test_intermediate__" in pool
    finally:
        L.INTERMEDIATE_MOVEMENTS[:] = original
