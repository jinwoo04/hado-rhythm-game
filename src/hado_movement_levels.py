"""HADO 동작 레벨 정의 — 게임 모드 풀 관리.

내일 중급 동작 7개가 추가될 때 INTERMEDIATE_MOVEMENTS 만 채우면
HARD 모드에서 자동으로 풀에 포함됨. 게임 코드는 수정 불필요.

BASIC: 현재 8개 학습 완료 (squat / lunge / slide / running_slide /
        side_step / burpee / rhythm_box / ready). ready는 휴식 동작이라
        게임 타겟 풀에서는 기본적으로 제외.
INTERMEDIATE: 내일 추가 학습 예정 7개. 학습 후 이름만 여기에 등록.
"""
from __future__ import annotations

from enum import Enum


class MovementLevel(str, Enum):
    EASY = "EASY"
    NORMAL = "NORMAL"
    HARD = "HARD"


# 확정된 7가지 동작 (2026-06-18)
# ready 는 휴식 자세 — 게임 타겟 풀에서는 제외
BASIC_MOVEMENTS: list[str] = [
    "squat",
    "lunge",
    "slide",
    "next_direction",
    "weaving",
    "burpee",
]

INTERMEDIATE_MOVEMENTS: list[str] = []

# EASY: 초보자용 3가지
EASY_MOVEMENTS: list[str] = [
    "squat",
    "lunge",
    "slide",
]

LEVEL_POOLS: dict[MovementLevel, list[str]] = {
    MovementLevel.EASY:   EASY_MOVEMENTS,
    MovementLevel.NORMAL: BASIC_MOVEMENTS,
    MovementLevel.HARD:   BASIC_MOVEMENTS + INTERMEDIATE_MOVEMENTS,
}


def all_movements() -> list[str]:
    """게임에 사용되는 모든 동작."""
    return BASIC_MOVEMENTS + INTERMEDIATE_MOVEMENTS
