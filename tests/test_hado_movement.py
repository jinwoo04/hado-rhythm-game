"""src/hado_movement.py 단위 테스트."""
from __future__ import annotations

import math
import numpy as np
import pytest


# ── 더미 Detection 생성 헬퍼 ────────────────────────────────────
def _make_kpts(positions: dict[int, tuple[float, float]],
               img_h: int = 480, img_w: int = 640) -> np.ndarray:
    """선택적 키포인트로 (17, 3) 배열 생성. 나머지는 conf=0."""
    kpts = np.zeros((17, 3), dtype=np.float32)
    for idx, (x, y) in positions.items():
        kpts[idx] = [x, y, 0.9]
    return kpts


def _make_det(kpts: np.ndarray, x1=100, y1=50, x2=300, y2=450):
    """더미 Detection 객체."""
    from src.detector import Detection
    return Detection(x1=x1, y1=y1, x2=x2, y2=y2, confidence=0.9, keypoints=kpts)


# ── 표준 직립 자세 키포인트 ────────────────────────────────────
def _standing_kpts(cx=200, top_y=80) -> np.ndarray:
    """표준 직립 자세: 어깨180, 엉덩이270, 무릎360, 발목430."""
    return _make_kpts({
        0:  (cx,       top_y),       # 코
        5:  (cx - 30,  top_y + 70),  # 왼어깨
        6:  (cx + 30,  top_y + 70),  # 오른어깨
        11: (cx - 20,  top_y + 160), # 왼엉덩이
        12: (cx + 20,  top_y + 160), # 오른엉덩이
        13: (cx - 20,  top_y + 240), # 왼무릎
        14: (cx + 20,  top_y + 240), # 오른무릎
        15: (cx - 20,  top_y + 330), # 왼발목
        16: (cx + 20,  top_y + 330), # 오른발목
    })


class TestMovementLabels:
    def test_all_9_movements_in_ko(self):
        from src.hado_movement import MOVEMENT_KO
        assert len(MOVEMENT_KO) == 9

    def test_all_colors_defined(self):
        from src.hado_movement import MOVEMENT_KO, MOVEMENT_COLOR
        for k in MOVEMENT_KO:
            assert k in MOVEMENT_COLOR

    def test_all_emojis_defined(self):
        from src.hado_movement import MOVEMENT_KO, MOVEMENT_EMOJI
        for k in MOVEMENT_KO:
            assert k in MOVEMENT_EMOJI


class TestMovementResultDataclass:
    def test_fields_exist(self):
        from src.hado_movement import MovementResult
        r = MovementResult(movement="squat", confidence=0.9)
        assert r.movement == "squat"
        assert r.confidence == 0.9
        assert isinstance(r.features, dict)

    def test_optional_features(self):
        from src.hado_movement import MovementResult
        r = MovementResult(movement="ready", confidence=0.5, features={"sw": 0.3})
        assert r.features["sw"] == 0.3


class TestClassifyHadoMovement:
    def test_returns_none_without_keypoints(self):
        from src.hado_movement import classify_hado_movement
        from src.detector import Detection
        det = Detection(x1=0, y1=0, x2=100, y2=200, confidence=0.9, keypoints=None)
        assert classify_hado_movement(det) is None

    def test_ready_for_standing(self):
        from src.hado_movement import classify_hado_movement
        kpts = _standing_kpts()
        det = _make_det(kpts, x1=160, y1=70, x2=240, y2=430)
        res = classify_hado_movement(det)
        assert res is not None
        assert res.movement in ("ready", "side_step")  # 직립 자세 허용

    def test_burpee_for_horizontal_bbox(self):
        """bbox가 가로로 넓으면 버피로 분류."""
        from src.hado_movement import classify_hado_movement
        # 수평 플랭크: bbox_aspect = 100 / 400 = 0.25 → bh 높음
        kpts = _make_kpts({
            5:  (150, 200), 6:  (180, 200),  # 어깨
            15: (300, 210), 16: (320, 215),   # 발목 (거의 같은 높이)
        })
        det = _make_det(kpts, x1=100, y1=180, x2=500, y2=280)  # 가로 400, 세로 100
        res = classify_hado_movement(det)
        assert res is not None
        assert res.movement == "burpee"

    def test_squat_for_wide_symmetric_low(self):
        """넓은 발 + 낮은 CoM → 스쿼트."""
        from src.hado_movement import classify_hado_movement
        # 어깨 폭=60, 발목 간격=120(sw=2.0), 낮은 자세
        kpts = _make_kpts({
            0:  (200, 90),
            5:  (170, 150), 6:  (230, 150),  # 어깨
            11: (180, 240), 12: (220, 240),  # 엉덩이
            13: (170, 320), 14: (230, 320),  # 무릎
            15: (155, 390), 16: (245, 390),  # 발목 (넓게)
        })
        det = _make_det(kpts, x1=130, y1=80, x2=270, y2=410)
        res = classify_hado_movement(det)
        assert res is not None
        assert res.movement == "squat"

    def test_result_confidence_range(self):
        """신뢰도는 0~1 범위."""
        from src.hado_movement import classify_hado_movement
        kpts = _standing_kpts()
        det = _make_det(kpts, x1=160, y1=70, x2=240, y2=430)
        res = classify_hado_movement(det)
        assert res is not None
        assert 0.0 <= res.confidence <= 1.0

    def test_features_dict_returned(self):
        """features 딕셔너리에 핵심 키 존재."""
        from src.hado_movement import classify_hado_movement
        kpts = _standing_kpts()
        det = _make_det(kpts, x1=160, y1=70, x2=240, y2=430)
        res = classify_hado_movement(det)
        assert res is not None
        for key in ("floor_proximity", "stance_width", "crouch_depth", "ankle_asym"):
            assert key in res.features


class TestExtractFeatures:
    def test_floor_proximity_low_for_standing(self):
        from src.hado_movement import _extract_features
        kpts = _standing_kpts()
        det = _make_det(kpts, x1=160, y1=70, x2=240, y2=430)
        f = _extract_features(det)
        assert f is not None
        assert f["floor_proximity"] < 0.3

    def test_bbox_horizontal_high_for_flat(self):
        """가로 bbox: bbox_horizontal 높아야 함."""
        from src.hado_movement import _extract_features
        kpts = _make_kpts({
            5: (150, 200), 6: (200, 200),
            15: (300, 210), 16: (350, 212),
        })
        det = _make_det(kpts, x1=100, y1=180, x2=500, y2=250)  # 400×70
        f = _extract_features(det)
        assert f is not None
        assert f["bbox_horizontal"] > 0.5

    def test_ankle_asym_for_one_leg_raised(self):
        """한 발 들어올리면 ankle_asym 큼."""
        from src.hado_movement import _extract_features
        kpts = _make_kpts({
            5:  (200, 150), 6:  (240, 150),
            11: (205, 240), 12: (235, 240),
            15: (210, 380),  # 왼발: 내려옴
            16: (230, 220),  # 오른발: 들어올림 (낮은 y)
        })
        det = _make_det(kpts, x1=180, y1=130, x2=260, y2=400)
        f = _extract_features(det)
        assert f is not None
        assert f["ankle_asym"] > 0.2
