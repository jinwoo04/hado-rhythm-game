"""HADO 기본동작 분류기.

하도리듬 (HADO Rhythm) — 2025 하도리듬 자료정리.pdf + 하도리듬운동트레이닝-1편.pptx 기반.

▶ 하도리듬이란?
  HADO AR 스포츠에서 최적의 퍼포먼스를 위한 기본 동작을 체계적으로 분류하고
  음악/리듬과 결합해 훈련하는 운동 교육 시스템. 2024 HADO KOREA CUP에서 전 세계
  대상으로 공식 시연. 개발자: 박진우 (7년 하도 선수 경력, 한국 국가대표).

▶ 하도리듬의 목적
  ① HADO 경기력 향상을 위한 체계적 기본기 훈련
  ② 스포츠 과학 + 음악 리듬을 결합한 훈련 방법 개발
  ③ HADO만의 독창적 동작 어휘 체계 구축 (전 세계 교육 가능 커리큘럼)

▶ 핵심 원칙 (PDF 기반)
  수직 손 자세 = 공격/차지 준비  ↔  수평/펼친 손 = 방어/쉴드
  이 원칙은 모든 기본 동작에 공통 적용됨.

▶ 동작 체계 (PDF 레벨 분류)
  입문(기본): 스쿼트 / 사이드스텝 / 런지 / 슬라이드
  기술:       런닝슬라이드 / 버피테스트
  응용/복합:  하도리듬박스 (기→↑→↓→우 반복 콤보)
  준비:       준비 자세 (기본 직립)

▶ 분류기 우선순위
  1. YOLOv8-cls  models/hado_movement_cls.pt  (Colab 학습 후 저장)
  2. sklearn pkl models/hado_movement_clf.pkl  (기존 ML)
  3. 규칙 기반   (fallback)
  라벨링: python -m tools.label_movements
  데이터셋: python -m tools.prepare_yolo_dataset
  학습(Colab): YOLOv8n-cls.pt fine-tune

COCO 17 keypoints:
  0:코  1:왼눈  2:오른눈  3:왼귀  4:오른귀
  5:왼어깨  6:오른어깨  7:왼팔꿈치  8:오른팔꿈치
  9:왼손목  10:오른손목  11:왼엉덩이  12:오른엉덩이
  13:왼무릎  14:오른무릎  15:왼발목  16:오른발목
"""
from __future__ import annotations

import math
import pickle
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from src.detector import Detection

# ── ML 모델 자동 로드 ────────────────────────────────────────────
_ML_MODEL: dict | None = None

def _load_ml_model() -> dict | None:
    """models/hado_movement_clf.pkl 로드 (존재할 때만)."""
    global _ML_MODEL
    if _ML_MODEL is not None:
        return _ML_MODEL
    model_path = Path(__file__).resolve().parent.parent / "models/hado_movement_clf.pkl"
    if not model_path.exists():
        return None
    try:
        with open(model_path, "rb") as f:
            _ML_MODEL = pickle.load(f)
        print(f"[HADOMovement] ML 모델 로드: {model_path.name} "
              f"({_ML_MODEL.get('n_samples', '?')}샘플, "
              f"{len(_ML_MODEL.get('class_names', []))}클래스)")
    except Exception as e:
        print(f"[HADOMovement] ML 모델 로드 실패 — 규칙 기반 분류기 사용: {e}")
        _ML_MODEL = None
    return _ML_MODEL


def _classify_ml(f: dict[str, float], det: Detection) -> Optional["MovementResult"]:
    """학습된 ML 모델로 동작 분류. 실패 시 None 반환."""
    model_data = _load_ml_model()
    if model_data is None:
        return None

    try:
        kpts = det.keypoints
        if kpts is None:
            return None

        bbox_h = max(1.0, det.y2 - det.y1)
        bbox_w = max(1.0, det.x2 - det.x1)

        # shoulder_asym_y, hip_asym_y, body_lean_forward 추가 특징
        def _kp_xy(i):
            if kpts[i, 2] >= _KP_CONF:
                return float(kpts[i, 0]), float(kpts[i, 1])
            return None

        ls, rs = _kp_xy(5), _kp_xy(6)
        lh, rh = _kp_xy(11), _kp_xy(12)

        bh = max(1.0, bbox_h)
        sh_asym = abs(ls[1] - rs[1]) / bh if (ls and rs) else 0.0
        hi_asym = abs(lh[1] - rh[1]) / bh if (lh and rh) else 0.0

        body_lean = 0.0
        if ls and rs and lh and rh:
            sh_cx = (ls[0] + rs[0]) / 2
            hi_cx = (lh[0] + rh[0]) / 2
            body_lean = (sh_cx - hi_cx) / max(30.0, f["scale"])

        feat = np.array([
            f["floor_proximity"],
            f["bbox_horizontal"],
            f["stance_width"],
            f["ankle_asym"],
            f["knee_raise"],
            f["crouch_depth"],
            f["knee_asym"],
            f["foot_fore_aft"],
            f["knee_bend_avg"] / 180.0,
            f["knee_bend_diff"] / 180.0,
            f["lateral_lean"],
            f["hand_low"],
            f["nose_level"],
            sh_asym,
            hi_asym,
            body_lean,
        ], dtype=np.float32).reshape(1, -1)

        clf = model_data["pipeline"]
        le  = model_data["label_encoder"]
        proba = clf.predict_proba(feat)[0]
        pred_idx = int(np.argmax(proba))
        pred_label = le.inverse_transform([pred_idx])[0]
        confidence = float(proba[pred_idx])

        return MovementResult(pred_label, round(confidence, 3), f)
    except Exception:
        return None


# ── YOLOv8-cls 모델 (최우선) ─────────────────────────────────────
_YOLO_CLS_MODEL = None

def _load_yolo_cls_model():
    """models/hado_movement_cls.pt 로드 (존재할 때만)."""
    global _YOLO_CLS_MODEL
    if _YOLO_CLS_MODEL is not None:
        return _YOLO_CLS_MODEL
    model_path = Path(__file__).resolve().parent.parent / "models/hado_movement_cls.pt"
    if not model_path.exists():
        return None
    try:
        from ultralytics import YOLO
        _YOLO_CLS_MODEL = YOLO(str(model_path))
        n_cls = len(_YOLO_CLS_MODEL.names)
        print(f"[HADOMovement] YOLO-cls 모델 로드: {model_path.name} ({n_cls}클래스)")
    except Exception as e:
        print(f"[HADOMovement] YOLO-cls 로드 실패 — 다음 분류기로 폴백: {e}")
        _YOLO_CLS_MODEL = None
    return _YOLO_CLS_MODEL


def _classify_yolo_cls(frame: "np.ndarray", det: Detection) -> Optional["MovementResult"]:
    """YOLO-cls: 사람 bbox 크롭 후 분류. 실패 시 None 반환."""
    model = _load_yolo_cls_model()
    if model is None:
        return None
    try:
        h, w = frame.shape[:2]
        pad = 10
        x1 = max(0, int(det.x1) - pad)
        y1 = max(0, int(det.y1) - pad)
        x2 = min(w, int(det.x2) + pad)
        y2 = min(h, int(det.y2) + pad)
        if x2 <= x1 or y2 <= y1:
            return None
        crop = frame[y1:y2, x1:x2]

        results = model(crop, verbose=False)
        if not results:
            return None
        probs = results[0].probs
        if probs is None:
            return None

        pred_idx  = int(probs.top1)
        confidence = float(probs.top1conf)
        pred_label = model.names[pred_idx]
        return MovementResult(pred_label, round(confidence, 3))
    except Exception:
        return None


# ── 키포인트 인덱스 ──────────────────────────────────────────────
_N          = 0
_LS, _RS    = 5, 6
_LE, _RE    = 7, 8
_LW, _RW    = 9, 10
_LH, _RH    = 11, 12
_LK, _RK    = 13, 14
_LA, _RA    = 15, 16

_KP_CONF    = 0.25   # 신뢰도 임계값

# ── 동작 레이블 (2026-06-18 확정 7가지) ─────────────────────────
MOVEMENT_KO: dict[str, str] = {
    "squat":          "스쿼트",
    "lunge":          "런지",
    "slide":          "슬라이드",
    "weaving":        "위빙",
    "next_direction": "넥디렉션",
    "burpee":         "버피테스트",
    "ready":          "준비 동작",
}

MOVEMENT_COLOR: dict[str, tuple[int, int, int]] = {
    "squat":          (0,   200, 255),   # 하늘
    "lunge":          (40,  180, 255),   # 파랑
    "slide":          (0,   255, 160),   # 민트
    "weaving":        (255, 220, 0),     # 노랑
    "next_direction": (255, 120, 50),    # 오렌지
    "burpee":         (200, 60,  255),   # 핫핑크
    "ready":          (160, 160, 160),   # 회색
}

MOVEMENT_EMOJI: dict[str, str] = {
    "squat":          "⬇",
    "lunge":          "↗",
    "slide":          "↔",
    "weaving":        "〜",
    "next_direction": "▶",
    "burpee":         "▼",
    "ready":          "●",
}


# ── 분류 결과 ────────────────────────────────────────────────────
@dataclass
class MovementResult:
    """HADO 기본동작 분류 결과."""
    movement:   str              # MOVEMENT_KO 키
    confidence: float            # 0.0~1.0
    features:   dict[str, float] = field(default_factory=dict)


# ── 내부 헬퍼 ────────────────────────────────────────────────────
def _kp(kpts: np.ndarray, idx: int) -> Optional[tuple[float, float]]:
    if kpts[idx, 2] >= _KP_CONF:
        return float(kpts[idx, 0]), float(kpts[idx, 1])
    return None


def _dist(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def _midpoint(a: tuple[float, float], b: tuple[float, float]) -> tuple[float, float]:
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _angle_deg(p_prev, p_joint, p_next) -> float:
    """관절각도: p_joint를 꼭짓점으로 하는 각도 (0~180도)."""
    v1 = (p_prev[0] - p_joint[0], p_prev[1] - p_joint[1])
    v2 = (p_next[0] - p_joint[0], p_next[1] - p_joint[1])
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    mag = math.hypot(*v1) * math.hypot(*v2)
    if mag < 1e-6:
        return 180.0
    return math.degrees(math.acos(max(-1.0, min(1.0, dot / mag))))


# ── 특징 추출 ────────────────────────────────────────────────────
def _extract_features(det: Detection) -> Optional[dict[str, float]]:
    """keypoints에서 분류에 사용할 특징값 추출."""
    if det.keypoints is None:
        return None

    kpts = det.keypoints
    bbox_h = max(1.0, det.y2 - det.y1)
    bbox_w = max(1.0, det.x2 - det.x1)

    ls, rs = _kp(kpts, _LS), _kp(kpts, _RS)
    lh, rh = _kp(kpts, _LH), _kp(kpts, _RH)
    lk, rk = _kp(kpts, _LK), _kp(kpts, _RK)
    la, ra = _kp(kpts, _LA), _kp(kpts, _RA)
    lw, rw = _kp(kpts, _LW), _kp(kpts, _RW)
    nose   = _kp(kpts, _N)

    # ── 몸통 높이 (어깨 ~ 발목 수직 거리) ──────────────────────
    body_h = bbox_h  # fallback
    if ls and rs and la and ra:
        sh_y = (ls[1] + rs[1]) / 2
        an_y = max(la[1], ra[1])   # 더 낮은 발목 기준
        if an_y > sh_y:
            body_h = an_y - sh_y

    # ── 어깨폭 (스케일) ─────────────────────────────────────────
    shoulder_w = abs(rs[0] - ls[0]) if (ls and rs) else max(bbox_w * 0.3, 30.0)
    scale = max(shoulder_w, 30.0)

    # ── 어깨/엉덩이 중점 ─────────────────────────────────────────
    sh_mid = _midpoint(ls, rs) if (ls and rs) else None
    hi_mid = _midpoint(lh, rh) if (lh and rh) else None

    # ── bbox 가로세로비: 낮을수록 수평 자세 (버피 플랭크) ───────
    bbox_aspect = bbox_h / max(1.0, bbox_w)   # 직립≈1.5~2.5, 플랭크≈0.4~0.7

    # ── floor_proximity: 어깨가 발목과 같은 높이 수준인가 ────────
    # 방법1: 어깨y와 발목y의 차이가 작음 = 수평 자세
    floor_proximity = 0.0
    if sh_mid and la and ra:
        ankle_y = max(la[1], ra[1])
        # 어깨가 발목보다 많이 높으면(y 작으면) 직립; 비슷하면 플랭크
        sh_to_ankle_norm = (ankle_y - sh_mid[1]) / max(1.0, body_h)
        # 직립: sh_to_ankle_norm ≈ 0.8~1.0 / 플랭크: ≈ 0.0~0.2
        floor_proximity = max(0.0, min(1.0, 1.0 - sh_to_ankle_norm / 0.5))

    # ── bbox_horizontal: bbox가 가로로 넓은가 (버피 플랭크 보조) ─
    bbox_horizontal = max(0.0, min(1.0, 1.0 - bbox_aspect / 0.8))

    # ── nose_level: 코가 발목보다 낮거나 비슷한가 (눕기 자세) ────
    nose_level = 0.0
    if nose and la and ra:
        ankle_y = max(la[1], ra[1])
        sh_y = sh_mid[1] if sh_mid else ankle_y - body_h
        nose_level = max(0.0, min(1.0, (nose[1] - sh_y) / max(1.0, ankle_y - sh_y)))

    # ── stance_width: 발목 간격 / 어깨폭 ───────────────────────
    stance_width = 0.0
    if la and ra:
        stance_width = abs(la[0] - ra[0]) / scale

    # ── ankle_asym: 두 발목 높이 차이 (하도리듬박스: 한 발 들어올림) ─
    # 양수 = 한 발이 다른 발보다 높이 올라감
    ankle_asym = 0.0
    if la and ra:
        ankle_asym = abs(la[1] - ra[1]) / max(1.0, body_h)

    # ── knee_raise: 무릎이 엉덩이보다 얼마나 높이 올라갔는가 ────
    knee_raise = 0.0
    if hi_mid and lk:
        knee_raise = max(knee_raise, (hi_mid[1] - lk[1]) / max(1.0, body_h))
    if hi_mid and rk:
        knee_raise = max(knee_raise, (hi_mid[1] - rk[1]) / max(1.0, body_h))
    # 힙 미감지 시 어깨를 기준으로 추정
    if knee_raise == 0.0 and sh_mid:
        sh_to_knee_ref = body_h * 0.45   # 어깨에서 무릎까지 체고의 ~45%
        if lk:
            raise_from_sh = (sh_mid[1] + sh_to_knee_ref - lk[1]) / max(1.0, body_h)
            knee_raise = max(knee_raise, raise_from_sh)
        if rk:
            raise_from_sh = (sh_mid[1] + sh_to_knee_ref - rk[1]) / max(1.0, body_h)
            knee_raise = max(knee_raise, raise_from_sh)

    # ── crouch_depth: CoM가 얼마나 낮은가 ──────────────────────
    crouch_depth = 0.0
    if hi_mid and la and ra:
        ankle_y = max(la[1], ra[1])
        top_y   = ankle_y - body_h
        crouch_depth = max(0.0, min(1.0, (hi_mid[1] - top_y) / max(1.0, body_h)))
    elif sh_mid and la and ra:
        # 힙 미감지 → 어깨 위치 기반 크라우치 추정
        ankle_y = max(la[1], ra[1])
        top_y   = ankle_y - body_h
        # 어깨가 체고의 50% 이상 아래면 낮은 자세
        sh_ratio = (sh_mid[1] - top_y) / max(1.0, body_h)
        crouch_depth = max(0.0, min(1.0, sh_ratio - 0.15))

    # ── knee_asymmetry: 두 무릎 높이 차이 (런지/슬라이드) ──────
    knee_asym = 0.0
    if lk and rk:
        knee_asym = abs(lk[1] - rk[1]) / max(1.0, body_h)

    # ── foot_fore_aft: 발목의 전후 비대칭 (런지) ────────────────
    foot_fore_aft = 0.0
    if la and ra:
        foot_fore_aft = abs(la[0] - ra[0]) / max(1.0, body_h)

    # ── knee_bend_avg / diff ─────────────────────────────────────
    knee_bend_avg = 180.0
    bends = []
    if lh and lk and la:
        bends.append(_angle_deg(lh, lk, la))
    if rh and rk and ra:
        bends.append(_angle_deg(rh, rk, ra))
    if bends:
        knee_bend_avg = sum(bends) / len(bends)

    knee_bend_diff = abs(bends[0] - bends[1]) if len(bends) == 2 else 0.0

    # ── lateral_lean: 상체 좌우 기울기 ─────────────────────────
    lateral_lean_signed = 0.0   # +: 오른쪽 기울기 / -: 왼쪽 기울기
    lateral_lean = 0.0
    if sh_mid and hi_mid:
        lateral_lean_signed = (sh_mid[0] - hi_mid[0]) / scale
        lateral_lean = abs(lateral_lean_signed)

    # ── hand_low: 손목이 무릎 아래 (런닝슬라이드: 손 지면 터치) ─
    hand_low = 0.0
    if lk and lw:
        hand_low = max(hand_low, (lw[1] - lk[1]) / max(1.0, body_h))
    if rk and rw:
        hand_low = max(hand_low, (rw[1] - rk[1]) / max(1.0, body_h))

    return {
        "floor_proximity":  floor_proximity,
        "bbox_horizontal":  bbox_horizontal,
        "nose_level":       nose_level,
        "stance_width":     stance_width,
        "ankle_asym":       ankle_asym,
        "knee_raise":       knee_raise,
        "crouch_depth":     crouch_depth,
        "knee_asym":        knee_asym,
        "foot_fore_aft":    foot_fore_aft,
        "knee_bend_avg":    knee_bend_avg,
        "knee_bend_diff":   knee_bend_diff,
        "lateral_lean":        lateral_lean,
        "lateral_lean_signed": lateral_lean_signed,
        "hand_low":            hand_low,
        "bbox_aspect":         bbox_aspect,
        "scale":               scale,
        "body_h":              body_h,
    }


# ── 키포인트 평활화 ──────────────────────────────────────────────
class KeypointSmoother:
    """EMA 기반 keypoint 시간적 평활화 — 관절 떨림 / 오감지 감소."""

    def __init__(self, alpha: float = 0.5):
        self._alpha = alpha          # 최신 프레임 가중치 (클수록 반응 빠름)
        self._prev: "np.ndarray | None" = None

    def reset(self) -> None:
        self._prev = None

    def smooth(self, kpts: "np.ndarray") -> "np.ndarray":
        """(17, 3) keypoints 배열에 EMA 적용."""
        if self._prev is None:
            self._prev = kpts.copy()
            return kpts
        out = kpts.copy()
        for i in range(17):
            if kpts[i, 2] >= _KP_CONF and self._prev[i, 2] >= _KP_CONF:
                out[i, 0] = self._alpha * kpts[i, 0] + (1 - self._alpha) * self._prev[i, 0]
                out[i, 1] = self._alpha * kpts[i, 1] + (1 - self._alpha) * self._prev[i, 1]
        self._prev = out.copy()
        return out


# ── 시퀀스 감지 ───────────────────────────────────────────────────
class SequenceDetector:
    """최근 N프레임 특징으로 시퀀스 동작 감지.

    위빙: lateral_lean_signed 방향 전환 + 중앙 crouch 피크
    """

    def __init__(self, window: int = 20):
        self._window = window
        self._lean_buf: "deque[float]" = deque(maxlen=window)
        self._crouch_buf: "deque[float]" = deque(maxlen=window)
        self._cooldown: int = 0   # 감지 후 버퍼 리셋 대기 프레임

    def reset(self) -> None:
        self._lean_buf.clear()
        self._crouch_buf.clear()
        self._cooldown = 0

    def update(self, f: "dict[str, float]") -> "str | None":
        """특징값 업데이트 → 감지된 동작 이름 반환. 없으면 None."""
        if self._cooldown > 0:
            self._cooldown -= 1
            return None

        self._lean_buf.append(f.get("lateral_lean_signed", 0.0))
        self._crouch_buf.append(f.get("crouch_depth", 0.0))

        if len(self._lean_buf) < self._window:
            return None

        result = self._detect_weaving()
        if result:
            self._lean_buf.clear()
            self._crouch_buf.clear()
            self._cooldown = self._window
        return result

    def _detect_weaving(self) -> "str | None":
        lean   = list(self._lean_buf)
        crouch = list(self._crouch_buf)
        n = len(lean)
        h = n // 2

        first_mean  = sum(lean[:h]) / h
        second_mean = sum(lean[h:]) / h

        # 전반/후반 lateral lean 이 충분한 진폭으로 방향 전환해야 위빙
        MIN_AMP = 0.08
        crossed = (
            (first_mean >  MIN_AMP and second_mean < -MIN_AMP) or
            (first_mean < -MIN_AMP and second_mean >  MIN_AMP)
        )
        if not crossed:
            return None

        # 중간 구간에 crouch 피크 → 스쿼트 자세를 지나는 위빙 원형
        mid_s = max(0, h - 3)
        mid_e = min(n, h + 3)
        mid_crouch  = sum(crouch[mid_s:mid_e]) / max(1, mid_e - mid_s)
        edge_crouch = (sum(crouch[:4]) + sum(crouch[n - 4:])) / 8

        if mid_crouch > 0.25 and mid_crouch > edge_crouch + 0.08:
            return "weaving"
        return None


_KP_SMOOTHER = KeypointSmoother(alpha=0.5)
_SEQ_DETECTOR = SequenceDetector(window=20)


def _refine_wide_stance(f: "dict[str, float]", det: Detection) -> "str | None":
    """슬라이드 vs 넥디렉션 구분.

    하체: 동일 (넓은 스탠스 + 한쪽 무릎 굽힘)
    슬라이드: 상체가 굽힌 무릎과 같은 방향 (바깥쪽)
    넥디렉션: 상체가 굽힌 무릎과 반대 방향 (안쪽/중앙)
    """
    kpts = det.keypoints
    if kpts is None:
        return None
    lk = _kp(kpts, _LK)
    rk = _kp(kpts, _RK)
    ls = _kp(kpts, _LS)
    rs = _kp(kpts, _RS)
    lh = _kp(kpts, _LH)
    rh = _kp(kpts, _RH)
    if not (lk and rk and ls and rs):
        return None

    sh_cx = (ls[0] + rs[0]) / 2
    hi_cx = (lh[0] + rh[0]) / 2 if (lh and rh) else sh_cx
    # 양수 = 상체가 오른쪽, 음수 = 상체가 왼쪽
    signed_lean = (sh_cx - hi_cx) / max(f.get("scale", 50.0), 1.0)

    # 굽힌 무릎 방향: y 큰 쪽(더 아래)이 더 굽힘 → -1=왼쪽, +1=오른쪽
    bent_dir = -1 if lk[1] > rk[1] else +1

    dot = signed_lean * bent_dir
    if dot >  0.10:   # 상체·무릎 같은 방향 → 슬라이드
        return "slide"
    if dot < -0.10:   # 상체·무릎 반대 방향 → 넥디렉션
        return "next_direction"
    return None       # 중간값 → 판단 보류


# ── 분류기 ──────────────────────────────────────────────────────
def classify_hado_movement(
    det: Detection,
    frame: Optional["np.ndarray"] = None,
) -> Optional[MovementResult]:
    """Detection keypoints → HADO 7가지 기본동작 분류.

    분류기 우선순위:
      1. YOLOv8-cls  (frame 제공 + models/hado_movement_cls.pt 존재 시)
      2. 시퀀스 감지 (위빙 — 최근 20프레임 lateral_lean 패턴)
      3. sklearn pkl (models/hado_movement_clf.pkl 존재 시)
      4. 규칙 기반   (fallback)

    Parameters
    ----------
    det   : keypoints를 포함한 Detection 객체
    frame : 원본 영상 프레임 (YOLO-cls에 필요, 없으면 생략 가능)

    Returns
    -------
    MovementResult 또는 None (keypoints 없을 때)
    """
    # 1순위: YOLO-cls (이미지 기반)
    if frame is not None:
        yolo_result = _classify_yolo_cls(frame, det)
        if yolo_result is not None:
            return yolo_result

    # 키포인트 EMA 평활화 (관절 떨림 감소)
    if det.keypoints is not None:
        sk = _KP_SMOOTHER.smooth(det.keypoints)
        det = Detection(det.x1, det.y1, det.x2, det.y2, det.confidence, sk)
    else:
        _SEQ_DETECTOR.reset()
        return None

    f = _extract_features(det)
    if f is None:
        return None

    # 2순위: 시퀀스 감지 (위빙 원형 동작)
    seq_result = _SEQ_DETECTOR.update(f)
    if seq_result is not None:
        return MovementResult(seq_result, 0.82, f)

    # 3순위: sklearn pkl (슬라이드/넥디렉션 보정 포함)
    ml_result = _classify_ml(f, det)
    if ml_result is not None:
        if ml_result.movement in ("slide", "next_direction"):
            refined = _refine_wide_stance(f, det)
            if refined:
                return MovementResult(refined, ml_result.confidence, f)
        return ml_result

    # 4순위: 규칙 기반 (7가지 유효 라벨만)
    fp  = f["floor_proximity"]
    bh  = f["bbox_horizontal"]
    nl  = f["nose_level"]
    sw  = f["stance_width"]
    aa  = f["ankle_asym"]
    cd  = f["crouch_depth"]
    ka  = f["knee_asym"]
    ffa = f["foot_fore_aft"]
    kbd = f["knee_bend_diff"]

    # ── 1. 버피테스트: 수평 자세 ─────────────────────────────
    if bh > 0.28 or fp > 0.65 or nl > 0.75:
        conf = max(bh * 1.5, fp, nl)
        return MovementResult("burpee", round(min(1.0, conf), 3), f)

    # ── 2. 슬라이드 / 넥디렉션: 넓은 스탠스 + 무릎 비대칭 ──
    if sw > 1.6 and (ka > 0.18 or aa > 0.14) and cd > 0.35:
        label = _refine_wide_stance(f, det) or "slide"
        conf  = min(1.0, (sw - 1.6) * 1.5 + max(ka, aa))
        return MovementResult(label, round(conf, 3), f)

    # ── 3. 스쿼트: 넓고 대칭, CoM 낮음 ─────────────────────
    if sw > 1.0 and cd > 0.35 and ka < 0.22:
        conf = min(1.0, (cd - 0.35) * 2.5 + (sw - 1.0) * 0.4)
        return MovementResult("squat", round(conf, 3), f)

    # ── 4. 런지: 전후 비대칭 + 한쪽 무릎 굽힘 ──────────────
    if ffa > 0.20 and ka > 0.12 and cd > 0.25:
        conf = min(1.0, max(kbd / 80.0, ffa, ka))
        return MovementResult("lunge", round(max(0.4, conf), 3), f)

    # ── 5. 준비 자세 (위빙은 SequenceDetector 에서 처리) ────
    standingness = 1.0 - max(cd, fp, bh)
    return MovementResult("ready", round(max(0.1, standingness), 3), f)


# ── 시각화 헬퍼 ─────────────────────────────────────────────────
def draw_movement_label(
    img: "np.ndarray",
    det: Detection,
    result: MovementResult,
    font_size: int = 16,
) -> None:
    """바운딩 박스 위에 동작 레이블 표시."""
    import cv2
    color = MOVEMENT_COLOR.get(result.movement, (180, 180, 180))
    emoji = MOVEMENT_EMOJI.get(result.movement, "?")
    label = MOVEMENT_KO.get(result.movement, result.movement)
    text  = f"{emoji} {label} ({result.confidence:.0%})"
    x, y  = int(det.x1), max(0, int(det.y1) - 5)
    try:
        from src.annotate import put_text_kr
        put_text_kr(img, text, (x, y - font_size), font_size, color)
    except Exception:
        cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                    0.6, color, 2, cv2.LINE_AA)


# ── 단독 테스트 ──────────────────────────────────────────────────
def main() -> None:
    """정적 이미지 / 웹캠에서 HADO 동작 분류 단독 테스트."""
    import argparse
    import cv2
    from src.detector import PersonDetector
    from src.pose import draw_skeleton

    parser = argparse.ArgumentParser(description="HADO 동작 분류 단독 테스트")
    parser.add_argument("--source", default="0")
    parser.add_argument("--model",  default="yolov8n-pose.pt")
    parser.add_argument("--imgsz",  type=int, default=320)
    args = parser.parse_args()

    detector = PersonDetector(model_path=args.model, imgsz=args.imgsz)
    try:
        src = int(args.source)
    except ValueError:
        src = args.source

    cap = cv2.VideoCapture(src)
    print("[HADOMovement] 실행 — ESC로 종료")
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        for det in detector.detect(frame):
            draw_skeleton(frame, det)
            res = classify_hado_movement(det, frame=frame)
            if res:
                draw_movement_label(frame, det, res)
        cv2.imshow("HADO Movement", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
