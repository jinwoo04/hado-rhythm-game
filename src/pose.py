"""YOLOv8-pose 키포인트 기반 자세 분석 모듈.

일반 카메라로 촬영된 선수 영상에서 자세(크라우치·쉴드준비·중립)를 분류하고
다음 동작을 보조 예측한다.

COCO 17 키포인트 (인덱스 기준)
-------------------------------
0:코   1:왼눈  2:오른눈  3:왼귀  4:오른귀
5:왼어깨  6:오른어깨  7:왼팔꿈치  8:오른팔꿈치
9:왼손목  10:오른손목  11:왼엉덩이  12:오른엉덩이
13:왼무릎  14:오른무릎  15:왼발목  16:오른발목
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import cv2
import numpy as np

from src.detector import Detection

# ── 키포인트 인덱스 ────────────────────────────────────────────
_N  = 0                           # 코
_LS, _RS = 5, 6                   # 어깨
_LE, _RE = 7, 8                   # 팔꿈치
_LW, _RW = 9, 10                  # 손목
_LH, _RH = 11, 12                 # 엉덩이
_LK, _RK = 13, 14                 # 무릎
_LA, _RA = 15, 16                 # 발목

# ── 스켈레톤 연결 ──────────────────────────────────────────────
SKELETON_PAIRS: list[tuple[int, int]] = [
    (_LS, _RS),                    # 어깨
    (_LS, _LE), (_LE, _LW),        # 왼팔
    (_RS, _RE), (_RE, _RW),        # 오른팔
    (_LS, _LH), (_RS, _RH),        # 상체 옆면
    (_LH, _RH),                    # 엉덩이
    (_LH, _LK), (_LK, _LA),        # 왼다리
    (_RH, _RK), (_RK, _RA),        # 오른다리
    (_N,  _LS), (_N,  _RS),        # 목 (코→어깨 근사)
]

# 몸통(흰색) / 팔(파랑) / 다리(초록) 색상 그룹
_LIMB_COLORS: dict[tuple[int, int], tuple[int, int, int]] = {
    (_LS, _RS):   (220, 220, 220),
    (_LS, _LH):   (220, 220, 220),
    (_RS, _RH):   (220, 220, 220),
    (_LH, _RH):   (220, 220, 220),
    (_N,  _LS):   (220, 220, 220),
    (_N,  _RS):   (220, 220, 220),
    (_LS, _LE):   (255, 140, 60),
    (_LE, _LW):   (255, 140, 60),
    (_RS, _RE):   (60,  140, 255),
    (_RE, _RW):   (60,  140, 255),
    (_LH, _LK):   (80,  220, 100),
    (_LK, _LA):   (80,  220, 100),
    (_RH, _RK):   (80,  200, 255),
    (_RK, _RA):   (80,  200, 255),
}

# ── 자세 레이블 ────────────────────────────────────────────────
POSTURE_KO = {
    "attack":  "공격 자세",
    "shield":  "쉴드 준비",
    "neutral": "중립",
}

POSTURE_COLOR: dict[str, tuple[int, int, int]] = {
    "attack":  (40,  160, 255),   # 주황
    "shield":  (60,  180, 100),   # 초록
    "neutral": (160, 160, 160),   # 회색
}

# 자세 → 예상 의도 매핑 (TacticEngine 보조 신호)
POSTURE_TO_INTENT = {
    "attack":  "direct_attack",
    "shield":  "shield_protect",
    "neutral": "",
}

# ── HADO 7동작 분류기 ────────────────────────────────────────────
ACTION_KO: dict[str, str] = {
    "charge":  "차지 준비",    # 팔 위로 들어올림 (발사 직전)
    "shoot":   "공격 발사",    # 팔 앞으로 뻗음 (팀 방향 기준)
    "shield":  "쉴드 방어",
    "dodge_l": "회피 좌",
    "dodge_r": "회피 우",
    "crouch":  "슬라이딩",
    "ready":   "준비 자세",
}

ACTION_COLOR: dict[str, tuple[int, int, int]] = {
    "charge":  (0,   180, 255),   # 하늘색
    "shoot":   (30,  120, 255),   # 주황
    "shield":  (40,  200,  80),   # 초록
    "dodge_l": (0,   220, 255),   # 노랑
    "dodge_r": (255, 200,   0),   # 하늘
    "crouch":  (200,  60, 255),   # 보라
    "ready":   (160, 160, 160),   # 회색
}

ACTION_EMOJI: dict[str, str] = {
    "charge":  "↑",
    "shoot":   "⚡",
    "shield":  "🛡",
    "dodge_l": "←",
    "dodge_r": "→",
    "crouch":  "↓",
    "ready":   "●",
}


from dataclasses import dataclass as _dc

@_dc
class ActionResult:
    """HADO 동작 분류 결과."""
    action: str             # ACTION_KO 키
    confidence: float       # 0.0~1.0
    scores: "dict[str, float]"  # 각 동작별 원시 점수


@_dc
class NextActionRec:
    """다음 동작 추천 항목."""
    action: str       # ACTION_KO 키
    reason: str       # 화면 표시용 한글 설명
    priority: float   # 0.0~1.0 (높을수록 강하게 추천)


# 현재 동작 → 다음 동작 전환 추천 테이블 (HADO 전술 기반)
NEXT_ACTION_RECS: dict[str, list[tuple[str, str, float]]] = {
    "charge":  [("shoot",   "차지 완료 → 발사",        1.0),
                ("dodge_l", "발사 포기 → 좌측 회피",   0.3)],
    "shoot":   [("dodge_r", "발사 후 우측 회피",        0.8),
                ("dodge_l", "발사 후 좌측 회피",        0.7),
                ("shield",  "발사 후 방어 준비",         0.4)],
    "shield":  [("charge",  "방어 해제 → 역공 차지",    0.9),
                ("dodge_l", "방어 중 측면 이동",         0.4)],
    "dodge_l": [("charge",  "회피 후 역공 차지",        0.8),
                ("shield",  "이동 후 방어 자세",         0.4)],
    "dodge_r": [("charge",  "회피 후 역공 차지",        0.8),
                ("shield",  "이동 후 방어 자세",         0.4)],
    "crouch":  [("charge",  "일어나서 차지 시작",        0.7),
                ("ready",   "자세 회복",                 0.4)],
    "ready":   [("charge",  "차지 시작",                 0.7),
                ("shield",  "방어 자세 취하기",          0.5)],
}


def classify_hado_action(
    det: "Detection",
    frame_center_x: float = 320.0,
) -> "ActionResult | None":
    """Detection 키포인트 → HADO 7동작 분류.

    Parameters
    ----------
    det             : 키포인트를 포함한 Detection 객체
    frame_center_x  : 프레임 중앙 x픽셀 (팀 방향 판단용, 기본 320)

    판정 우선순위: 슬라이딩 > 차지 > 공격발사 > 쉴드방어 > 회피좌/우 > 준비

    스케일 정규화:
        scale = max(어깨폭, 몸통높이×0.6, 30px)
        → 카메라 거리에 관계없이 동일 임계값 적용
    """
    if det.keypoints is None:
        return None

    kpts   = det.keypoints
    bbox_h = max(1.0, det.y2 - det.y1)
    bbox_w = max(1.0, det.x2 - det.x1)

    def _k(i: int) -> "tuple[float,float] | None":
        if kpts[i, 2] >= _KP_CONF_MIN:
            return float(kpts[i, 0]), float(kpts[i, 1])
        return None

    ls, rs = _k(_LS), _k(_RS)
    le, re = _k(_LE), _k(_RE)
    lw, rw = _k(_LW), _k(_RW)
    lh, rh = _k(_LH), _k(_RH)
    lk, rk = _k(_LK), _k(_RK)

    # ── 스케일: 어깨폭 + 몸통높이 기반 정규화 ─────────────────
    shoulder_w = abs(rs[0] - ls[0]) if (ls and rs) else 0.0
    torso_h = 0.0
    if ls and rs and lh and rh:
        sh_y = (ls[1] + rs[1]) / 2
        hi_y = (lh[1] + rh[1]) / 2
        torso_h = abs(hi_y - sh_y)
    scale = max(shoulder_w, torso_h * 0.6, 30.0)

    # 팀 방향: 팀A(왼쪽, x<center)→+1(오른쪽이 적 방향), 팀B→-1
    center_x = (det.x1 + det.x2) / 2
    facing = +1.0 if center_x < frame_center_x else -1.0

    # ── 슬라이딩 점수: 어깨~무릎 수직 압축 ─────────────────────
    crouch = 0.0
    if ls and rs and lk and rk:
        sh_y = (ls[1] + rs[1]) / 2
        kn_y = (lk[1] + rk[1]) / 2
        span = kn_y - sh_y
        crouch = max(0.0, 1.0 - span / (0.45 * bbox_h))

    # ── 차지 점수: 손목이 어깨보다 위, 거의 수직 방향 ──────────
    # scale 기반: dy/scale < -0.55 and |dx/scale| < 0.70
    charge = 0.0
    for wrist, shoulder in [(lw, ls), (rw, rs)]:
        if wrist and shoulder:
            dx = (wrist[0] - shoulder[0]) / scale
            dy = (wrist[1] - shoulder[1]) / scale   # 위 = 음수
            if dy < -0.55 and abs(dx) < 0.70:
                charge = max(charge, min(1.0, abs(dy) - 0.55 + 0.5))

    # ── 공격발사 점수: 손목이 팀 방향(앞)으로 뻗임 ─────────────
    # scale 기반: forward/scale > 0.65, |dy/scale| < 0.50
    shoot = 0.0
    for wrist, shoulder in [(lw, ls), (rw, rs)]:
        if wrist and shoulder:
            dx_raw  = (wrist[0] - shoulder[0]) / scale
            dy_norm = abs((wrist[1] - shoulder[1]) / scale)
            forward = dx_raw * facing   # 팀 방향 기준 앞쪽이 양수
            if forward > 0.65 and dy_norm < 0.50:
                shoot = max(shoot, min(1.0, forward - 0.65 + 0.5))

    # ── 쉴드방어 점수: 양팔 벌림 + 양 손목 팔꿈치 위 ───────────
    spread = 0.0
    if lw and ls:
        spread = max(spread, abs(lw[0] - ls[0]) / bbox_w)
    if rw and rs:
        spread = max(spread, abs(rw[0] - rs[0]) / bbox_w)
    lw_raised = bool(lw and le and lw[1] < le[1])
    rw_raised = bool(rw and re and rw[1] < re[1])
    shield = spread * (1.2 if (lw_raised and rw_raised) else 0.6)

    # ── 회피 점수: 어깨 중점이 엉덩이 중점에서 수평으로 벗어남 ─
    dodge_l = dodge_r = 0.0
    if ls and rs and lh and rh:
        sh_cx = (ls[0] + rs[0]) / 2
        hi_cx = (lh[0] + rh[0]) / 2
        lean  = (sh_cx - hi_cx) / bbox_w
        if lean < 0:
            dodge_l = min(1.0, abs(lean) * 3)
        else:
            dodge_r = min(1.0, lean * 3)

    scores = {
        "crouch":  round(crouch,  3),
        "charge":  round(charge,  3),
        "shoot":   round(shoot,   3),
        "shield":  round(shield,  3),
        "dodge_l": round(dodge_l, 3),
        "dodge_r": round(dodge_r, 3),
        "ready":   0.0,
    }

    # 우선순위 판정
    if crouch > 0.55:
        action, conf = "crouch", crouch
    elif charge > 0.30:
        action, conf = "charge", min(1.0, charge)
    elif shoot > 0.30:
        action, conf = "shoot", min(1.0, shoot)
    elif shield > 0.55:
        action, conf = "shield", min(1.0, shield)
    elif dodge_l > 0.30:
        action, conf = "dodge_l", dodge_l
    elif dodge_r > 0.30:
        action, conf = "dodge_r", dodge_r
    else:
        action, conf = "ready", 1.0 - max(crouch, charge, shoot, shield, dodge_l, dodge_r)

    return ActionResult(action=action, confidence=round(conf, 3), scores=scores)


def recommend_next_action(
    current_action: str,
    history: "list[str] | None" = None,
) -> "list[NextActionRec]":
    """현재 동작 + 최근 히스토리 → 다음 동작 추천 (우선순위 내림차순, 최대 2개).

    Parameters
    ----------
    current_action : 현재 확정된 동작 레이블 (ACTION_KO 키)
    history        : 최근 N프레임 동작 레이블 리스트 (오래된 것 먼저)
    """
    base = NEXT_ACTION_RECS.get(current_action, [])
    recs = [NextActionRec(action=a, reason=r, priority=p) for a, r, p in base]

    # 같은 동작이 4프레임 이상 지속되면 전환 추천 가중치 +20%
    if history and len(history) >= 4 and all(a == current_action for a in history[-4:]):
        recs = [NextActionRec(action=r.action, reason=r.reason,
                              priority=min(1.0, r.priority * 1.2)) for r in recs]

    recs.sort(key=lambda r: r.priority, reverse=True)
    return recs[:2]


def sample_vest_hue(frame: "np.ndarray", det: "Detection") -> int:
    """선수 조끼 색상(HSV hue) 샘플링.

    토르소 영역(어깨 + 엉덩이 키포인트 bounding box)을 HSV로 변환하고
    채도·명도 필터를 통과한 픽셀의 원형 평균 hue를 반환한다.

    Returns
    -------
    int : hue 0–179 (OpenCV 기준), 또는 -1 (키포인트 불충분 / 픽셀 부족)
    """
    import math as _math

    if det.keypoints is None:
        return -1

    kpts = det.keypoints
    h_img, w_img = frame.shape[:2]

    def _kp(i: int):
        if kpts[i, 2] >= 0.30:
            return float(kpts[i, 0]), float(kpts[i, 1])
        return None

    ls, rs = _kp(_LS), _kp(_RS)
    lh, rh = _kp(_LH), _kp(_RH)

    if not (ls and rs and lh and rh):
        return -1

    all_x = [ls[0], rs[0], lh[0], rh[0]]
    all_y = [ls[1], rs[1], lh[1], rh[1]]
    x1 = int(max(0, min(all_x)))
    y1 = int(max(0, min(ls[1], rs[1])))
    x2 = int(min(w_img, max(all_x)))
    y2 = int(min(h_img, max(lh[1], rh[1])))

    if x2 - x1 < 6 or y2 - y1 < 10:
        return -1

    torso = frame[y1:y2, x1:x2]
    hsv   = cv2.cvtColor(torso, cv2.COLOR_BGR2HSV)

    # 채도 > 70, 명도 60~240 → 그림자·하이라이트 제외
    sat_mask = (hsv[..., 1] > 70) & (hsv[..., 2] > 60) & (hsv[..., 2] < 240)
    if sat_mask.sum() < 25:
        return -1

    hues = hsv[..., 0][sat_mask].astype(np.float32)
    # 원형 평균 hue (0↔180 wrap 처리)
    rad  = hues * (2 * _math.pi / 180.0)
    cs   = float(np.cos(rad).mean())
    sn   = float(np.sin(rad).mean())
    mean_rad = _math.atan2(sn, cs)
    mean_hue = (mean_rad * 180.0 / (2 * _math.pi)) % 180.0
    return int(round(mean_hue))


# ── 데이터클래스 ───────────────────────────────────────────────
@dataclass
class PoseFeatures:
    """단일 선수의 자세 특징값."""
    keypoints:    np.ndarray       # (17, 3) — x, y, conf
    crouch_score: float            # 0.0~1.0 (높을수록 낮은 자세)
    arm_spread:   float            # 0.0~1.0 (팔을 크게 벌린 정도)
    posture:      str              # "attack" | "shield" | "neutral"
    intent_hint:  str = field(default="")  # POSTURE_TO_INTENT 매핑값


# ── 자세 분석 ──────────────────────────────────────────────────
_KP_CONF_MIN = 0.25   # 이 신뢰도 미만 키포인트는 무시


def _kp(kpts: np.ndarray, idx: int) -> Optional[tuple[float, float]]:
    """신뢰도 통과한 키포인트 좌표 반환."""
    if kpts[idx, 2] >= _KP_CONF_MIN:
        return float(kpts[idx, 0]), float(kpts[idx, 1])
    return None


def analyze_pose(det: Detection) -> Optional[PoseFeatures]:
    """Detection에서 자세 특징 추출. keypoints 없으면 None 반환."""
    if det.keypoints is None:
        return None

    kpts = det.keypoints           # (17, 3)
    bbox_h = max(1.0, det.y2 - det.y1)
    bbox_w = max(1.0, det.x2 - det.x1)

    # ── 크라우치 점수 ──────────────────────────────────────────
    # 어깨 중점 y ~ 발목 중점 y 의 수직 간격 / bbox 높이
    # 직립: ~0.65×bbox_h  /  크라우치: ~0.35×bbox_h
    sh_l, sh_r = _kp(kpts, _LS), _kp(kpts, _RS)
    an_l, an_r = _kp(kpts, _LA), _kp(kpts, _RA)

    crouch_score = 0.0
    if sh_l and sh_r and an_l and an_r:
        sh_y  = (sh_l[1] + sh_r[1]) / 2
        an_y  = (an_l[1] + an_r[1]) / 2
        span  = an_y - sh_y            # 양수 (발목이 아래)
        crouch_score = float(max(0.0, min(1.0, 1.0 - span / (0.60 * bbox_h))))
    elif sh_l and sh_r:
        # 발목 미감지: 무릎으로 대체
        kn_l, kn_r = _kp(kpts, _LK), _kp(kpts, _RK)
        if kn_l and kn_r:
            sh_y = (sh_l[1] + sh_r[1]) / 2
            kn_y = (kn_l[1] + kn_r[1]) / 2
            crouch_score = float(max(0.0, min(1.0, 1.0 - (kn_y - sh_y) / (0.40 * bbox_h))))

    # ── 팔 벌림 (쉴드 준비 지표) ───────────────────────────────
    # 손목과 어깨 사이 수평 거리 / bbox 폭
    arm_spread = 0.0
    lw = _kp(kpts, _LW)
    rw = _kp(kpts, _RW)
    if lw and sh_l:
        arm_spread = max(arm_spread, abs(lw[0] - sh_l[0]) / bbox_w)
    if rw and sh_r:
        arm_spread = max(arm_spread, abs(rw[0] - sh_r[0]) / bbox_w)

    # ── 자세 분류 ──────────────────────────────────────────────
    if arm_spread > 0.50 and crouch_score < 0.40:
        posture = "shield"
    elif crouch_score > 0.45:
        posture = "attack"
    else:
        posture = "neutral"

    return PoseFeatures(
        keypoints=kpts,
        crouch_score=round(crouch_score, 3),
        arm_spread=round(arm_spread, 3),
        posture=posture,
        intent_hint=POSTURE_TO_INTENT.get(posture, ""),
    )


# ── 시각화 ─────────────────────────────────────────────────────
def draw_skeleton(
    img: np.ndarray,
    det: Detection,
    base_color: tuple[int, int, int] = (200, 200, 200),
    use_limb_colors: bool = True,
) -> None:
    """카메라 뷰에 스켈레톤 오버레이 (in-place).

    Parameters
    ----------
    base_color      : use_limb_colors=False 일 때 단색
    use_limb_colors : True면 팔·다리별 색상 구분
    """
    if det.keypoints is None:
        return

    kpts = det.keypoints

    # 관절 연결선
    for a, b in SKELETON_PAIRS:
        pa, pb = _kp(kpts, a), _kp(kpts, b)
        if pa and pb:
            color = (_LIMB_COLORS.get((a, b)) or base_color) if use_limb_colors else base_color
            cv2.line(img,
                     (int(pa[0]), int(pa[1])),
                     (int(pb[0]), int(pb[1])),
                     color, 2, cv2.LINE_AA)

    # 관절 점
    for i in range(17):
        p = _kp(kpts, i)
        if p:
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (255, 255, 255), -1, cv2.LINE_AA)


def draw_keypoint_ids(
    img: np.ndarray,
    det: Detection,
    conf_min: float = 0.25,
) -> None:
    """스켈레톤 위에 키포인트 인덱스(0–16) 표시 (발표·디버그용)."""
    if det.keypoints is None:
        return
    kpts = det.keypoints
    for i in range(17):
        if kpts[i, 2] >= conf_min:
            x, y = int(kpts[i, 0]), int(kpts[i, 1])
            cv2.putText(img, str(i), (x + 4, y - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 0), 1, cv2.LINE_AA)


def draw_posture_label(
    img: np.ndarray,
    det: Detection,
    features: PoseFeatures,
) -> None:
    """바운딩 박스 상단에 자세 분류 라벨 표시 (한글 PIL)."""
    try:
        from src.annotate import put_text_kr
        color = POSTURE_COLOR.get(features.posture, (200, 200, 200))
        label = POSTURE_KO.get(features.posture, features.posture)
        score_txt = f"{label} ({features.crouch_score:.2f})"
        put_text_kr(img, score_txt, (int(det.x1), max(0, int(det.y1) - 18)), 13, color)
    except Exception:
        pass


def draw_movement_arrow(
    img: np.ndarray,
    track_history: list[tuple[float, float]],
    color: tuple[int, int, int] = (0, 220, 255),
    predict_frames: int = 12,
) -> None:
    """트랙 히스토리로 이동 벡터 → 예측 위치 화살표 표시."""
    if len(track_history) < 4:
        return
    # 최근 4 프레임 평균 속도
    recent = track_history[-4:]
    vx = (recent[-1][0] - recent[0][0]) / 3
    vy = (recent[-1][1] - recent[0][1]) / 3
    if abs(vx) < 1.0 and abs(vy) < 1.0:
        return   # 거의 정지

    cx, cy = int(recent[-1][0]), int(recent[-1][1])
    px = int(cx + vx * predict_frames)
    py = int(cy + vy * predict_frames)
    cv2.arrowedLine(img, (cx, cy), (px, py), color,
                    2, tipLength=0.35, line_type=cv2.LINE_AA)


# ── 단독 테스트 ────────────────────────────────────────────────
def main() -> None:
    """yolov8n-pose로 단일 이미지/영상 자세 분석."""
    import argparse
    from src.detector import PersonDetector

    parser = argparse.ArgumentParser(description="Pose 분석 테스트")
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
    if not cap.isOpened():
        print(f"[Pose] 소스 열기 실패: {src}")
        return

    print("[Pose] 실행 중 — ESC로 종료")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        dets = detector.detect(frame)
        for det in dets:
            cv2.rectangle(frame,
                          (int(det.x1), int(det.y1)),
                          (int(det.x2), int(det.y2)),
                          (80, 80, 80), 1)
            draw_skeleton(frame, det)
            feat = analyze_pose(det)
            if feat:
                draw_posture_label(frame, det, feat)

        cv2.imshow("Pose Analysis (ESC to quit)", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
