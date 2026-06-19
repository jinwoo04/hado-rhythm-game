"""웹캠에서 동작별 keypoint feature 수집 → sklearn 즉시 재학습.

동작당 10초씩 웹캠으로 직접 수집하여 학습 데이터 도메인 미스매치 해소.

실행:
    python -m tools.collect_and_train                        # 전체 7동작
    python -m tools.collect_and_train --sec 15               # 동작당 15초
    python -m tools.collect_and_train --movements squat lunge slide
    python -m tools.collect_and_train --source 1             # 카메라 인덱스 1
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── 동작 목록 (label, 한글, 힌트) ───────────────────────────────
ALL_MOVEMENTS = [
    ("ready",          "준비 자세",    "편하게 자연스럽게 서 있어요"),
    ("squat",          "스쿼트",       "양발 넓게, 무릎 굽혀 낮추기"),
    ("lunge",          "런지",         "한발 앞으로 내딛고 앞무릎 90도"),
    ("slide",          "슬라이드",     "옆으로 넓게, 한쪽 무릎 깊게 굽힘"),
    ("next_direction", "넥디렉션",     "슬라이드 자세에서 상체를 반대쪽으로"),
    ("weaving",        "위빙",         "상체로 좌→하→우 원형 반복"),
    ("burpee",         "버피테스트",   "바닥 엎드렸다 일어서기 반복"),
]

# 학습에 사용할 특징 (scale·body_h 는 정규화 보조값이라 제외)
FEATURE_NAMES = [
    "floor_proximity", "bbox_horizontal", "nose_level",
    "stance_width", "ankle_asym", "knee_raise", "crouch_depth",
    "knee_asym", "foot_fore_aft", "knee_bend_avg", "knee_bend_diff",
    "lateral_lean", "lateral_lean_signed", "hand_low", "bbox_aspect",
]

COUNTDOWN_SEC  = 3
DEFAULT_STRIDE = 4   # N프레임마다 1개만 학습 사용 (연속 프레임 중복 감소)

# ── 색상 ────────────────────────────────────────────────────────
_GREEN  = (60, 220, 60)
_YELLOW = (40, 210, 210)
_WHITE  = (220, 220, 220)
_GRAY   = (130, 130, 130)
_RED    = (60, 60, 210)


def _txt(img, text, xy, size_px, color):
    """한글 포함 텍스트 렌더링. PIL 실패 시 ASCII fallback."""
    try:
        from src.annotate import put_text_kr
        put_text_kr(img, text, xy, size_px, color)
    except Exception:
        cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                    size_px / 30, color, 2, cv2.LINE_AA)


def _progress_bar(img, elapsed, total, color):
    h, w = img.shape[:2]
    y0 = h - 16
    ratio = min(1.0, elapsed / max(1e-6, total))
    cv2.rectangle(img, (0, y0), (w, h),          (30, 30, 30), -1)
    cv2.rectangle(img, (0, y0), (int(w * ratio), h), color,    -1)


# ── 단일 동작 수집 ────────────────────────────────────────────────
def collect_one(
    cap: cv2.VideoCapture,
    detector,
    label: str,
    label_ko: str,
    hint: str,
    record_sec: float,
    win: str,
) -> list[dict[str, float]]:
    from src.detector import Detection
    from src.hado_movement import _KP_SMOOTHER, _extract_features
    from src.pose import draw_skeleton

    _KP_SMOOTHER.reset()
    samples: list[dict[str, float]] = []

    # ── 카운트다운 ───────────────────────────────────────────────
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        left = COUNTDOWN_SEC - (time.time() - t0)
        if left <= 0:
            break

        ov = frame.copy()
        cv2.rectangle(ov, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(ov, 0.45, frame, 0.55, 0, frame)

        _txt(frame, label_ko,          (w // 2 - 130, h // 2 - 80), 40, _YELLOW)
        _txt(frame, hint,              (w // 2 - 200, h // 2 - 20), 18, _GRAY)
        _txt(frame, str(int(left)+1),  (w // 2 -  32, h // 2 + 60), 80, _WHITE)
        _txt(frame, "READY...",        (w // 2 -  60, h // 2 + 150), 20, _GRAY)

        cv2.imshow(win, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            return []

    # ── 수집 ─────────────────────────────────────────────────────
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame = cv2.flip(frame, 1)
        h, w = frame.shape[:2]

        elapsed = time.time() - t0
        if elapsed >= record_sec:
            break

        dets = detector.detect(frame)
        target = max(dets, key=lambda d: d.area) if dets else None

        if target is not None:
            draw_skeleton(frame, target)
            if target.keypoints is not None:
                sk = _KP_SMOOTHER.smooth(target.keypoints)
                sdet = Detection(
                    target.x1, target.y1, target.x2, target.y2,
                    target.confidence, sk,
                )
                f = _extract_features(sdet)
                if f is not None:
                    samples.append(f)

        _txt(frame, label_ko, (w // 2 - 130, 20), 36, _GREEN)
        _txt(frame, f"REC  {max(0, int(record_sec - elapsed) + 1)}s"
                    f"  ({len(samples)} frames)",
             (12, h - 28), 18, _GREEN)
        _progress_bar(frame, elapsed, record_sec, _GREEN)

        cv2.imshow(win, frame)
        if cv2.waitKey(1) & 0xFF == 27:
            return []

    print(f"  [{label}] {len(samples)} 프레임 수집 완료")
    return samples


# ── 학습 ─────────────────────────────────────────────────────────
def train_and_save(all_samples: list[tuple[str, dict[str, float]]]) -> None:
    import warnings
    warnings.filterwarnings("ignore")
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import LabelEncoder

    X = np.array(
        [[f.get(k, 0.0) for k in FEATURE_NAMES] for _, f in all_samples],
        dtype=np.float32,
    )
    y = np.array([lbl for lbl, _ in all_samples])

    le = LabelEncoder()
    y_enc = le.fit_transform(y)

    n_cls = len(set(y))
    cv_k  = min(5, min(np.bincount(y_enc)))   # 가장 적은 클래스 샘플 수 기준
    clf   = RandomForestClassifier(
        n_estimators=300, max_depth=None,
        min_samples_leaf=2, random_state=42, n_jobs=-1,
    )

    print(f"\n[Train] 클래스: {list(le.classes_)}")
    print(f"[Train] 총 샘플: {len(X)}  |  CV k={cv_k}")
    scores = cross_val_score(clf, X, y_enc, cv=cv_k, scoring="accuracy")
    print(f"[Train] CV 정확도: {scores.mean():.1%} ± {scores.std():.1%}")

    clf.fit(X, y_enc)

    out = PROJECT_ROOT / "models/hado_movement_clf.pkl"
    with open(out, "wb") as fh:
        pickle.dump({
            "clf":           clf,
            "le":            le,
            "feature_names": FEATURE_NAMES,
            "class_names":   list(le.classes_),
            "n_samples":     len(X),
        }, fh)
    print(f"[Train] 저장: {out}\n")


# ── 메인 ─────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="웹캠 동작 수집 → 즉시 재학습")
    parser.add_argument("--source",    default="0")
    parser.add_argument("--sec",       type=float, default=10.0,
                        help="동작당 수집 시간(초), 기본 10")
    parser.add_argument("--movements", nargs="+",
                        help="수집할 동작만 지정 (기본: 전체 7개)")
    parser.add_argument("--imgsz",     type=int,   default=320)
    parser.add_argument("--conf",      type=float, default=0.40)
    parser.add_argument("--device",    default="cpu")
    args = parser.parse_args()

    from src.detector import PersonDetector

    # 모델 선택
    ncnn = PROJECT_ROOT / "yolov8n-pose_ncnn_model"
    onnx = PROJECT_ROOT / "yolov8n-pose.onnx"
    model_path = (
        str(ncnn) if ncnn.exists() else
        str(onnx) if onnx.exists() else
        "yolov8n-pose.pt"
    )
    detector = PersonDetector(model_path=model_path, imgsz=args.imgsz,
                              conf_threshold=args.conf, device=args.device)

    try:
        source = int(args.source)
    except ValueError:
        source = args.source
    cap = cv2.VideoCapture(source)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

    win = "HADO — 동작 수집"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 960, 540)

    # 수집할 동작 필터
    targets = args.movements or [m[0] for m in ALL_MOVEMENTS]
    movements = [m for m in ALL_MOVEMENTS if m[0] in targets]

    print(f"[Collect] 동작당 {args.sec:.0f}초  |  {len(movements)}개 동작")
    print("[Collect] ESC로 중단\n")

    all_samples: list[tuple[str, dict[str, float]]] = []

    for label, label_ko, hint in movements:
        print(f"[Collect] {label_ko} ({label}) 수집 중...")
        frames = collect_one(cap, detector, label, label_ko, hint, args.sec, win)
        if not frames:
            print("[Collect] 중단됨")
            break
        for f in frames:
            all_samples.append((label, f))

    cap.release()
    cv2.destroyAllWindows()

    if not all_samples:
        print("[Collect] 수집된 데이터 없음. 종료.")
        return

    # ── 샘플 수 요약 ─────────────────────────────────────────────
    from collections import Counter
    counts = Counter(lbl for lbl, _ in all_samples)
    print("\n[Collect] 수집 결과:")
    for lbl, cnt in sorted(counts.items()):
        print(f"  {lbl:18s}: {cnt} 프레임")

    train_and_save(all_samples)
    print("[Collect] 완료! 이제 rhythm_game_demo 를 실행해보세요.")


if __name__ == "__main__":
    main()
