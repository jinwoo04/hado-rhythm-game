"""HADO 기본동작 분류기 학습 도구.

label_movements.py로 생성한 labels.csv를 읽어
YOLOv8n-pose keypoint 특징값으로 sklearn 분류기를 학습한다.

학습 파이프라인:
  labels.csv → keypoint 특징 추출 → RandomForest 학습
  → models/hado_movement_clf.pkl 저장
  → 분류 성능 리포트 출력

실행:
    python -m tools.train_movement_model
    python -m tools.train_movement_model --eval   # 평가만 (재학습 없이)
    python -m tools.train_movement_model --min-samples 5  # 클래스당 최소 샘플 수
"""
from __future__ import annotations

import argparse
import csv
import json
import pickle
import sys
from pathlib import Path
from collections import Counter

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

LABEL_FILE  = PROJECT_ROOT / "data/reference_actions/labels.csv"
MODEL_FILE  = PROJECT_ROOT / "models/hado_movement_clf.pkl"
FEATURE_FILE = PROJECT_ROOT / "models/hado_movement_features.json"


# ── 특징 추출 ────────────────────────────────────────────────────
def kpts_to_feature(kpts_array: np.ndarray, bbox_h: float, bbox_w: float) -> np.ndarray | None:
    """(17,3) keypoints → 분류에 사용할 특징 벡터 반환.

    특징 구성 (16차원):
      [0]  floor_proximity    — 어깨-발목 수평도 (버피)
      [1]  bbox_horizontal    — bbox 가로세로비 (버피 플랭크)
      [2]  stance_width       — 발목 간격 / 어깨폭 (넓은 자세)
      [3]  ankle_asym         — 발목 높이 차이 (리듬박스)
      [4]  knee_raise         — 무릎이 엉덩이 위로 올라감 (리듬박스)
      [5]  crouch_depth       — 엉덩이 낮이 (스쿼트/런지)
      [6]  knee_asym          — 두 무릎 높이 차이 (런지/슬라이드)
      [7]  foot_fore_aft      — 발목 전후 비대칭 (런지)
      [8]  knee_bend_avg      — 평균 무릎 굽힘각
      [9]  knee_bend_diff     — 두 무릎 굽힘각 차이
      [10] lateral_lean       — 상체 좌우 기울기
      [11] hand_low           — 손목 아래 위치 (런닝슬라이드)
      [12] nose_level         — 코 높이 (누운 자세)
      [13] shoulder_asym_y    — 두 어깨 높이 차이
      [14] hip_asym_y         — 두 엉덩이 높이 차이
      [15] body_lean_forward  — 상체 앞쪽 기울기
    """
    from src.detector import Detection
    from src.hado_movement import _extract_features

    # 더미 Detection 객체 생성
    det = Detection(
        x1=0, y1=0,
        x2=float(bbox_w), y2=float(bbox_h),
        confidence=0.9,
        keypoints=kpts_array.astype(np.float32),
    )
    f = _extract_features(det)
    if f is None:
        return None

    # 추가 특징: 두 어깨 높이 차이, 두 엉덩이 높이 차이, 상체 앞기울기
    def _kp(i):
        if kpts_array[i, 2] >= 0.25:
            return float(kpts_array[i, 0]), float(kpts_array[i, 1])
        return None

    ls, rs = _kp(5), _kp(6)
    lh, rh = _kp(11), _kp(12)
    nose = _kp(0)
    bh = max(1.0, bbox_h)

    sh_asym = abs(ls[1]-rs[1])/bh if (ls and rs) else 0.0
    hi_asym = abs(lh[1]-rh[1])/bh if (lh and rh) else 0.0

    body_lean = 0.0
    if ls and rs and lh and rh and nose:
        sh_cx = (ls[0]+rs[0])/2
        hi_cx = (lh[0]+rh[0])/2
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
        f["knee_bend_avg"] / 180.0,   # 0~1 정규화
        f["knee_bend_diff"] / 180.0,
        f["lateral_lean"],
        f["hand_low"],
        f["nose_level"],
        sh_asym,
        hi_asym,
        body_lean,
    ], dtype=np.float32)

    return feat


FEATURE_NAMES = [
    "floor_proximity", "bbox_horizontal", "stance_width", "ankle_asym",
    "knee_raise", "crouch_depth", "knee_asym", "foot_fore_aft",
    "knee_bend_avg_norm", "knee_bend_diff_norm", "lateral_lean", "hand_low",
    "nose_level", "shoulder_asym_y", "hip_asym_y", "body_lean_forward",
]


# ── 학습 ─────────────────────────────────────────────────────────
def load_dataset(label_file: Path, min_samples: int = 3):
    """labels.csv → (X, y, class_names)."""
    if not label_file.exists():
        print(f"[Train] 라벨 파일 없음: {label_file}")
        print("       먼저 python -m tools.label_movements 를 실행하세요.")
        sys.exit(1)

    rows = []
    with open(label_file, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    # 현재 유효한 동작 라벨만 학습 (구버전 라벨 자동 제외)
    from tools.label_movements import MOVEMENT_LABELS
    valid_labels = {k for k in MOVEMENT_LABELS if k != "skip"}

    X, y = [], []
    skipped = 0
    for row in rows:
        label = row["label"]
        if label == "skip" or label not in valid_labels or not row.get("keypoints"):
            skipped += 1
            continue
        try:
            kpts_list = json.loads(row["keypoints"])
            kpts = np.array(kpts_list, dtype=np.float32)  # (17,3)
            if kpts.shape != (17, 3):
                skipped += 1
                continue
        except Exception:
            skipped += 1
            continue

        feat = kpts_to_feature(kpts, bbox_h=480, bbox_w=640)
        if feat is None:
            skipped += 1
            continue

        X.append(feat)
        y.append(label)

    print(f"[Train] 로드: {len(X)}개 (건너뜀: {skipped}개)")

    # 클래스별 샘플 수 확인
    counts = Counter(y)
    print("\n클래스별 샘플 수:")
    valid_classes = set()
    for label, cnt in sorted(counts.items()):
        from tools.label_movements import MOVEMENT_LABELS
        ko = MOVEMENT_LABELS.get(label, {}).get("ko", label)
        status = "✓" if cnt >= min_samples else f"✗ (최소 {min_samples}개 필요)"
        print(f"  {ko:16} ({label}): {cnt}개 {status}")
        if cnt >= min_samples:
            valid_classes.add(label)

    # 최소 샘플 미달 클래스 제외
    filtered = [(x, lbl) for x, lbl in zip(X, y) if lbl in valid_classes]
    if not filtered:
        print("\n[Train] 학습 가능한 클래스 없음. 라벨링을 더 진행하세요.")
        sys.exit(1)

    X_f, y_f = zip(*filtered)
    return np.array(X_f), list(y_f), sorted(valid_classes)


def train(args) -> None:
    try:
        from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
        from sklearn.preprocessing import LabelEncoder
        from sklearn.model_selection import cross_val_score, StratifiedKFold
        from sklearn.metrics import classification_report, confusion_matrix
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("[Train] sklearn 필요: pip install scikit-learn")
        sys.exit(1)

    print("=" * 60)
    print("HADO 기본동작 분류기 학습")
    print("=" * 60)

    X, y, class_names = load_dataset(LABEL_FILE, min_samples=args.min_samples)
    print(f"\n학습 데이터: {len(X)}개 샘플, {len(class_names)}개 클래스\n")

    le = LabelEncoder()
    le.fit(class_names)
    y_enc = le.transform(y)

    # 모델 파이프라인
    clf = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            min_samples_leaf=2,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    # Cross-validation
    if len(X) >= 10:
        cv_k = min(5, min(Counter(y).values()))
        if cv_k >= 2:
            cv = StratifiedKFold(n_splits=cv_k, shuffle=True, random_state=42)
            scores = cross_val_score(clf, X, y_enc, cv=cv, scoring="accuracy")
            print(f"교차검증 정확도: {scores.mean():.1%} (±{scores.std():.1%})")

    # 전체 데이터로 최종 학습
    clf.fit(X, y_enc)

    # 학습 데이터 성능
    y_pred = clf.predict(X)
    print("\n[학습 데이터 성능]")
    print(classification_report(
        y_enc, y_pred,
        target_names=[le.inverse_transform([i])[0] for i in range(len(class_names))],
        zero_division=0,
    ))

    # 특징 중요도
    rf_model = clf.named_steps["rf"]
    importances = rf_model.feature_importances_
    print("\n[특징 중요도 Top 8]")
    top_idx = np.argsort(importances)[::-1][:8]
    for i in top_idx:
        print(f"  {FEATURE_NAMES[i]:25} {importances[i]:.3f}")

    # 저장
    MODEL_FILE.parent.mkdir(parents=True, exist_ok=True)
    model_data = {
        "pipeline":    clf,
        "label_encoder": le,
        "class_names": class_names,
        "feature_names": FEATURE_NAMES,
        "n_samples":   len(X),
    }
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_data, f)

    # 특징 정보 저장 (참고용)
    with open(FEATURE_FILE, "w", encoding="utf-8") as f:
        json.dump({
            "feature_names": FEATURE_NAMES,
            "class_names": class_names,
            "n_samples": len(X),
        }, f, ensure_ascii=False, indent=2)

    print(f"\n모델 저장: {MODEL_FILE}")
    print(f"다음 단계: movement_demo.py 실행 시 자동으로 ML 모델 적용됨")
    print("실행: ./run.sh movement  또는  python -m src.movement_demo")


def evaluate(args) -> None:
    """저장된 모델로 전체 라벨 데이터 평가."""
    if not MODEL_FILE.exists():
        print("[Eval] 모델 없음. 먼저 학습: python -m tools.train_movement_model")
        sys.exit(1)

    try:
        from sklearn.metrics import classification_report
    except ImportError:
        print("[Eval] sklearn 필요: pip install scikit-learn")
        sys.exit(1)

    with open(MODEL_FILE, "rb") as f:
        data = pickle.load(f)

    clf = data["pipeline"]
    le  = data["label_encoder"]

    X, y, class_names = load_dataset(LABEL_FILE, min_samples=1)
    y_enc = le.transform([lbl for lbl in y if lbl in le.classes_])
    X_valid = np.array([x for x, lbl in zip(X, y) if lbl in le.classes_])

    y_pred = clf.predict(X_valid)
    print("\n[모델 평가]")
    print(classification_report(
        y_enc, y_pred,
        target_names=le.classes_,
        zero_division=0,
    ))


def main() -> None:
    parser = argparse.ArgumentParser(description="HADO 기본동작 분류기 학습")
    parser.add_argument("--eval",        action="store_true", help="평가만 실행")
    parser.add_argument("--min-samples", type=int, default=3,
                        help="학습에 포함할 클래스 최소 샘플 수 (기본: 3)")
    args = parser.parse_args()

    if args.eval:
        evaluate(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
