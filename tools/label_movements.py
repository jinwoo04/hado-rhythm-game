"""HADO 기본동작 라벨링 도구.

207장의 참조 사진에 YOLOv8n-pose로 keypoint를 추출하고
사용자가 직접 정확한 동작 레이블을 부여한다.
결과는 data/reference_actions/labels.csv에 저장된다.

하도리듬 동작 체계 (2025 하도리듬 자료정리.pdf 기준):
  입문(기본) 동작: 스쿼트 / 사이드스텝 / 런지 / 슬라이드
  기술 동작:     반단 / 버트 / 크로스 / 복닥
  응용/복합 동작: 하프크로스 / 하도리듬박스
  준비 자세:      ready

실행:
    python -m tools.label_movements
    python -m tools.label_movements --resume   # 이전에 중단한 지점부터
    python -m tools.label_movements --source data/reference_actions/raw_jpg
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── HADO 7가지 동작 레이블 (2026-06-18 확정) ─────────────────────
MOVEMENT_LABELS: dict[str, dict] = {
    "ready":          {"ko": "준비 동작",      "key": "0", "level": "기본"},
    "squat":          {"ko": "스쿼트",         "key": "1", "level": "입문"},
    "lunge":          {"ko": "런지",           "key": "2", "level": "입문"},
    "slide":          {"ko": "슬라이드",       "key": "3", "level": "입문"},
    "weaving":        {"ko": "위빙",           "key": "4", "level": "기술"},
    "next_direction": {"ko": "넥디렉션",       "key": "5", "level": "기술"},
    "burpee":         {"ko": "버피테스트",     "key": "6", "level": "기술"},
    "skip":           {"ko": "건너뜀 (불명확)","key": "s", "level": "-"},
}

KEY_TO_LABEL = {v["key"]: k for k, v in MOVEMENT_LABELS.items()}

LABEL_COLORS: dict[str, tuple[int, int, int]] = {
    "ready":          (160, 160, 160),
    "squat":          (0,   200, 255),
    "lunge":          (40,  180, 255),
    "slide":          (0,   255, 160),
    "weaving":        (255, 220, 0),
    "next_direction": (255, 120, 50),
    "burpee":         (200, 60,  255),
}


def _put_kr(img: np.ndarray, text: str, xy: tuple[int, int],
            size: int, color: tuple[int, int, int]) -> None:
    try:
        from src.annotate import put_text_kr
        put_text_kr(img, text, xy, size, color)
    except Exception:
        cv2.putText(img, text, xy, cv2.FONT_HERSHEY_SIMPLEX,
                    size / 30, color, 1, cv2.LINE_AA)


def _draw_skeleton(img: np.ndarray, kpts: np.ndarray) -> None:
    PAIRS = [(5,6),(5,7),(7,9),(6,8),(8,10),(5,11),(6,12),(11,12),
             (11,13),(13,15),(12,14),(14,16),(0,5),(0,6)]
    for a, b in PAIRS:
        if kpts[a,2] > 0.25 and kpts[b,2] > 0.25:
            pa = (int(kpts[a,0]), int(kpts[a,1]))
            pb = (int(kpts[b,0]), int(kpts[b,1]))
            cv2.line(img, pa, pb, (0, 220, 180), 2, cv2.LINE_AA)
    for i in range(17):
        if kpts[i,2] > 0.25:
            cv2.circle(img, (int(kpts[i,0]), int(kpts[i,1])), 4, (255,255,255), -1)
            cv2.putText(img, str(i), (int(kpts[i,0])+4, int(kpts[i,1])-3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255,255,0), 1)


def _draw_legend(img: np.ndarray, auto_pred: str | None = None) -> None:
    h, w = img.shape[:2]
    panel_w = 230
    overlay = img.copy()
    cv2.rectangle(overlay, (w - panel_w, 0), (w, h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, img, 0.25, 0, img)

    _put_kr(img, "HADO 동작 라벨링", (w - panel_w + 8, 10), 14, (200, 200, 200))

    if auto_pred:
        color = LABEL_COLORS.get(auto_pred, (180, 180, 180))
        ko = MOVEMENT_LABELS.get(auto_pred, {}).get("ko", auto_pred)
        _put_kr(img, f"AI 예측: [{ko}]", (w - panel_w + 8, 34), 13, color)
        _put_kr(img, "SPACE = 예측 수락", (w - panel_w + 8, 54), 12, (120, 200, 120))

    y = 80
    for label, info in MOVEMENT_LABELS.items():
        if label == "skip":
            _put_kr(img, f"[S] {info['ko']}", (w - panel_w + 8, y), 12, (100, 100, 100))
        else:
            color = LABEL_COLORS.get(label, (160, 160, 160))
            lv = info["level"]
            _put_kr(img, f"[{info['key']}] {info['ko']} ({lv})",
                    (w - panel_w + 8, y), 12, color)
        y += 22

    _put_kr(img, "B = 이전 사진", (w - panel_w + 8, y + 8), 12, (120, 120, 120))
    _put_kr(img, "Q = 저장 후 종료", (w - panel_w + 8, y + 26), 12, (120, 120, 120))


def run(args) -> None:
    from src.detector import PersonDetector
    from src.hado_movement import classify_hado_movement, MOVEMENT_KO

    # 모델 로드
    onnx = PROJECT_ROOT / "yolov8n-pose.onnx"
    pt   = PROJECT_ROOT / "yolov8n-pose.pt"
    model_path = str(onnx) if onnx.exists() else str(pt)
    detector = PersonDetector(model_path=model_path, imgsz=320, conf_threshold=0.35)

    # 사진 목록
    src_dir = Path(args.source)
    photos  = sorted(src_dir.glob("*.jpg")) + sorted(src_dir.glob("*.jpeg"))
    if not photos:
        print(f"[Label] 사진 없음: {src_dir}")
        return
    print(f"[Label] {len(photos)}장 발견: {src_dir}")

    # 기존 라벨 로드
    label_file = PROJECT_ROOT / "data/reference_actions/labels.csv"
    label_file.parent.mkdir(parents=True, exist_ok=True)
    labels: dict[str, dict] = {}  # filename → {label, kpts_json}

    if label_file.exists():
        with open(label_file, newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                labels[row["filename"]] = row
        print(f"[Label] 기존 라벨 {len(labels)}개 로드")

    # 시작 인덱스
    idx = 0
    if args.resume:
        labeled_names = set(labels.keys())
        for i, p in enumerate(photos):
            if p.name not in labeled_names:
                idx = i
                break
        print(f"[Label] 재개: {idx}번째 사진부터")

    cv2.namedWindow("HADO Labeling", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("HADO Labeling", 900, 540)

    while 0 <= idx < len(photos):
        path = photos[idx]
        img  = cv2.imread(str(path))
        if img is None:
            idx += 1
            continue

        # Keypoint 감지
        dets = detector.detect(img)
        target = max(dets, key=lambda d: d.area) if dets else None
        kpts   = target.keypoints if target else None
        auto_pred = None

        if kpts is not None:
            _draw_skeleton(img, kpts)
            res = classify_hado_movement(target)
            if res:
                auto_pred = res.movement

        # 현재 라벨 표시
        existing = labels.get(path.name, {}).get("label", "")
        ex_color = LABEL_COLORS.get(existing, (200, 200, 200))
        if existing:
            ko = MOVEMENT_LABELS.get(existing, {}).get("ko", existing)
            _put_kr(img, f"현재: {ko}", (10, 10), 18, ex_color)

        # 진행률 + 파일명 (ASCII — cv2 그대로 사용)
        cv2.putText(img, f"{idx+1}/{len(photos)}  {path.name}",
                    (10, img.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (150, 150, 150), 1)

        _draw_legend(img, auto_pred)
        cv2.imshow("HADO Labeling", img)

        key = chr(cv2.waitKey(0) & 0xFF).lower()

        if key == 'q':
            break
        elif key == 'b':
            idx = max(0, idx - 1)
            continue
        elif key == ' ' and auto_pred:
            # AI 예측 수락
            chosen = auto_pred
        elif key in KEY_TO_LABEL:
            chosen = KEY_TO_LABEL[key]
        else:
            # 미인식 키 → 다음
            idx += 1
            continue

        if chosen == "skip":
            labels.pop(path.name, None)
            idx += 1
            continue

        # 라벨 저장
        kpts_json = json.dumps(kpts.tolist()) if kpts is not None else ""
        labels[path.name] = {
            "filename": path.name,
            "label":    chosen,
            "keypoints": kpts_json,
        }

        # CSV 즉시 저장 (중간 종료 대비)
        _save_csv(label_file, labels)
        print(f"[Label] {path.name} → {MOVEMENT_LABELS[chosen]['ko']} ({chosen})")
        idx += 1

    cv2.destroyAllWindows()
    _save_csv(label_file, labels)

    # 라벨 통계
    from collections import Counter
    counts = Counter(v["label"] for v in labels.values())
    print(f"\n라벨링 완료: {len(labels)}/{len(photos)}장")
    print("─" * 40)
    for label, cnt in sorted(counts.items(), key=lambda x: -x[1]):
        ko = MOVEMENT_LABELS.get(label, {}).get("ko", label)
        print(f"  {ko:16} ({label}): {cnt}장")
    print(f"\n저장: {label_file}")
    print("다음 단계: python -m tools.train_movement_model")


def _save_csv(path: Path, labels: dict) -> None:
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["filename", "label", "keypoints"])
        writer.writeheader()
        writer.writerows(labels.values())


def main() -> None:
    parser = argparse.ArgumentParser(description="HADO 기본동작 라벨링 도구")
    parser.add_argument("--source", default="data/reference_actions/raw_jpg",
                        help="사진 폴더 경로")
    parser.add_argument("--resume", action="store_true",
                        help="라벨이 없는 사진부터 재개")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
