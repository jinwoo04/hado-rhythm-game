"""YOLO classification 학습용 데이터셋 준비 도구.

labels.csv + raw_jpg → YOLO cls 폴더 구조로 변환
  dataset/
    train/
      squat/  lunge/  slide/  weaving/  next_direction/  burpee/  ready/
    val/
      squat/  ...

실행:
    python -m tools.prepare_yolo_dataset
    python -m tools.prepare_yolo_dataset --val-ratio 0.2
"""
from __future__ import annotations

import argparse
import csv
import random
import shutil
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
LABEL_FILE   = PROJECT_ROOT / "data/reference_actions/labels.csv"
RAW_JPG_DIR  = PROJECT_ROOT / "data/reference_actions/raw_jpg"
DATASET_DIR  = PROJECT_ROOT / "data/yolo_cls_dataset"

VALID_LABELS = {
    "ready", "squat", "lunge", "slide",
    "weaving", "next_direction", "burpee",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-ratio", type=float, default=0.2,
                        help="검증 세트 비율 (기본 0.2 = 20%%)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = random.Random(args.seed)

    # CSV 읽기
    rows: list[dict] = []
    with open(LABEL_FILE, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["label"] in VALID_LABELS:
                jpg = RAW_JPG_DIR / row["filename"]
                # PNG → JPG 변환된 파일명도 확인
                jpg_alt = RAW_JPG_DIR / (
                    Path(row["filename"]).stem.replace(" ", "_").replace(":", "-") + ".jpg"
                )
                if jpg.exists():
                    rows.append({"label": row["label"], "path": jpg})
                elif jpg_alt.exists():
                    rows.append({"label": row["label"], "path": jpg_alt})

    print(f"사용 가능한 이미지: {len(rows)}장")

    # 클래스별 분리 후 train/val 분할
    from collections import defaultdict
    by_class: dict[str, list] = defaultdict(list)
    for r in rows:
        by_class[r["label"]].append(r["path"])

    # 기존 데이터셋 폴더 초기화
    if DATASET_DIR.exists():
        shutil.rmtree(DATASET_DIR)

    total_train, total_val = 0, 0
    print(f"\n{'클래스':16} {'전체':>5} {'train':>6} {'val':>5}")
    print("-" * 36)

    for label, paths in sorted(by_class.items()):
        rng.shuffle(paths)
        n_val   = max(1, int(len(paths) * args.val_ratio))
        val_paths   = paths[:n_val]
        train_paths = paths[n_val:]

        for split, split_paths in [("train", train_paths), ("val", val_paths)]:
            dst = DATASET_DIR / split / label
            dst.mkdir(parents=True, exist_ok=True)
            for src in split_paths:
                shutil.copy2(src, dst / src.name)

        print(f"  {label:16} {len(paths):>4}장  {len(train_paths):>5}장  {len(val_paths):>4}장")
        total_train += len(train_paths)
        total_val   += len(val_paths)

    print("-" * 36)
    print(f"  {'합계':16} {total_train+total_val:>4}장  {total_train:>5}장  {total_val:>4}장")
    print(f"\n데이터셋 저장: {DATASET_DIR}")
    print()
    print("=" * 50)
    print("다음 단계 — Google Colab에서 학습:")
    print("=" * 50)
    print("""
1. data/yolo_cls_dataset/ 폴더를 구글 드라이브에 업로드

2. Colab 셀에 아래 코드 실행:

!pip install ultralytics -q

from google.colab import drive
drive.mount('/content/drive')

from ultralytics import YOLO
model = YOLO('yolov8n-cls.pt')
model.train(
    data='/content/drive/MyDrive/yolo_cls_dataset',
    epochs=100,
    imgsz=224,
    batch=32,
    patience=20,
    name='hado_movement',
)

3. runs/classify/hado_movement/weights/best.pt 다운로드

4. 프로젝트 models/ 폴더에 hado_movement_cls.pt 로 저장
""")


if __name__ == "__main__":
    main()
