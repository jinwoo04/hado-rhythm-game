"""HEIC / PNG → JPG 변환 후 raw_jpg 폴더에 추가.

"하도 기본 동작" 폴더에 새 사진이 추가될 때마다 실행.
이미 변환된 파일은 건너뜀 (재실행 안전).

실행:
    python -m tools.import_raw_images
    python -m tools.import_raw_images --source "/Users/jinu/Desktop/하도 기본 동작"
"""
from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SOURCE = Path("/Users/jinu/Desktop/하도 기본 동작")
RAW_JPG_DIR = PROJECT_ROOT / "data/reference_actions/raw_jpg"


def _safe_stem(name: str) -> str:
    """파일명에서 공백과 특수문자를 안전하게 변환."""
    return name.replace(" ", "_").replace(":", "-")


def _to_jpg(src: Path, dst_dir: Path) -> Path | None:
    """단일 파일 변환. 성공 시 저장 경로 반환, 이미 존재하면 None."""
    stem = _safe_stem(src.stem)
    dst = dst_dir / f"{stem}.jpg"
    if dst.exists():
        return None  # 이미 변환됨

    try:
        if src.suffix.upper() in (".HEIC", ".HEIF"):
            import pillow_heif
            pillow_heif.register_heif_opener()

        img = Image.open(src)
        if img.mode != "RGB":
            img = img.convert("RGB")
        img.save(dst, "JPEG", quality=92)
        return dst
    except Exception as e:
        print(f"  ✗ 변환 실패 {src.name}: {e}")
        return None


def main() -> None:
    parser = argparse.ArgumentParser(description="이미지 변환 후 raw_jpg에 추가")
    parser.add_argument("--source", default=str(DEFAULT_SOURCE),
                        help=f"원본 폴더 (기본: {DEFAULT_SOURCE})")
    args = parser.parse_args()

    src_dir = Path(args.source)
    if not src_dir.exists():
        print(f"[Import] 폴더 없음: {src_dir}")
        return

    RAW_JPG_DIR.mkdir(parents=True, exist_ok=True)

    # 변환 대상: HEIC, HEIF, PNG
    exts = {".heic", ".heif", ".png", ".PNG", ".HEIC", ".HEIF"}
    sources = sorted(p for p in src_dir.iterdir() if p.suffix.upper() in {e.upper() for e in exts})

    if not sources:
        print(f"[Import] 변환할 파일 없음: {src_dir}")
        return

    print(f"[Import] 원본 {len(sources)}개 발견 → {RAW_JPG_DIR}")

    added, skipped, failed = 0, 0, 0
    for src in sources:
        result = _to_jpg(src, RAW_JPG_DIR)
        if result is None:
            skipped += 1
        else:
            added += 1
            if added <= 5 or added % 50 == 0:
                print(f"  변환: {src.name} → {result.name}")

    print(f"\n완료: 추가 {added}장 / 건너뜀(기존) {skipped}장 / 실패 {failed}장")
    print(f"총 raw_jpg: {len(list(RAW_JPG_DIR.glob('*.jpg')))}장")

    if added > 0:
        print()
        print("다음 단계:")
        print("  1) 라벨링:    python -m tools.label_movements --resume")
        print("  2) 데이터셋:  python -m tools.prepare_yolo_dataset")
        print("  3) Colab에 data/yolo_cls_dataset/ 업로드 후 학습")


if __name__ == "__main__":
    main()
