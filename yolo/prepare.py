#!/usr/bin/env python3
"""raw 이미지+라벨을 train/val 로 나누고 data.yaml 을 쓴다.

  python yolo/prepare.py
  python yolo/prepare.py --seg   # labels_seg 폴리곤
"""

from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yolo.config import (
    CUSTOM_ROOT,
    DATA_YAML,
    RAW_IMAGES,
    RAW_LABELS,
    RAW_LABELS_SEG,
    SPLIT_SEED,
    VAL_RATIO,
    add_class_argument,
    class_from_args,
)


def _images() -> list[Path]:
    return sorted(list(RAW_IMAGES.glob("*.jpg")) + list(RAW_IMAGES.glob("*.png")))


def _write_yaml(class_name: str) -> None:
    rel = CUSTOM_ROOT.relative_to(DATA_YAML.parent).as_posix()
    body = (
        f"path: {rel}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: {class_name}\n"
    )
    DATA_YAML.write_text(body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="split raw labels into train/val")
    parser.add_argument("--seg", action="store_true", help="labels_seg 폴리곤을 씀")
    add_class_argument(parser)
    args = parser.parse_args()
    class_name = class_from_args(args)
    label_root = RAW_LABELS_SEG if args.seg else RAW_LABELS
    hint = "python yolo/auto_seg.py" if args.seg else "python yolo/label.py"

    images = _images()
    if not images:
        raise SystemExit(f"이미지가 없습니다: {RAW_IMAGES}")
    pairs: list[tuple[Path, Path]] = []
    missing = 0
    for img in images:
        lab = label_root / f"{img.stem}.txt"
        if not lab.exists():
            missing += 1
            continue
        pairs.append((img, lab))
    if missing:
        print(f"라벨 없는 이미지 {missing}장 — 건너뜀. {hint} 로 마저 표시.")
    if len(pairs) < 10:
        raise SystemExit(f"라벨 쌍이 {len(pairs)}개뿐입니다. 클래스당 80장 이상을 권장합니다.")
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(pairs)
    n_val = max(1, int(round(len(pairs) * VAL_RATIO)))
    if n_val >= len(pairs):
        n_val = 1
    val = pairs[:n_val]
    train = pairs[n_val:]
    for split, items in (("train", train), ("val", val)):
        img_dir = CUSTOM_ROOT / "images" / split
        lab_dir = CUSTOM_ROOT / "labels" / split
        if img_dir.exists():
            shutil.rmtree(img_dir)
        if lab_dir.exists():
            shutil.rmtree(lab_dir)
        img_dir.mkdir(parents=True, exist_ok=True)
        lab_dir.mkdir(parents=True, exist_ok=True)
        for img, lab in items:
            shutil.copy2(img, img_dir / img.name)
            shutil.copy2(lab, lab_dir / lab.name)
    _write_yaml(class_name)
    print(f"train={len(train)}  val={len(val)}  class={class_name}  seg={args.seg}")
    print(f"yaml={DATA_YAML}")
    print("다음: python yolo/train.py --seg" if args.seg else "다음: python yolo/train.py")


if __name__ == "__main__":
    main()
