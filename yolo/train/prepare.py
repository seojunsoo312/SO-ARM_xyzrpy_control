#!/usr/bin/env python3
"""raw 이미지+라벨을 train/val 목록으로 나눈다. 파일은 복사하지 않는다.

  python yolo/train/prepare.py
  python yolo/train/prepare.py --seg   # labels_seg 폴리곤
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yolo.config import (
    DATA_YAML,
    RAW_IMAGES,
    RAW_LABELS,
    RAW_LABELS_SEG,
    SEG_VIEW,
    SPLIT_SEED,
    SPLITS_DIR,
    TRAIN_TXT,
    VAL_RATIO,
    VAL_TXT,
    add_class_argument,
    class_from_args,
)


def _images() -> list[Path]:
    return sorted(list(RAW_IMAGES.glob("*.jpg")) + list(RAW_IMAGES.glob("*.png")))


def _write_list(dest: Path, images: list[Path]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        "".join(f"{img.absolute().as_posix()}\n" for img in images),
        encoding="utf-8",
    )


def _write_yaml(class_name: str, *, seg: bool) -> None:
    # Ultralytics 8.4 는 상대 path 를 yaml 위치가 아니라 settings datasets_dir 에 붙인다.
    root = DATA_YAML.parent.resolve()
    body = (
        f"path: {root.as_posix()}\n"
        f"train: {TRAIN_TXT.resolve().relative_to(root).as_posix()}\n"
        f"val: {VAL_TXT.resolve().relative_to(root).as_posix()}\n"
        f"nc: 1\n"
        f"names:\n"
        f"  0: {class_name}\n"
    )
    DATA_YAML.write_text(body, encoding="utf-8")


def _link(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    dest = target.resolve()
    if link.is_symlink() or link.exists():
        if link.is_symlink() and link.resolve() == dest:
            return
        link.unlink()
    link.symlink_to(os.path.relpath(dest, start=link.parent.resolve()))


def _ensure_seg_view() -> None:
    _link(SEG_VIEW / "images", RAW_IMAGES)
    _link(SEG_VIEW / "labels", RAW_LABELS_SEG)


def main() -> None:
    parser = argparse.ArgumentParser(description="split raw labels into train/val lists")
    parser.add_argument("--seg", action="store_true", help="labels_seg 폴리곤을 씀")
    add_class_argument(parser)
    args = parser.parse_args()
    class_name = class_from_args(args)
    label_root = RAW_LABELS_SEG if args.seg else RAW_LABELS
    hint = "python yolo/train/auto_seg.py" if args.seg else "python yolo/train/label.py"

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

    if args.seg:
        _ensure_seg_view()
        img_root = SEG_VIEW / "images"
        train_imgs = [img_root / img.name for img, _ in train]
        val_imgs = [img_root / img.name for img, _ in val]
    else:
        train_imgs = [img for img, _ in train]
        val_imgs = [img for img, _ in val]

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    _write_list(TRAIN_TXT, train_imgs)
    _write_list(VAL_TXT, val_imgs)
    _write_yaml(class_name, seg=args.seg)
    print(f"train={len(train)}  val={len(val)}  class={class_name}  seg={args.seg}")
    print(f"목록 {TRAIN_TXT}  {VAL_TXT}  (이미지 복사 없음)")
    print(f"yaml={DATA_YAML}")
    print("다음: python yolo/train/train.py --seg" if args.seg else "다음: python yolo/train/train.py")


if __name__ == "__main__":
    main()
