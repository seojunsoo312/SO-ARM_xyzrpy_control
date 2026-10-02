#!/usr/bin/env python3
"""raw 이미지+라벨을 train/val 목록으로 나눈다. 파일은 복사하지 않는다.

  python yolo/train/prepare.py
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yolo.config import (
    DATA_YAML,
    RAW_IMAGES,
    RAW_LABELS,
    SPLIT_SEED,
    SPLITS_DIR,
    TRAIN_TXT,
    VAL_RATIO,
    VAL_TXT,
    pose_names_yaml,
)


def _images() -> list[Path]:
    return sorted(list(RAW_IMAGES.glob("*.jpg")) + list(RAW_IMAGES.glob("*.png")))


def _write_list(dest: Path, images: list[Path]) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(
        "".join(f"{img.absolute().as_posix()}\n" for img in images),
        encoding="utf-8",
    )


def _write_yaml() -> None:
    # Ultralytics 8.4 는 상대 path 를 yaml 위치가 아니라 settings datasets_dir 에 붙인다.
    root = DATA_YAML.parent.resolve()
    body = (
        f"path: {root.as_posix()}\n"
        f"train: {TRAIN_TXT.resolve().relative_to(root).as_posix()}\n"
        f"val: {VAL_TXT.resolve().relative_to(root).as_posix()}\n"
        f"{pose_names_yaml()}"
    )
    DATA_YAML.write_text(body, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="split raw labels into train/val lists")
    parser.parse_args()

    images = _images()
    if not images:
        raise SystemExit(f"이미지가 없습니다: {RAW_IMAGES}")
    pairs: list[tuple[Path, Path]] = []
    missing = 0
    for img in images:
        lab = RAW_LABELS / f"{img.stem}.txt"
        if not lab.exists():
            missing += 1
            continue
        pairs.append((img, lab))
    if missing:
        print(f"라벨 없는 이미지 {missing}장 — 건너뜀. python yolo/train/label.py 로 마저 표시.")
    if len(pairs) < 10:
        raise SystemExit(f"라벨 쌍이 {len(pairs)}개뿐입니다. 클래스당 80장 이상을 권장합니다.")
    rng = random.Random(SPLIT_SEED)
    rng.shuffle(pairs)
    n_val = max(1, int(round(len(pairs) * VAL_RATIO)))
    if n_val >= len(pairs):
        n_val = 1
    val = pairs[:n_val]
    train = pairs[n_val:]

    train_imgs = [img for img, _ in train]
    val_imgs = [img for img, _ in val]

    SPLITS_DIR.mkdir(parents=True, exist_ok=True)
    _write_list(TRAIN_TXT, train_imgs)
    _write_list(VAL_TXT, val_imgs)
    _write_yaml()
    print(f"train={len(train)}  val={len(val)}  classes=세우기,눕히기,비스듬히")
    print(f"목록 {TRAIN_TXT}  {VAL_TXT}  (이미지 복사 없음)")
    print(f"yaml={DATA_YAML}")
    print("다음: python yolo/train/train.py")


if __name__ == "__main__":
    main()
