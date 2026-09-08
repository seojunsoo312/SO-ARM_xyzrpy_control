#!/usr/bin/env python3
"""SAM 마스크를 눈으로 확인하고 틀린 장만 버린다.

폴리곤을 새로 그리지는 않는다. 지우면 빈 라벨이 된다.
그 장은 auto_seg --force 로 다시 뽑거나, 학습에서 빼진다.

  python yolo/review_seg.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yolo.boxes import load_yolo_seg_txt, save_yolo_seg_txt
from yolo.config import RAW_IMAGES, RAW_LABELS_SEG, add_class_argument, class_from_args, quiet_gtk

quiet_gtk()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

WIN = "YOLO review seg"


def _image_paths() -> list[Path]:
    return sorted(list(RAW_IMAGES.glob("*.jpg")) + list(RAW_IMAGES.glob("*.png")))


def _seg_path(image: Path) -> Path:
    return RAW_LABELS_SEG / f"{image.stem}.txt"


def _draw(bgr: np.ndarray, polygons, index: int, total: int, class_name: str) -> np.ndarray:
    h, w = bgr.shape[:2]
    overlay = bgr.copy()
    out = bgr.copy()
    for poly in polygons:
        pts = np.array([(int(x * w), int(y * h)) for x, y in poly], dtype=np.int32)
        if len(pts) < 3:
            continue
        cv2.fillPoly(overlay, [pts], (0, 180, 0))
        cv2.polylines(out, [pts], True, (0, 255, 0), 2)
    vis = cv2.addWeighted(overlay, 0.35, out, 0.65, 0)
    lines = [
        f"{index + 1}/{total}  {class_name}  masks={len(polygons)}",
        "n=next  p=prev  d=undo  r=clear  q=quit",
    ]
    y = 22
    for line in lines:
        cv2.putText(
            vis, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
        )
        y += 20
    return vis


def main() -> None:
    parser = argparse.ArgumentParser(description="review SAM masks")
    add_class_argument(parser)
    args = parser.parse_args()
    class_name = class_from_args(args)

    images = _image_paths()
    if not images:
        raise SystemExit(f"이미지가 없습니다: {RAW_IMAGES}")
    labeled = [p for p in images if _seg_path(p).exists()]
    if not labeled:
        raise SystemExit("마스크 라벨이 없습니다. 먼저 python yolo/auto_seg.py")
    idx = 0
    polygons = load_yolo_seg_txt(_seg_path(labeled[idx]))
    bgr = cv2.imread(str(labeled[idx]))
    if bgr is None:
        raise SystemExit(f"읽기 실패: {labeled[idx]}")
    cv2.namedWindow(WIN)
    print(f"{len(labeled)}장. 틀린 마스크는 d/r 로 지운다.")
    try:
        while True:
            cv2.imshow(WIN, _draw(bgr, polygons, idx, len(labeled), class_name))
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            if key == ord("q"):
                save_yolo_seg_txt(_seg_path(labeled[idx]), polygons)
                break
            if key == ord("d") and polygons:
                polygons.pop()
            if key == ord("r"):
                polygons.clear()
            if key in (ord("n"), ord("p")):
                save_yolo_seg_txt(_seg_path(labeled[idx]), polygons)
                idx = idx + 1 if key == ord("n") else idx - 1
                idx = max(0, min(idx, len(labeled) - 1))
                bgr = cv2.imread(str(labeled[idx]))
                if bgr is None:
                    print(f"읽기 실패: {labeled[idx]}")
                    continue
                polygons = load_yolo_seg_txt(_seg_path(labeled[idx]))
                print(f"{labeled[idx].name}  masks={len(polygons)}")
    finally:
        cv2.destroyAllWindows()
    print("다음: python yolo/prepare.py --seg")


if __name__ == "__main__":
    main()
