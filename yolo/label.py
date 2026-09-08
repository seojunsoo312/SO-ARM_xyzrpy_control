#!/usr/bin/env python3
"""드래그로 박스를 그려 YOLO txt 라벨을 만든다. 클래스는 하나(인덱스 0).

빈 장면도 저장한다 (네거티브). 물건이 없으면 박스 없이 n.

  python yolo/label.py
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yolo.boxes import load_yolo_txt, save_yolo_txt
from yolo.config import RAW_IMAGES, RAW_LABELS, add_class_argument, class_from_args, quiet_gtk

quiet_gtk()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

WIN = "YOLO label"
MIN_BOX = 6


def _image_paths() -> list[Path]:
    return sorted(list(RAW_IMAGES.glob("*.jpg")) + list(RAW_IMAGES.glob("*.png")))


def _label_path(image: Path) -> Path:
    return RAW_LABELS / f"{image.stem}.txt"


@dataclass
class Session:
    boxes: list[tuple[int, int, int, int]] = field(default_factory=list)
    drag: tuple[int, int] | None = None
    cur: tuple[int, int] | None = None
    w: int = 0
    h: int = 0

    def on_mouse(self, event, x, y, _flags, _param) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            self.drag = (int(x), int(y))
            self.cur = self.drag
        elif event == cv2.EVENT_MOUSEMOVE and self.drag is not None:
            self.cur = (int(x), int(y))
        elif event == cv2.EVENT_LBUTTONUP and self.drag is not None:
            self.cur = (int(x), int(y))
            self._finish_drag()
            self.drag = None
            self.cur = None

    def _finish_drag(self) -> None:
        if self.drag is None or self.cur is None:
            return
        x1, y1 = self.drag
        x2, y2 = self.cur
        xa, xb = sorted((x1, x2))
        ya, yb = sorted((y1, y2))
        xa = max(0, min(xa, self.w - 1))
        xb = max(0, min(xb, self.w - 1))
        ya = max(0, min(ya, self.h - 1))
        yb = max(0, min(yb, self.h - 1))
        if xb - xa < MIN_BOX or yb - ya < MIN_BOX:
            return
        self.boxes.append((xa, ya, xb, yb))

    def draw(self, bgr: np.ndarray, index: int, total: int, class_name: str) -> np.ndarray:
        vis = bgr.copy()
        for x1, y1, x2, y2 in self.boxes:
            cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                vis,
                class_name,
                (x1, max(16, y1 - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )
        if self.drag is not None and self.cur is not None:
            cv2.rectangle(vis, self.drag, self.cur, (0, 255, 255), 1)
        lines = [
            f"{index + 1}/{total}  {class_name}  boxes={len(self.boxes)}  {self.w}x{self.h}",
            "drag=box  n=next  p=prev  d=undo  r=clear  q=quit",
        ]
        y = 22
        for line in lines:
            cv2.putText(
                vis, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
            )
            y += 20
        return vis


def main() -> None:
    parser = argparse.ArgumentParser(description="draw YOLO boxes")
    add_class_argument(parser)
    args = parser.parse_args()
    class_name = class_from_args(args)

    images = _image_paths()
    if not images:
        raise SystemExit(f"이미지가 없습니다. 먼저 python yolo/capture.py  → {RAW_IMAGES}")
    RAW_LABELS.mkdir(parents=True, exist_ok=True)
    idx = 0
    bgr = cv2.imread(str(images[idx]))
    if bgr is None:
        raise SystemExit(f"읽기 실패: {images[idx]}")
    sess = Session()
    sess.h, sess.w = bgr.shape[:2]
    sess.boxes = load_yolo_txt(_label_path(images[idx]), sess.w, sess.h)
    cv2.namedWindow(WIN)
    cv2.setMouseCallback(WIN, sess.on_mouse)
    print(f"{len(images)}장. 드래그로 박스. 빈 장면은 박스 없이 n.")
    try:
        while True:
            vis = sess.draw(bgr, idx, len(images), class_name)
            cv2.imshow(WIN, vis)
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            if key == ord("q"):
                save_yolo_txt(_label_path(images[idx]), sess.boxes, sess.w, sess.h)
                break
            if key == ord("d"):
                if sess.boxes:
                    sess.boxes.pop()
            if key == ord("r"):
                sess.boxes.clear()
            if key in (ord("n"), ord("p")):
                save_yolo_txt(_label_path(images[idx]), sess.boxes, sess.w, sess.h)
                idx = idx + 1 if key == ord("n") else idx - 1
                idx = max(0, min(idx, len(images) - 1))
                bgr = cv2.imread(str(images[idx]))
                if bgr is None:
                    print(f"읽기 실패: {images[idx]}")
                    continue
                sess.h, sess.w = bgr.shape[:2]
                sess.boxes = load_yolo_txt(_label_path(images[idx]), sess.w, sess.h)
                sess.drag = None
                sess.cur = None
                print(f"{images[idx].name}  boxes={len(sess.boxes)}")
    finally:
        cv2.destroyAllWindows()
    labeled = len(list(RAW_LABELS.glob("*.txt")))
    print(f"라벨 {labeled}개. 다음: python yolo/auto_seg.py")


if __name__ == "__main__":
    main()
