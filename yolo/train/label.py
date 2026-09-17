#!/usr/bin/env python3
"""네 점을 찍어 YOLO-OBB 라벨을 만든다. 직각을 강제하지 않는다.

빈 장면은 점 없이 n.

  python yolo/train/label.py

클릭 4번 = 꼭짓점. 찍은 네 점을 이은 사각형이 그대로 라벨이 된다.
우클릭 또는 d = 마지막 점/박스 취소.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yolo.config import RAW_IMAGES, RAW_LABELS, add_class_argument, class_from_args, quiet_gtk

quiet_gtk()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from yolo.train.boxes import Quad, load_yolo_txt, order_quad, save_yolo_txt  # noqa: E402

WIN = "YOLO OBB label"
MIN_SIDE = 6
N_CORNERS = 4


def _image_paths() -> list[Path]:
    return sorted(list(RAW_IMAGES.glob("*.jpg")) + list(RAW_IMAGES.glob("*.png")))


def _label_path(image: Path) -> Path:
    return RAW_LABELS / f"{image.stem}.txt"


def _quad_ok(quad: Quad) -> bool:
    pts = order_quad(quad)
    for i in range(4):
        if float(np.linalg.norm(pts[i] - pts[(i + 1) % 4])) < MIN_SIDE:
            return False
    return True


def _draw_quad(vis: np.ndarray, quad: Quad, color, thick: int) -> None:
    pts = order_quad(quad).astype(np.int32)
    cv2.polylines(vis, [pts], True, color, thick)
    for x, y in pts:
        cv2.circle(vis, (int(x), int(y)), 4, color, -1)


@dataclass
class Session:
    boxes: list[Quad] = field(default_factory=list)
    pending: list[tuple[int, int]] = field(default_factory=list)
    hover: tuple[int, int] | None = None
    w: int = 0
    h: int = 0

    def undo(self) -> None:
        if self.pending:
            self.pending.pop()
            return
        if self.boxes:
            self.boxes.pop()

    def clear(self) -> None:
        if self.pending:
            self.pending.clear()
            return
        self.boxes.clear()

    def on_mouse(self, event, x, y, _flags, _param) -> None:
        x, y = int(x), int(y)
        if event == cv2.EVENT_MOUSEMOVE:
            self.hover = (x, y)
        elif event == cv2.EVENT_LBUTTONDOWN:
            self.pending.append((x, y))
            if len(self.pending) >= N_CORNERS:
                quad = order_quad(np.array(self.pending, dtype=np.float32))
                self.pending.clear()
                if not _quad_ok(quad):
                    print("점이 너무 가깝습니다. 다시 찍으세요.")
                    return
                self.boxes.append(quad)
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.undo()

    def draw(self, bgr: np.ndarray, index: int, total: int, class_name: str) -> np.ndarray:
        vis = bgr.copy()
        for quad in self.boxes:
            _draw_quad(vis, quad, (0, 255, 0), 2)
            x1, y1 = order_quad(quad)[0]
            cv2.putText(
                vis,
                class_name,
                (int(x1), max(16, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
            )

        pts = list(self.pending)
        if self.hover is not None and 0 < len(pts) < N_CORNERS:
            preview = order_quad(np.array(pts + [self.hover], dtype=np.float32))
            _draw_quad(vis, preview, (0, 255, 255), 1)
        elif len(pts) >= 2:
            cv2.polylines(
                vis,
                [np.array(pts, dtype=np.int32)],
                False,
                (0, 255, 255),
                1,
            )
        for i, (px, py) in enumerate(self.pending):
            cv2.circle(vis, (px, py), 5, (0, 255, 255), -1)
            cv2.putText(
                vis,
                str(i + 1),
                (px + 6, py - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 255),
                1,
            )

        lines = [
            f"{index + 1}/{total}  {class_name}  boxes={len(self.boxes)}  "
            f"pts={len(self.pending)}/{N_CORNERS}  {self.w}x{self.h}",
            "click 4 corners  rmb/d=undo  r=clear  n/p  q=quit",
        ]
        y = 22
        for line in lines:
            cv2.putText(
                vis, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
            )
            y += 20
        return vis


def main() -> None:
    parser = argparse.ArgumentParser(description="click 4 corners for YOLO OBB")
    add_class_argument(parser)
    args = parser.parse_args()
    class_name = class_from_args(args)

    images = _image_paths()
    if not images:
        raise SystemExit(f"이미지가 없습니다. 먼저 python yolo/train/capture.py  → {RAW_IMAGES}")
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
    print(f"{len(images)}장. 꼭짓점 4번 클릭 (직각 아님). 우클릭/d 취소. 빈 장면은 점 없이 n.")
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
                sess.undo()
            if key == ord("r"):
                sess.clear()
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
                sess.pending.clear()
                sess.hover = None
                print(f"{images[idx].name}  boxes={len(sess.boxes)}")
    finally:
        cv2.destroyAllWindows()
    labeled = len(list(RAW_LABELS.glob("*.txt")))
    print(f"라벨 {labeled}개. 다음: python yolo/train/prepare.py")


if __name__ == "__main__":
    main()
