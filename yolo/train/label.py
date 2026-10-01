#!/usr/bin/env python3
"""두 점을 찍어 YOLO 가로세로 박스 라벨을 만든다.

자세는 0 서있기, 1 눕히기, 2 비스듬히. 숫자 키로 고른 뒤 대각 두 점을 찍는다.
이미 있는 박스 안을 클릭하면 그 박스만 지금 자세로 바꾼다. 좌표는 그대로다.

  python yolo/train/label.py

우클릭 또는 d = 마지막 점/박스 취소. 빈 장면은 점 없이 n.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yolo.config import POSE_CLASSES, RAW_IMAGES, RAW_LABELS, pose_name, quiet_gtk

quiet_gtk()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from yolo.train.boxes import (  # noqa: E402
    LabeledBox,
    aabb_quad,
    load_yolo_txt,
    quad_to_xyxy,
    save_yolo_txt,
)

WIN = "YOLO box label"
MIN_SIDE = 6
_CLASS_BGR = {
    0: (80, 220, 80),
    1: (255, 180, 0),
    2: (80, 80, 255),
}


def _image_paths() -> list[Path]:
    return sorted(list(RAW_IMAGES.glob("*.jpg")) + list(RAW_IMAGES.glob("*.png")))


def _label_path(image: Path) -> Path:
    return RAW_LABELS / f"{image.stem}.txt"


def _box_ok(quad) -> bool:
    x1, y1, x2, y2 = quad_to_xyxy(quad)
    return (x2 - x1) >= MIN_SIDE and (y2 - y1) >= MIN_SIDE


def _draw_box(vis: np.ndarray, quad, color, thick: int) -> None:
    x1, y1, x2, y2 = [int(round(v)) for v in quad_to_xyxy(quad)]
    cv2.rectangle(vis, (x1, y1), (x2, y2), color, thick)


def _hit_box(boxes: list[LabeledBox], x: int, y: int) -> int | None:
    for i in range(len(boxes) - 1, -1, -1):
        x1, y1, x2, y2 = quad_to_xyxy(boxes[i].quad)
        if x1 <= x <= x2 and y1 <= y <= y2:
            return i
    return None


@dataclass
class Session:
    boxes: list[LabeledBox] = field(default_factory=list)
    pending: list[tuple[int, int]] = field(default_factory=list)
    hover: tuple[int, int] | None = None
    active_class: int = 0
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
            if not self.pending:
                hit = _hit_box(self.boxes, x, y)
                if hit is not None:
                    box = self.boxes[hit]
                    self.boxes[hit] = LabeledBox(self.active_class, box.quad)
                    print(f"박스 {hit + 1} → {pose_name(self.active_class, korean=True)}")
                    return
            self.pending.append((x, y))
            if len(self.pending) >= 2:
                (x1, y1), (x2, y2) = self.pending
                quad = aabb_quad(x1, y1, x2, y2)
                self.pending.clear()
                if not _box_ok(quad):
                    print("박스가 너무 작습니다. 다시 찍으세요.")
                    return
                self.boxes.append(LabeledBox(self.active_class, quad))
        elif event == cv2.EVENT_RBUTTONDOWN:
            self.undo()

    def draw(self, bgr: np.ndarray, index: int, total: int) -> np.ndarray:
        vis = bgr.copy()
        for box in self.boxes:
            color = _CLASS_BGR.get(box.class_id, (0, 255, 0))
            _draw_box(vis, box.quad, color, 2)
            x1, y1, _x2, _y2 = quad_to_xyxy(box.quad)
            cv2.putText(
                vis,
                pose_name(box.class_id, korean=True),
                (int(x1), max(16, int(y1) - 6)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
            )

        if self.pending and self.hover is not None:
            x1, y1 = self.pending[0]
            _draw_box(vis, aabb_quad(x1, y1, self.hover[0], self.hover[1]), (0, 255, 255), 1)
        for px, py in self.pending:
            cv2.circle(vis, (px, py), 5, (0, 255, 255), -1)

        active = pose_name(self.active_class, korean=True)
        lines = [
            f"{index + 1}/{total}  다음={active}  boxes={len(self.boxes)}  "
            f"pts={len(self.pending)}/2  {self.w}x{self.h}",
            "0 서있기  1 눕히기  2 비스듬히  |  박스 안 클릭=그 박스만 변경",
            "click 2 corners  rmb/d=undo  r=clear  n/p  q=quit",
        ]
        y = 22
        for line in lines:
            cv2.putText(
                vis, line, (8, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA
            )
            y += 20
        return vis


def _set_active_class(sess: Session, key: int) -> None:
    for cid, _name, ko in POSE_CLASSES:
        if key == ord(str(cid)):
            sess.active_class = cid
            print(f"다음 박스: {ko}")
            return


def main() -> None:
    parser = argparse.ArgumentParser(description="click 2 corners for a YOLO box")
    parser.parse_args()

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
    print(
        f"{len(images)}장. 0/1/2 로 자세를 고르고 대각 두 점. "
        "있는 박스 안을 클릭하면 자세만 바뀐다. 우클릭/d 취소."
    )
    try:
        while True:
            vis = sess.draw(bgr, idx, len(images))
            cv2.imshow(WIN, vis)
            key = cv2.waitKey(20) & 0xFF
            if key == 255:
                continue
            _set_active_class(sess, key)
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
