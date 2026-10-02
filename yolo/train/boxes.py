"""픽셀 박스 ↔ YOLO detect txt.

저장 형식:
  class_index cx cy w h   # 0~1, 축정렬
내부 표현:
  LabeledBox(class_id, (4, 2) 픽셀). 네 점은 가로세로 박스의 꼭짓점이다.
class_id 는 POSE_CLASSES (0 세우기, 1 눕히기, 2 비스듬히).

예전 네 점(OBB) 줄은 읽을 때 외접 가로세로 박스로 접는다.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yolo.config import CLASS_ID

Quad = np.ndarray


@dataclass(frozen=True)
class LabeledBox:
    class_id: int
    quad: Quad


def _class_id(raw: str) -> int:
    try:
        cid = int(float(raw))
    except ValueError:
        return CLASS_ID
    return cid if cid >= 0 else CLASS_ID


def order_quad(pts: np.ndarray) -> Quad:
    """네 점을 중심 기준 반시계로 정렬한다. 클릭 순서는 상관없다."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]


def aabb_quad(x1: float, y1: float, x2: float, y2: float) -> Quad:
    """가로세로 박스 두 꼭짓점 → (4, 2)."""
    xa, xb = (x1, x2) if x1 <= x2 else (x2, x1)
    ya, yb = (y1, y2) if y1 <= y2 else (y2, y1)
    return np.array(((xa, ya), (xb, ya), (xb, yb), (xa, yb)), dtype=np.float32)


def quad_to_xyxy(quad: Quad) -> list[float]:
    """네 점 → 축정렬 xyxy."""
    pts = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    return [
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    ]


def load_yolo_txt(path: Path, w: int, h: int) -> list[LabeledBox]:
    if not path.exists():
        return []
    boxes: list[LabeledBox] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 9:
            cid = _class_id(parts[0])
            nums = [float(v) for v in parts[1:]]
            xs = [nums[i] * w for i in range(0, 8, 2)]
            ys = [nums[i] * h for i in range(1, 8, 2)]
            boxes.append(LabeledBox(cid, aabb_quad(min(xs), min(ys), max(xs), max(ys))))
        elif len(parts) == 5:
            cid = _class_id(parts[0])
            _cid, cx, cy, bw, bh = parts
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            boxes.append(
                LabeledBox(
                    cid,
                    aabb_quad(cx - bw / 2.0, cy - bh / 2.0, cx + bw / 2.0, cy + bh / 2.0),
                )
            )
    return boxes


def save_yolo_txt(path: Path, boxes: list[LabeledBox], w: int, h: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for box in boxes:
        x1, y1, x2, y2 = quad_to_xyxy(box.quad)
        bw = max(0.0, x2 - x1) / w
        bh = max(0.0, y2 - y1) / h
        cx = np.clip(((x1 + x2) / 2.0) / w, 0.0, 1.0)
        cy = np.clip(((y1 + y2) / 2.0) / h, 0.0, 1.0)
        lines.append(
            f"{int(box.class_id)} {float(cx):.6f} {float(cy):.6f} "
            f"{float(np.clip(bw, 0.0, 1.0)):.6f} {float(np.clip(bh, 0.0, 1.0)):.6f}"
        )
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit("라벨 UI는 python yolo/train/label.py")
