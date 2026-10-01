"""픽셀 박스 ↔ YOLO detect txt.

저장 형식:
  class_index cx cy w h   # 0~1, 축정렬
내부 표현:
  LabeledBox(class_id, (4, 2) 픽셀). 네 점은 가로세로 박스의 꼭짓점이다.
class_id 는 POSE_CLASSES (0 서있기, 1 눕히기, 2 비스듬히).

예전 네 점(OBB) 줄은 읽을 때 외접 가로세로 박스로 접는다.
세그 폴리곤 헬퍼는 labels_seg 용. 박스 라벨과 별개다.
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
    """네 점 → 축정렬 xyxy. SAM 프롬프트·오버레이."""
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


Polygon = list[tuple[float, float]]


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def simplify_polygon_px(
    pts: list[tuple[float, float]] | object, epsilon_ratio: float = 0.008
) -> list[tuple[float, float]]:
    import cv2

    arr = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    if len(arr) < 3:
        return [(float(x), float(y)) for x, y in arr]
    peri = cv2.arcLength(arr, True)
    eps = max(1.0, peri * epsilon_ratio)
    approx = cv2.approxPolyDP(arr, eps, True).reshape(-1, 2)
    if len(approx) < 3:
        return [(float(x), float(y)) for x, y in arr]
    return [(float(x), float(y)) for x, y in approx]


def load_yolo_seg_txt(path: Path) -> list[tuple[int, Polygon]]:
    if not path.exists():
        return []
    polygons: list[tuple[int, Polygon]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            continue
        cid = _class_id(parts[0])
        coords = [float(v) for v in parts[1:]]
        poly = [
            (_clamp01(coords[i]), _clamp01(coords[i + 1]))
            for i in range(0, len(coords), 2)
        ]
        if len(poly) >= 3:
            polygons.append((cid, poly))
    return polygons


def save_yolo_seg_txt(
    path: Path,
    polygons: list[Polygon] | list[tuple[int, Polygon]],
    class_ids: list[int] | None = None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, item in enumerate(polygons):
        if isinstance(item, tuple):
            cid, poly = int(item[0]), item[1]
        else:
            cid = CLASS_ID if class_ids is None or i >= len(class_ids) else int(class_ids[i])
            poly = item
        if len(poly) < 3:
            continue
        body = " ".join(f"{_clamp01(x):.6f} {_clamp01(y):.6f}" for x, y in poly)
        lines.append(f"{cid} {body}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit("라벨 UI는 python yolo/train/label.py")
