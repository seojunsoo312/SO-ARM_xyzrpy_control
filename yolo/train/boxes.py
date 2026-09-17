"""픽셀 네 꼭짓점 ↔ YOLO OBB txt (1클래스).

저장 형식:
  class_index x1 y1 x2 y2 x3 y3 x4 y4   # 0~1, 중심 기준 반시계
내부 표현:
  (4, 2) float 픽셀 좌표. 직각을 강제하지 않는다.

세그 폴리곤 헬퍼는 labels_seg 용. OBB 라벨과 별개다.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yolo.config import CLASS_ID

Quad = np.ndarray


def order_quad(pts: np.ndarray) -> Quad:
    """네 점을 중심 기준 반시계로 정렬한다. 클릭 순서는 상관없다."""
    pts = np.asarray(pts, dtype=np.float32).reshape(-1, 2)
    c = pts.mean(axis=0)
    ang = np.arctan2(pts[:, 1] - c[1], pts[:, 0] - c[0])
    return pts[np.argsort(ang)]


def quad_to_xyxy(quad: Quad) -> list[float]:
    """OBB 네 점 → 축정렬 AABB. SAM 프롬프트·오버레이 폴백."""
    pts = np.asarray(quad, dtype=np.float32).reshape(-1, 2)
    return [
        float(pts[:, 0].min()),
        float(pts[:, 1].min()),
        float(pts[:, 0].max()),
        float(pts[:, 1].max()),
    ]


def load_yolo_txt(path: Path, w: int, h: int) -> list[Quad]:
    if not path.exists():
        return []
    boxes: list[Quad] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 9:
            nums = [float(v) for v in parts[1:]]
            corners = np.array(nums, dtype=np.float32).reshape(4, 2)
            corners[:, 0] *= w
            corners[:, 1] *= h
            boxes.append(order_quad(corners))
        elif len(parts) == 5:
            _cid, cx, cy, bw, bh = parts
            cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
            x1, y1 = cx - bw / 2.0, cy - bh / 2.0
            x2, y2 = cx + bw / 2.0, cy + bh / 2.0
            boxes.append(
                order_quad(
                    np.array(
                        ((x1, y1), (x2, y1), (x2, y2), (x1, y2)),
                        dtype=np.float32,
                    )
                )
            )
    return boxes


def save_yolo_txt(path: Path, boxes: list[Quad], w: int, h: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for quad in boxes:
        corners = order_quad(quad).copy()
        corners[:, 0] = np.clip(corners[:, 0] / w, 0.0, 1.0)
        corners[:, 1] = np.clip(corners[:, 1] / h, 0.0, 1.0)
        body = " ".join(f"{v:.6f}" for v in corners.reshape(-1))
        lines.append(f"{CLASS_ID} {body}")
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


def load_yolo_seg_txt(path: Path) -> list[Polygon]:
    if not path.exists():
        return []
    polygons: list[Polygon] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7 or (len(parts) - 1) % 2 != 0:
            continue
        coords = [float(v) for v in parts[1:]]
        poly = [
            (_clamp01(coords[i]), _clamp01(coords[i + 1]))
            for i in range(0, len(coords), 2)
        ]
        if len(poly) >= 3:
            polygons.append(poly)
    return polygons


def save_yolo_seg_txt(path: Path, polygons: list[Polygon]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for poly in polygons:
        if len(poly) < 3:
            continue
        body = " ".join(f"{_clamp01(x):.6f} {_clamp01(y):.6f}" for x, y in poly)
        lines.append(f"{CLASS_ID} {body}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit("라벨 UI는 python yolo/train/label.py")
