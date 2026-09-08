"""Pixel xyxy <-> YOLO normalized xywh (one class)."""

from __future__ import annotations

from pathlib import Path

from yolo.config import CLASS_ID


def clamp_xyxy(
    x1: float, y1: float, x2: float, y2: float, w: int, h: int
) -> tuple[int, int, int, int]:
    xa, xb = sorted((x1, x2))
    ya, yb = sorted((y1, y2))
    xa = max(0, min(int(round(xa)), w - 1))
    xb = max(0, min(int(round(xb)), w - 1))
    ya = max(0, min(int(round(ya)), h - 1))
    yb = max(0, min(int(round(yb)), h - 1))
    return xa, ya, xb, yb


def xyxy_to_yolo(
    x1: int, y1: int, x2: int, y2: int, w: int, h: int
) -> tuple[int, float, float, float, float]:
    bw = max(x2 - x1, 1)
    bh = max(y2 - y1, 1)
    cx = (x1 + x2) / 2.0 / w
    cy = (y1 + y2) / 2.0 / h
    return CLASS_ID, cx, cy, bw / w, bh / h


def yolo_to_xyxy(
    cx: float, cy: float, bw: float, bh: float, w: int, h: int
) -> tuple[int, int, int, int]:
    x1 = (cx - bw / 2.0) * w
    y1 = (cy - bh / 2.0) * h
    x2 = (cx + bw / 2.0) * w
    y2 = (cy + bh / 2.0) * h
    return clamp_xyxy(x1, y1, x2, y2, w, h)


def load_yolo_txt(path: Path, w: int, h: int) -> list[tuple[int, int, int, int]]:
    if not path.exists():
        return []
    boxes: list[tuple[int, int, int, int]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) != 5:
            continue
        _cid, cx, cy, bw, bh = parts
        boxes.append(yolo_to_xyxy(float(cx), float(cy), float(bw), float(bh), w, h))
    return boxes


def save_yolo_txt(
    path: Path, boxes: list[tuple[int, int, int, int]], w: int, h: int
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for x1, y1, x2, y2 in boxes:
        cid, cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
        lines.append(f"{cid} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
    path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")


Polygon = list[tuple[float, float]]


def _clamp01(value: float) -> float:
    return min(1.0, max(0.0, value))


def simplify_polygon_px(
    pts: list[tuple[float, float]] | object, epsilon_ratio: float = 0.008
) -> list[tuple[float, float]]:
    import cv2
    import numpy as np

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
