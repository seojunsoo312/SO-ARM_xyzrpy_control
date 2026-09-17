#!/usr/bin/env python3
"""OBB 라벨을 SAM 프롬프트로 마스크(폴리곤)로 바꾼다.

기존 OBB(`datasets/raw/labels`)는 그대로 둔다. SAM에는 AABB(외접 박스)를 넣는다.
결과는 `datasets/raw/labels_seg`. 빈 책상(박스 없음)은 빈 txt.

  python yolo/train/auto_seg.py
  python yolo/train/auto_seg.py --force          # 이미 있는 마스크도 다시
  python yolo/train/auto_seg.py --model sam_b.pt # 더 정확, VRAM 큼
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yolo.train.boxes import load_yolo_txt, quad_to_xyxy, save_yolo_seg_txt, simplify_polygon_px
from yolo.config import (
    RAW_IMAGES,
    RAW_LABELS,
    RAW_LABELS_SEG,
    SAM_MODEL,
    quiet_gtk,
)

quiet_gtk()


def _images() -> list[Path]:
    return sorted(list(RAW_IMAGES.glob("*.jpg")) + list(RAW_IMAGES.glob("*.png")))


def _device() -> str | int:
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


def _polygons_from_result(result, w: int, h: int):
    if result.masks is None or result.masks.xy is None:
        return []
    out = []
    for pts in result.masks.xy:
        if pts is None or len(pts) < 3:
            continue
        simp = simplify_polygon_px(pts)
        if len(simp) < 3:
            continue
        out.append([(x / w, y / h) for x, y in simp])
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="OBB labels → SAM polygons")
    parser.add_argument("--model", default=SAM_MODEL, help="ultralytics SAM 가중치")
    parser.add_argument("--imgsz", type=int, default=640, help="Orin은 640. OOM이면 512")
    parser.add_argument("--force", action="store_true", help="labels_seg 가 있어도 덮어씀")
    args = parser.parse_args()

    images = _images()
    if not images:
        raise SystemExit(f"이미지가 없습니다: {RAW_IMAGES}")

    try:
        from ultralytics import SAM
        import cv2
        import torch
    except ModuleNotFoundError:
        raise SystemExit(
            "ultralytics 가 이 파이썬에 없습니다.\n"
            f"지금 실행: {sys.executable}\n"
            "conda activate lerobot 한 뒤: python yolo/train/auto_seg.py"
        )

    RAW_LABELS_SEG.mkdir(parents=True, exist_ok=True)
    device = _device()
    print(f"SAM={args.model}  device={device}  imgsz={args.imgsz}")
    print("첫 실행이면 mobile_sam.pt 를 받습니다. 펜던트는 끄세요.")
    sam = SAM(args.model)

    done = skipped = empty = failed = 0
    for img_path in images:
        lab = RAW_LABELS / f"{img_path.stem}.txt"
        dest = RAW_LABELS_SEG / f"{img_path.stem}.txt"
        if not lab.exists():
            print(f"건너뜀(OBB 라벨 없음): {img_path.name}")
            skipped += 1
            continue
        if dest.exists() and not args.force:
            skipped += 1
            continue
        bgr = cv2.imread(str(img_path))
        if bgr is None:
            print(f"읽기 실패: {img_path}")
            failed += 1
            continue
        h, w = bgr.shape[:2]
        boxes = load_yolo_txt(lab, w, h)
        if not boxes:
            save_yolo_seg_txt(dest, [])
            empty += 1
            print(f"{img_path.name}  빈 장면")
            continue
        xyxy = [quad_to_xyxy(quad) for quad in boxes]
        try:
            results = sam.predict(
                bgr,
                bboxes=xyxy,
                verbose=False,
                device=device,
                imgsz=args.imgsz,
            )
            polys = _polygons_from_result(results[0], w, h) if results else []
        except Exception as exc:
            print(f"실패 {img_path.name}: {exc}")
            failed += 1
            continue
        if len(polys) != len(boxes):
            print(
                f"주의 {img_path.name}: 박스 {len(boxes)} → 마스크 {len(polys)}. "
                "review_seg 에서 확인."
            )
        save_yolo_seg_txt(dest, polys)
        done += 1
        print(f"{img_path.name}  masks={len(polys)}")
        del results
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"완료 {done}  빈장면 {empty}  건너뜀 {skipped}  실패 {failed}")
    print("다음: python yolo/train/review_seg.py")


if __name__ == "__main__":
    main()
