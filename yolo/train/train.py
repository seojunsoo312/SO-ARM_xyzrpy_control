#!/usr/bin/env python3
"""YOLO11n-OBB 를 이 보드에서 1-클래스 파인튜닝한다.

COCO 에 없는 물건이어도 pretrained nano 가중치에서 시작하는 편이 낫다.
클래스 의미는 data.yaml 의 이름 하나뿐이다.

  python yolo/train/train.py
  python yolo/train/train.py --seg
  python yolo/train/train.py --epochs 50 --batch 2
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yolo.config import (
    BEST_PT,
    BEST_SEG_PT,
    add_class_argument,
    class_from_args,
    DATA_YAML,
    RUNS_DIR,
    TRAIN_BATCH,
    TRAIN_EPOCHS,
    TRAIN_IMGSZ,
    TRAIN_SEG_BATCH,
    TRAIN_TXT,
    TRAIN_WORKERS,
    WEIGHTS_DIR,
    default_start_pt,
)


def _device() -> str | int:
    import torch

    if torch.cuda.is_available():
        return 0
    print("경고: CUDA 없음. CPU 학습은 Orin Nano에서 사실상 불가능에 가깝다.")
    return "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="1-class YOLO11n-OBB train")
    parser.add_argument("--seg", action="store_true", help="yolo11n-seg + 폴리곤 라벨")
    parser.add_argument("--epochs", type=int, default=TRAIN_EPOCHS)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--imgsz", type=int, default=TRAIN_IMGSZ)
    parser.add_argument("--model", default=None, help="시작 가중치 (전이학습)")
    add_class_argument(parser)
    args = parser.parse_args()
    batch = args.batch if args.batch is not None else (
        TRAIN_SEG_BATCH if args.seg else TRAIN_BATCH
    )
    start_model = args.model or default_start_pt(seg=args.seg)
    run_name = "train_seg" if args.seg else "train"
    dest = BEST_SEG_PT if args.seg else BEST_PT

    n_train = 0
    if TRAIN_TXT.is_file():
        n_train = sum(1 for line in TRAIN_TXT.read_text(encoding="utf-8").splitlines() if line.strip())
    if n_train < 8:
        raise SystemExit("데이터셋이 없습니다. python yolo/train/prepare.py 를 먼저 실행하세요.")
    if not DATA_YAML.exists():
        raise SystemExit(f"없음: {DATA_YAML}")

    try:
        from ultralytics import YOLO
    except ModuleNotFoundError:
        raise SystemExit(
            "ultralytics 가 이 파이썬에 없습니다.\n"
            f"지금 실행: {sys.executable}\n"
            "conda activate lerobot 한 뒤, /bin/python 말고:\n"
            "  python yolo/train/train.py"
        )

    device = _device()
    print(
        f"class={class_from_args(args)}  device={device}  batch={batch}  "
        f"epochs={args.epochs}  model={start_model}"
    )
    model = YOLO(start_model)
    model.train(
        data=str(DATA_YAML),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=batch,
        device=device,
        workers=TRAIN_WORKERS,
        project=str(RUNS_DIR),
        name=run_name,
        exist_ok=True,
        cache=False,
        patience=30,
        plots=True,
    )
    run_best = RUNS_DIR / run_name / "weights" / "best.pt"
    if not run_best.exists():
        raise SystemExit(f"학습 결과 없음: {run_best}")
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_best, dest)
    print(f"복사 {run_best} → {dest}")
    print("다음: python yolo/train/detect.py --seg" if args.seg else "다음: python yolo/train/detect.py")


if __name__ == "__main__":
    main()
