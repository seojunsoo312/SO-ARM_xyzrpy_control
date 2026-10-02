#!/usr/bin/env python3
"""학습한 자세 3클래스 가중치로 Orbbec 실시간 인식.

  python yolo/train/detect.py
  python yolo/train/detect.py --weights yolo/weights/best.pt --conf 0.35

GPU 설치 확인만 하려면 COCO nano 를 잠깐 쓸 수 있다 (우리 물건은 안 잡힘):

  python yolo/train/detect.py --pretrained
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yolo.config import (
    BEST_PT,
    DETECT_CONF,
    POSE_CLASSES,
    add_class_argument,
    class_from_args,
    default_start_pt,
    quiet_gtk,
)

quiet_gtk()

import cv2  # noqa: E402
import numpy as np  # noqa: E402

from vision.camera import grab_bgr, open_camera  # noqa: E402

WIN = "YOLO detect"


def _device() -> str | int:
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="3-pose live detect")
    parser.add_argument("--weights", type=Path, default=None)
    parser.add_argument("--conf", type=float, default=DETECT_CONF)
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="yolo11n COCO 로 GPU/카메라만 확인. 커스텀 물건은 안 잡힘.",
    )
    add_class_argument(parser)
    args = parser.parse_args()
    class_name = class_from_args(args)

    if args.pretrained:
        weights = default_start_pt()
        print("사전학습 COCO. 우리 클래스는 여기 없다. GPU·카메라 확인용.")
    else:
        weights = args.weights if args.weights is not None else BEST_PT
        if not Path(weights).exists():
            raise SystemExit(
                f"가중치 없음: {weights}\n"
                "python yolo/train/train.py 를 먼저 하거나, GPU 확인만 하면 --pretrained"
            )

    # 카메라 연 채로 YOLO 로드하면 프레임 큐가 넘쳐 OpenNI USB 가 끊긴다.
    print("YOLO 로드 중...")
    from ultralytics import YOLO

    device = _device()
    model = YOLO(str(weights))
    if not args.pretrained:
        model.model.names = {cid: name for cid, name, _ko in POSE_CLASSES}
    print(f"class={class_name}  device={device}  conf={args.conf}  q=종료")

    cap = open_camera()
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    try:
        while True:
            frame = grab_bgr(cap)
            if frame is None:
                print("프레임을 읽을 수 없습니다.")
                break
            results = model.predict(
                frame,
                conf=args.conf,
                device=device,
                verbose=False,
                imgsz=640,
            )
            result = results[0]
            vis = result.plot()
            n = 0
            obb = getattr(result, "obb", None)
            boxes = result.boxes
            dets = boxes if boxes is not None and len(boxes) else obb
            if dets is not None:
                n = len(dets)
                for det in dets:
                    raw = det.xyxy[0]
                    if hasattr(raw, "cpu"):
                        raw = raw.cpu().numpy()
                    xyxy = [float(v) for v in np.asarray(raw).reshape(-1)[:4]]
                    cx = (xyxy[0] + xyxy[2]) / 2.0
                    cy = (xyxy[1] + xyxy[3]) / 2.0
                    cv2.circle(vis, (int(cx), int(cy)), 4, (0, 0, 255), -1)
            cv2.putText(
                vis,
                f"n={n}  q=quit",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WIN, vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
