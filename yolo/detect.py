#!/usr/bin/env python3
"""학습한 1-클래스 가중치로 Orbbec 실시간 인식.

  python yolo/detect.py
  python yolo/detect.py --seg
  python yolo/detect.py --weights yolo/weights/best.pt --conf 0.35

GPU 설치 확인만 하려면 COCO nano 를 잠깐 쓸 수 있다 (우리 물건은 안 잡힘):

  python yolo/detect.py --pretrained
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from yolo.config import BEST_PT, BEST_SEG_PT, DETECT_CONF, add_class_argument, class_from_args, quiet_gtk

quiet_gtk()

import cv2  # noqa: E402

from vision.camera import grab_bgr, open_camera  # noqa: E402

WIN = "YOLO detect"


def _device() -> str | int:
    import torch

    return 0 if torch.cuda.is_available() else "cpu"


def main() -> None:
    parser = argparse.ArgumentParser(description="1-class live detect")
    parser.add_argument("--seg", action="store_true", help="세그 가중치 + 마스크 중심")
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
        weights = "yolo11n-seg.pt" if args.seg else "yolo11n.pt"
        print("사전학습 COCO. 우리 클래스는 여기 없다. GPU·카메라 확인용.")
    else:
        weights = args.weights if args.weights is not None else (
            BEST_SEG_PT if args.seg else BEST_PT
        )
        if not Path(weights).exists():
            hint = "python yolo/train.py --seg" if args.seg else "python yolo/train.py"
            raise SystemExit(
                f"가중치 없음: {weights}\n"
                f"{hint} 를 먼저 하거나, GPU 확인만 하면 --pretrained"
            )

    cap = open_camera()
    cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
    preview = grab_bgr(cap)
    if preview is None:
        cap.release()
        raise SystemExit("프레임을 읽을 수 없습니다.")
    vis0 = preview.copy()
    cv2.putText(
        vis0,
        "loading YOLO...",
        (8, 32),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    cv2.imshow(WIN, vis0)
    cv2.waitKey(1)
    print(f"창 '{WIN}' 을 띄웠습니다. 모델 로드 중...")

    from ultralytics import YOLO

    device = _device()
    model = YOLO(str(weights))
    print(f"class={class_name}  device={device}  conf={args.conf}  q=종료")
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
            boxes = result.boxes
            masks = result.masks
            if boxes is not None:
                n = len(boxes)
                for i, box in enumerate(boxes):
                    xyxy = box.xyxy[0].tolist()
                    cx = (xyxy[0] + xyxy[2]) / 2.0
                    cy = (xyxy[1] + xyxy[3]) / 2.0
                    kind = "box"
                    if masks is not None and masks.xy is not None and i < len(masks.xy):
                        pts = masks.xy[i]
                        if pts is not None and len(pts) >= 3:
                            cx = float(pts[:, 0].mean())
                            cy = float(pts[:, 1].mean())
                            kind = "mask"
                    conf = float(box.conf[0]) if box.conf is not None else 0.0
                    cls_id = int(box.cls[0]) if box.cls is not None else 0
                    names = result.names
                    if isinstance(names, dict):
                        name = names.get(cls_id, class_name)
                    else:
                        name = names[cls_id] if 0 <= cls_id < len(names) else class_name
                    cv2.circle(vis, (int(cx), int(cy)), 4, (0, 0, 255), -1)
                    cv2.putText(
                        vis,
                        f"{name} {conf:.2f} {kind} ({int(cx)},{int(cy)})",
                        (int(xyxy[0]), max(16, int(xyxy[1]) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        (0, 0, 255),
                        1,
                        cv2.LINE_AA,
                    )
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
