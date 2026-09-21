#!/usr/bin/env python3
"""Orbbec RGB에서 학습용 프레임을 저장한다.

펜던트·비전 캘리브와 동시에 켜지 말 것. 카메라가 하나만 열린다.
열리면 Orbbec filters 창이 같이 뜬다 (Color exposure/gain 등).

  프로젝트 루트에서
  conda activate lerobot
  python yolo/train/capture.py
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from yolo.config import (
    RAW_IMAGES,
    RAW_LABELS,
    RAW_LABELS_SEG,
    add_class_argument,
    class_from_args,
    quiet_gtk,
)

quiet_gtk()

import cv2  # noqa: E402

from vision.camera import grab_bgr, open_camera  # noqa: E402

WIN = "YOLO capture"


def _stamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]


def _list_saved() -> list[Path]:
    return sorted(list(RAW_IMAGES.glob("*.jpg")) + list(RAW_IMAGES.glob("*.png")))


def _undo_last(saved: list[Path]) -> None:
    if not saved:
        print("지울 사진이 없습니다.")
        return
    path = saved.pop()
    path.unlink(missing_ok=True)
    (RAW_LABELS / f"{path.stem}.txt").unlink(missing_ok=True)
    (RAW_LABELS_SEG / f"{path.stem}.txt").unlink(missing_ok=True)
    print(f"삭제 {path.name}  남은 {len(saved)}")


def main() -> None:
    parser = argparse.ArgumentParser(description="save RGB frames for YOLO")
    add_class_argument(parser)
    args = parser.parse_args()
    class_name = class_from_args(args)

    RAW_IMAGES.mkdir(parents=True, exist_ok=True)
    cap = open_camera()
    saved = _list_saved()
    print(f"클래스={class_name}  저장={RAW_IMAGES}")
    print("SPACE=저장  u=마지막 삭제  q=종료. 위치·회전·조명·가림·빈 책상을 섞을 것.")
    try:
        while True:
            frame = grab_bgr(cap)
            if frame is None:
                print("프레임을 읽을 수 없습니다.")
                break
            vis = frame.copy()
            h, w = vis.shape[:2]
            cv2.putText(
                vis,
                f"{class_name}  saved={len(saved)}  {w}x{h}  SPACE=save  u=undo  q=quit",
                (8, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
            cv2.imshow(WIN, vis)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord("u"):
                _undo_last(saved)
            if key == ord(" "):
                path = RAW_IMAGES / f"{_stamp()}.jpg"
                cv2.imwrite(str(path), frame)
                saved.append(path)
                print(f"저장 {path.name}  total={len(saved)}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    print(f"총 {len(saved)}장. 다음: python yolo/train/label.py")


if __name__ == "__main__":
    main()
