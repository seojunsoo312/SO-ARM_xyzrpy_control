#!/usr/bin/env python3
"""Orbbec RGB 미리보기. 프로젝트 루트에서:

  python vision/vision_test.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.camera import FRAME_HEIGHT, FRAME_WIDTH, ROTATE_180, grab_bgr, open_camera

import cv2


def main() -> None:
    cap = open_camera()
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"카메라 시작: {actual_width}x{actual_height} (요청 {FRAME_WIDTH}x{FRAME_HEIGHT}, FPS {actual_fps})")
    print(f"rotate_180={ROTATE_180}  q=종료")
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
                f"{w}x{h}  rot180={int(ROTATE_180)}  q=quit",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow("Camera Test", vis)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
