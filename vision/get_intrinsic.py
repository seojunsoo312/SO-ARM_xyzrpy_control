#!/usr/bin/env python3
"""Orbbec SDK 공장 intrinsic → vision/calib_data/intrinsics.json

Viewer는 필요 없다. SDK 라이브러리와 카메라만 있으면 된다. Viewer가 켜져 있으면 실패한다.

  프로젝트 루트에서
  python vision/get_intrinsic.py
  ORBBEC_SDK_DIR=/path/to/OrbbecSDK python vision/get_intrinsic.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from vision.calib import INTRINSICS_JSON, save_intrinsics
from vision.camera import FRAME_HEIGHT, FRAME_WIDTH
from vision.rgbd import read_factory_intrinsics


def main() -> None:
    parser = argparse.ArgumentParser(description="Orbbec SDK factory color intrinsics")
    parser.add_argument("--width", type=int, default=FRAME_WIDTH)
    parser.add_argument("--height", type=int, default=FRAME_HEIGHT)
    parser.add_argument("--depth-width", type=int, default=640)
    parser.add_argument("--depth-height", type=int, default=400)
    parser.add_argument("--out", type=Path, default=INTRINSICS_JSON)
    args = parser.parse_args()

    payload = read_factory_intrinsics(
        color_size=(args.width, args.height),
        depth_size=(args.depth_width, args.depth_height),
    )
    dest = save_intrinsics(payload, args.out)
    K = payload["camera_matrix"]
    size = payload["image_size"]
    print(f"color {size[0]}x{size[1]}")
    print(f"fx={K[0][0]:.3f}  fy={K[1][1]:.3f}  cx={K[0][2]:.3f}  cy={K[1][2]:.3f}")
    print("dist", [round(v, 6) for v in payload["dist_coeffs"]])
    print(f"저장 {dest}")


if __name__ == "__main__":
    main()
