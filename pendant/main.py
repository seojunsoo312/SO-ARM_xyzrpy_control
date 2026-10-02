#!/usr/bin/env python
"""Joint + Cartesian jog, mesh collision, absolute go-to. Connect for real hardware."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
for path in (PROJECT, ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ui_style import apply_ui_theme, enable_xft_tk

enable_xft_tk()

import customtkinter as ctk
import numpy as np

apply_ui_theme()

from gui_app import PendantGui
from motion import (
    DEFAULT_PORT,
    DEFAULT_ROBOT_ID,
    DEFAULT_URDF,
    Controller,
    Hardware,
    RobotKinematics,
)
from motion.visualizer import Visualizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="SO101_6DOF pendant — joint / Cartesian jog and go-to")
    p.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p.add_argument("--no-open", action="store_true", help="Do not auto-open the Meshcat browser")
    p.add_argument(
        "--tcp-offset-mm",
        default=None,
        help="TCP in L6_1 (gripper body) as x,y,z mm. Default: fixed/moving jaw-tip midpoint. Use 0,0,0 for wrist_roll origin.",
    )
    p.add_argument("--port", default=DEFAULT_PORT, help="Feetech serial port")
    p.add_argument("--robot-id", default=DEFAULT_ROBOT_ID, help="LeRobot robot id (calib file stem)")
    p.add_argument(
        "--calibration-dir",
        type=Path,
        default=None,
        help=(
            "Directory with {robot-id}.json. Default: lerobot-calibrate cache "
            "(~/.cache/huggingface/lerobot/calibration/robots/so_follower)"
        ),
    )
    p.add_argument("--dof-mode", type=int, default=7, choices=(6, 7))
    p.add_argument(
        "--grasp",
        action="store_true",
        help="사물 위치 패널을 조작 가능하게 연다. 없으면 같은 패널이 잠긴다",
    )
    return p.parse_args()


def parse_tcp_offset(raw: str | None) -> np.ndarray | None:
    if raw is None:
        return None
    parts = [float(x) for x in raw.split(",")]
    if len(parts) != 3:
        raise SystemExit("--tcp-offset-mm needs x,y,z")
    return np.array(parts, dtype=float) / 1000.0


def main() -> None:
    args = parse_args()
    kin = RobotKinematics(args.urdf, tcp_offset=parse_tcp_offset(args.tcp_offset_mm))
    q = kin.q_home()
    open_browser = not args.no_open
    viz = Visualizer(kin, open_browser=open_browser)
    viz.display(q)
    if not open_browser and viz.url:
        print(viz.url)

    hw = Hardware(
        port=args.port,
        robot_id=args.robot_id,
        calibration_dir=args.calibration_dir,
        dof_mode=args.dof_mode,
    )
    ctrl = Controller(kin, hardware=hw)
    ctrl.start()

    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    PendantGui(root, ctrl, viz, meshcat_url=viz.url or "", grasp=args.grasp)
    root.mainloop()


if __name__ == "__main__":
    main()
