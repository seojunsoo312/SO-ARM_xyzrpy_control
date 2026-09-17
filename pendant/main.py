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

import customtkinter as ctk
import numpy as np

from gui_app import PendantGui
from motion import (
    DEFAULT_PORT,
    DEFAULT_ROBOT_ID,
    DEFAULT_URDF,
    EE_FRAME,
    TCP_FRAME,
    TCP_OFFSET_IN_L6,
    URDF_JOINT_NAMES,
    Controller,
    Hardware,
    RobotKinematics,
)
from visualizer import Visualizer


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
            "(~/.cache/huggingface/lerobot/calibration/robots/so_follower), "
            "then ./calibration/so_follower"
        ),
    )
    p.add_argument("--dof-mode", type=int, default=7, choices=(6, 7))
    p.add_argument(
        "--grasp",
        action="store_true",
        help="물체 집기 teach 창을 같은 Meshcat에 연다",
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
    pose = kin.forward_tcp(q)
    off_mm = kin.tcp_offset * 1000.0

    print(f"URDF     {kin.urdf_path}")
    print(f"nq       {kin.model.nq}  joints={list(URDF_JOINT_NAMES)}")
    print(f"TCP      {TCP_FRAME} in {EE_FRAME}  offset {off_mm[0]:+.1f},{off_mm[1]:+.1f},{off_mm[2]:+.1f} mm")
    print(f"         (CAD default {tuple(np.round(TCP_OFFSET_IN_L6 * 1000, 1))})")
    xyz = pose.xyz_mm
    rpy = pose.rpy_deg
    print(f"HOME TCP x={xyz[0]:.1f} y={xyz[1]:.1f} z={xyz[2]:.1f} mm")
    print(f"HOME RPY r={rpy[0]:.1f} p={rpy[1]:.1f} y={rpy[2]:.1f} deg")
    print(f"HW       port={args.port}  id={args.robot_id}  dof={args.dof_mode}")
    print(
        f"calib    {args.calibration_dir or 'LeRobot cache (same as lerobot-calibrate), else ./calibration/so_follower'}"
    )

    viz = Visualizer(kin, open_browser=not args.no_open)
    viz.display(q)
    url = viz.url or "(meshcat server running)"
    print(f"Meshcat  {url}")

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
    PendantGui(root, ctrl, viz, meshcat_url=url)
    if args.grasp:
        from teach_grasp import attach_teach_window

        teach = attach_teach_window(root, viz, meshcat_url=url, controller=ctrl)
        if teach is not None:
            print("Teach    물체 집기 창 (같은 Meshcat, 대기/집기 위치 이동)")
    root.mainloop()


if __name__ == "__main__":
    main()
