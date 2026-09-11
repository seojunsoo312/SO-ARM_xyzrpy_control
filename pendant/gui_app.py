"""Joint + Cartesian jog and absolute go-to."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk
import numpy as np

from motion.controller import (
    DEFAULT_ARM_TORQUE_PCT,
    DEFAULT_GRIPPER_TORQUE_PCT,
    GRIPPER_VEL_UNIT_S,
    JOG_ROT_DEG_S,
    JOG_ROT_MAX_DEG_S,
    JOG_ROT_MIN_DEG_S,
    JOG_VEL_MAX_MPS,
    JOG_VEL_MIN_MPS,
    JOG_VEL_MPS,
    JOINT_VEL_DEG_S,
    ROT_FRAME_BASE,
    ROT_FRAME_TCP,
    TORQUE_PCT_MAX,
    TORQUE_PCT_MIN,
    Controller,
)
from motion.hw_controller import grip_100_to_user, grip_user_to_100
from motion.robot_kinematics import GRIPPER_JOINT, LEROBOT_FROM_URDF, URDF_JOINT_NAMES
from visualizer import Visualizer

DISPLAY_MS = 33

# Meshcat triad: X red, Y green, Z blue. TCP 모드 RPY = TCP 축.
_RPY_COLOR_TCP = {
    "Roll": "#E53935",
    "Pitch": "#43A047",
    "Yaw": "#1E88E8",
    "roll": "#E53935",
    "pitch": "#43A047",
    "yaw": "#1E88E8",
}
# Base 모드: 초기자세에서 보인 TCP 색 대응 (Roll→파랑, Pitch→빨강, Yaw→초록).
_RPY_COLOR_BASE = {
    "Roll": "#1E88E8",
    "Pitch": "#E53935",
    "Yaw": "#43A047",
    "roll": "#1E88E8",
    "pitch": "#E53935",
    "yaw": "#43A047",
}


class PendantGui:
    def __init__(self, root: ctk.CTk, controller: Controller, visualizer: Visualizer, *, meshcat_url: str = "") -> None:
        self.root = root
        self._ctrl = controller
        self._viz = visualizer
        self.root.title("SO101_6DOF pendant — jog / go-to")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        status = ctk.CTkLabel(
            root,
            text=(
                f"Hold +/− to jog. Joint {JOINT_VEL_DEG_S:.0f}°/s · "
                f"gripper {GRIPPER_VEL_UNIT_S:.0f}/s · "
                f"XYZ {JOG_VEL_MIN_MPS * 1000:.0f}–{JOG_VEL_MAX_MPS * 1000:.0f} mm/s · "
                f"RPY {JOG_ROT_MIN_DEG_S:.0f}–{JOG_ROT_MAX_DEG_S:.0f}°/s · "
                f"Move EE=XYZ/RPY · Move joints=토크% · "
                f"Meshcat: {meshcat_url or 'open the printed URL'}"
            ),
            anchor="w",
        )
        status.pack(fill="x", padx=12, pady=(12, 4))

        self.tcp_label = ctk.CTkLabel(root, text="TCP: —", font=ctk.CTkFont(family="monospace", size=14), anchor="w")
        self.rpy_label = ctk.CTkLabel(root, text="RPY: —", font=ctk.CTkFont(family="monospace", size=14), anchor="w")
        self.err_label = ctk.CTkLabel(
            root, text="err:  (virtual)", font=ctk.CTkFont(family="monospace", size=14), anchor="w"
        )
        self.mode_label = ctk.CTkLabel(root, text="mode: virtual", font=ctk.CTkFont(family="monospace"), anchor="w")
        self.rot_frame_label = ctk.CTkLabel(
            root,
            text="rot: base (project Rz180°)",
            font=ctk.CTkFont(family="monospace", size=14),
            anchor="w",
        )
        self.fault_label = ctk.CTkLabel(root, text="", text_color="#f87171", anchor="w")
        self.tcp_label.pack(anchor="w", padx=12)
        self.rpy_label.pack(anchor="w", padx=12)
        self.err_label.pack(anchor="w", padx=12)
        self.mode_label.pack(anchor="w", padx=12)
        self.rot_frame_label.pack(anchor="w", padx=12)
        self.fault_label.pack(anchor="w", padx=12, pady=(0, 4))

        bar = ctk.CTkFrame(root, fg_color="transparent")
        bar.pack(side="bottom", fill="x", padx=12, pady=(4, 12))
        ctk.CTkButton(bar, text="HOME", width=80, command=self._ctrl.home).pack(side="left", padx=(0, 8))
        ctk.CTkButton(bar, text="초기자세", width=90, command=self._goto_init_pose).pack(
            side="left", padx=(0, 8)
        )
        ctk.CTkButton(bar, text="Stop", width=80, command=self._ctrl.stop_all).pack(side="left", padx=(0, 8))
        ctk.CTkButton(
            bar,
            text="E-stop",
            width=90,
            fg_color="#b91c1c",
            hover_color="#991b1b",
            command=self._ctrl.estop,
        ).pack(side="left", padx=(0, 8))
        self.connect_btn = ctk.CTkButton(bar, text="Connect", width=110, command=self._toggle_connect)
        self.connect_btn.pack(side="right")

        body = ctk.CTkFrame(root, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=12, pady=8)
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)
        body.grid_rowconfigure(0, weight=1)

        self.root.minsize(900, 920)
        self.root.geometry("980x980")
        left = ctk.CTkScrollableFrame(body, width=440, height=700)
        right = ctk.CTkScrollableFrame(body, width=440, height=700)
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
        right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))

        ctk.CTkLabel(left, text="Joints", anchor="w").pack(anchor="w", padx=8, pady=(8, 4))
        self.joint_labels: dict[str, ctk.CTkLabel] = {}
        for name in URDF_JOINT_NAMES:
            row = ctk.CTkFrame(left)
            row.pack(fill="x", pady=3, padx=6)
            alias = LEROBOT_FROM_URDF[name]
            unit = "0–100" if name == GRIPPER_JOINT else "deg"
            lab = ctk.CTkLabel(
                row,
                text=f"{name} ({alias}): — {unit}",
                width=280,
                anchor="w",
                font=ctk.CTkFont(family="monospace"),
            )
            lab.pack(side="left", padx=(4, 8))
            self.joint_labels[name] = lab
            self._hold_button(
                row,
                "−",
                lambda j=name: self._ctrl.set_joint_jog(j, -1),
                lambda j=name: self._ctrl.set_joint_jog(j, 0),
            )
            self._hold_button(
                row,
                "+",
                lambda j=name: self._ctrl.set_joint_jog(j, 1),
                lambda j=name: self._ctrl.set_joint_jog(j, 0),
            )

        ctk.CTkLabel(left, text="Torque %  (Connect 후 모터에 적용)", anchor="w").pack(
            anchor="w", padx=8, pady=(12, 4)
        )
        self._torque_labels: dict[str, ctk.CTkLabel] = {}
        self._torque_sliders: dict[str, ctk.CTkSlider] = {}
        self._torque_after: dict[str, str | None] = {}
        for name in URDF_JOINT_NAMES:
            default = (
                DEFAULT_GRIPPER_TORQUE_PCT if name == GRIPPER_JOINT else DEFAULT_ARM_TORQUE_PCT
            )
            row = ctk.CTkFrame(left)
            row.pack(fill="x", pady=2, padx=6)
            alias = LEROBOT_FROM_URDF[name]
            lab = ctk.CTkLabel(
                row,
                text=f"{name} ({alias})  {default:.0f}%",
                width=220,
                anchor="w",
                font=ctk.CTkFont(family="monospace"),
            )
            lab.pack(side="left", padx=(4, 8))
            self._torque_labels[name] = lab
            slider = ctk.CTkSlider(
                row,
                from_=TORQUE_PCT_MIN,
                to=TORQUE_PCT_MAX,
                number_of_steps=int(TORQUE_PCT_MAX - TORQUE_PCT_MIN),
                command=lambda v, j=name: self._on_torque_slider(j, v),
            )
            slider.set(default)
            slider.pack(side="left", fill="x", expand=True, padx=(0, 4))
            self._torque_sliders[name] = slider
            self._torque_after[name] = None
            self._ctrl.set_joint_torque_pct(name, default)

        ctk.CTkLabel(left, text="Go-to joints (arm deg · gripper 0–100)", anchor="w").pack(
            anchor="w", padx=8, pady=(12, 4)
        )
        self.joint_entries: dict[str, ctk.CTkEntry] = {}
        for name in URDF_JOINT_NAMES:
            row = ctk.CTkFrame(left)
            row.pack(fill="x", pady=2, padx=6)
            unit = "0–100" if name == GRIPPER_JOINT else "deg"
            ctk.CTkLabel(row, text=f"{name} {unit}", width=90, anchor="w").pack(side="left")
            ent = ctk.CTkEntry(row, width=90)
            ent.insert(0, "50" if name == GRIPPER_JOINT else "0")
            ent.pack(side="left", padx=6)
            self.joint_entries[name] = ent
        btn = ctk.CTkFrame(left, fg_color="transparent")
        btn.pack(fill="x", padx=6, pady=6)
        ctk.CTkButton(btn, text="Use current", width=110, command=self._fill_joints).pack(side="left", padx=(0, 6))
        ctk.CTkButton(btn, text="Move joints", width=110, command=self._move_joints).pack(side="left")

        ctk.CTkLabel(right, text="TCP (jaw-tip midpoint, project base Rz180°)", anchor="w").pack(
            anchor="w", padx=8, pady=(8, 4)
        )
        self._cart_row(right, "X", "x")
        self._cart_row(right, "Y", "y")
        self._cart_row(right, "Z", "z")

        rot_bar = ctk.CTkFrame(right, fg_color="transparent")
        rot_bar.pack(fill="x", padx=6, pady=(10, 2))
        ctk.CTkLabel(rot_bar, text="RPY jog frame", anchor="w").pack(side="left", padx=(4, 8))
        self._rot_frame_seg = ctk.CTkSegmentedButton(
            rot_bar,
            values=["base", "TCP"],
            command=self._on_rot_frame,
            width=160,
        )
        self._rot_frame_seg.set("base")
        self._rot_frame_seg.pack(side="left")

        self._rpy_jog_labels: dict[str, ctk.CTkLabel] = {}
        self._cart_row(right, "Roll", "wx", color=_RPY_COLOR_BASE["Roll"], store_label=True)
        self._cart_row(right, "Pitch", "wy", color=_RPY_COLOR_BASE["Pitch"], store_label=True)
        self._cart_row(right, "Yaw", "wz", color=_RPY_COLOR_BASE["Yaw"], store_label=True)

        grip = ctk.CTkFrame(right)
        grip.pack(fill="x", pady=(12, 3), padx=6)
        ctk.CTkLabel(grip, text="Gripper", width=70, anchor="w").pack(side="left", padx=(4, 8))
        self._hold_button(
            grip,
            "Close",
            lambda: self._ctrl.set_joint_jog(GRIPPER_JOINT, -1),
            lambda: self._ctrl.set_joint_jog(GRIPPER_JOINT, 0),
            width=8,
        )
        self._hold_button(
            grip,
            "Open",
            lambda: self._ctrl.set_joint_jog(GRIPPER_JOINT, 1),
            lambda: self._ctrl.set_joint_jog(GRIPPER_JOINT, 0),
            width=8,
        )

        ctk.CTkLabel(right, text="Go-to TCP (mm / deg) · project base Rz180°", anchor="w").pack(
            anchor="w", padx=8, pady=(12, 4)
        )
        self.ee_entries: dict[str, ctk.CTkEntry] = {}
        self._rpy_ee_labels: dict[str, ctk.CTkLabel] = {}
        for key, label in (
            ("x", "x mm"),
            ("y", "y mm"),
            ("z", "z mm"),
            ("roll", "roll"),
            ("pitch", "pitch"),
            ("yaw", "yaw"),
        ):
            row = ctk.CTkFrame(right)
            row.pack(fill="x", pady=2, padx=6)
            label_kw: dict = {"text": label, "width": 70, "anchor": "w"}
            if key in _RPY_COLOR_BASE:
                label_kw["text_color"] = _RPY_COLOR_BASE[key]
            lab = ctk.CTkLabel(row, **label_kw)
            lab.pack(side="left")
            if key in _RPY_COLOR_BASE:
                self._rpy_ee_labels[key] = lab
            ent = ctk.CTkEntry(row, width=90)
            ent.insert(0, "0")
            ent.pack(side="left", padx=6)
            self.ee_entries[key] = ent
        btn = ctk.CTkFrame(right, fg_color="transparent")
        btn.pack(fill="x", padx=6, pady=6)
        ctk.CTkButton(btn, text="Use current", width=110, command=self._fill_ee).pack(side="left", padx=(0, 6))
        ctk.CTkButton(btn, text="Move EE", width=110, command=self._move_ee).pack(side="left")

        vel = ctk.CTkFrame(right)
        vel.pack(fill="x", padx=6, pady=(4, 8))
        self._tcp_vel_label = ctk.CTkLabel(
            vel,
            text=f"XYZ jog  {JOG_VEL_MPS * 1000:.0f} mm/s",
            anchor="w",
        )
        self._tcp_vel_label.pack(anchor="w", padx=4, pady=(6, 0))
        self._tcp_vel_slider = ctk.CTkSlider(
            vel,
            from_=JOG_VEL_MIN_MPS * 1000.0,
            to=JOG_VEL_MAX_MPS * 1000.0,
            number_of_steps=10,
            command=self._on_tcp_jog_speed,
        )
        self._tcp_vel_slider.set(JOG_VEL_MPS * 1000.0)
        self._tcp_vel_slider.pack(fill="x", padx=4, pady=(4, 4))
        self._ctrl.set_tcp_jog_mm_s(JOG_VEL_MPS * 1000.0)

        self._rpy_vel_label = ctk.CTkLabel(
            vel,
            text=f"RPY jog  {JOG_ROT_DEG_S:.0f} deg/s",
            anchor="w",
        )
        self._rpy_vel_label.pack(anchor="w", padx=4, pady=(6, 0))
        self._rpy_vel_slider = ctk.CTkSlider(
            vel,
            from_=JOG_ROT_MIN_DEG_S,
            to=JOG_ROT_MAX_DEG_S,
            number_of_steps=int(JOG_ROT_MAX_DEG_S - JOG_ROT_MIN_DEG_S),
            command=self._on_rpy_jog_speed,
        )
        self._rpy_vel_slider.set(JOG_ROT_DEG_S)
        self._rpy_vel_slider.pack(fill="x", padx=4, pady=(4, 8))
        self._ctrl.set_rpy_jog_deg_s(JOG_ROT_DEG_S)

        self.root.bind_all("<ButtonRelease-1>", self._on_global_release, add="+")
        self._fill_joints()
        self._fill_ee()
        self._schedule_display()

    def _cart_row(
        self,
        parent: ctk.CTkFrame,
        title: str,
        axis: str,
        *,
        color: str | None = None,
        store_label: bool = False,
    ) -> None:
        row = ctk.CTkFrame(parent)
        row.pack(fill="x", pady=3, padx=6)
        label_kw: dict = {"text": title, "width": 60, "anchor": "w"}
        if color is not None:
            label_kw["text_color"] = color
        lab = ctk.CTkLabel(row, **label_kw)
        lab.pack(side="left", padx=(4, 8))
        if store_label:
            self._rpy_jog_labels[title] = lab
        self._hold_button(
            row,
            "−",
            lambda a=axis: self._ctrl.set_cart_jog(a, -1),
            lambda a=axis: self._ctrl.set_cart_jog(a, 0),
        )
        self._hold_button(
            row,
            "+",
            lambda a=axis: self._ctrl.set_cart_jog(a, 1),
            lambda a=axis: self._ctrl.set_cart_jog(a, 0),
        )

    def _hold_button(self, parent: ctk.CTkFrame, text: str, on_press, on_release, *, width: int = 3) -> None:
        btn = tk.Button(parent, text=text, width=width, takefocus=False)
        btn._is_jog = True  # type: ignore[attr-defined]
        btn.pack(side="left", padx=2)
        btn.bind("<ButtonPress-1>", lambda _e: on_press())
        btn.bind("<ButtonRelease-1>", lambda _e: on_release())

    def _on_global_release(self, event: tk.Event) -> None:
        w = event.widget
        while w is not None:
            if getattr(w, "_is_jog", None):
                return
            w = getattr(w, "master", None)
        # Mouse-up outside a jog button stops jog, but must not cancel go-to.
        self._ctrl.clear_jog()

    def _joint_display(self, name: str, user_deg: float) -> float:
        if name == GRIPPER_JOINT:
            return grip_user_to_100(user_deg)
        return float(user_deg)

    def _fill_joints(self) -> None:
        st = self._ctrl.snapshot()
        for name, deg in st.joints_deg.items():
            self.joint_entries[name].delete(0, "end")
            self.joint_entries[name].insert(0, f"{self._joint_display(name, deg):.2f}")

    def _fill_ee(self) -> None:
        st = self._ctrl.snapshot()
        xyz = st.pose.xyz_mm
        rpy = st.pose.rpy_deg
        vals = {
            "x": xyz[0],
            "y": xyz[1],
            "z": xyz[2],
            "roll": rpy[0],
            "pitch": rpy[1],
            "yaw": rpy[2],
        }
        for key, val in vals.items():
            self.ee_entries[key].delete(0, "end")
            self.ee_entries[key].insert(0, f"{val:.2f}")

    def _move_joints(self) -> None:
        try:
            target: dict[str, float] = {}
            for name in URDF_JOINT_NAMES:
                raw = float(self.joint_entries[name].get())
                target[name] = grip_100_to_user(raw) if name == GRIPPER_JOINT else raw
        except ValueError:
            messagebox.showerror("Go-to", "Joint values must be numbers.")
            return
        self._ctrl.start_joint_goto(target)

    def _goto_init_pose(self) -> None:
        from motion.robot_kinematics import INIT_POSE_JOINTS_DEG

        for name in URDF_JOINT_NAMES:
            shown = (
                grip_user_to_100(INIT_POSE_JOINTS_DEG[name])
                if name == GRIPPER_JOINT
                else INIT_POSE_JOINTS_DEG[name]
            )
            self.joint_entries[name].delete(0, "end")
            self.joint_entries[name].insert(0, f"{shown:.2f}")
        self._ctrl.init_pose()

    def _move_ee(self) -> None:
        try:
            xyz = np.array([float(self.ee_entries[k].get()) for k in ("x", "y", "z")], dtype=float)
            rpy = np.array(
                [float(self.ee_entries[k].get()) for k in ("roll", "pitch", "yaw")],
                dtype=float,
            )
        except ValueError:
            messagebox.showerror("Go-to", "TCP values must be numbers.")
            return
        self._ctrl.start_ee_goto(xyz, rpy)

    def _on_tcp_jog_speed(self, value: float) -> None:
        lo = int(JOG_VEL_MIN_MPS * 1000.0)
        hi = int(JOG_VEL_MAX_MPS * 1000.0)
        mm = int(round(float(value)))
        mm = max(lo, min(hi, mm))
        self._ctrl.set_tcp_jog_mm_s(mm)
        self._tcp_vel_label.configure(text=f"XYZ jog  {mm} mm/s")

    def _on_rpy_jog_speed(self, value: float) -> None:
        deg = int(round(float(value)))
        deg = max(int(JOG_ROT_MIN_DEG_S), min(int(JOG_ROT_MAX_DEG_S), deg))
        self._ctrl.set_rpy_jog_deg_s(deg)
        self._rpy_vel_label.configure(text=f"RPY jog  {deg} deg/s")

    def _on_torque_slider(self, joint: str, value: float) -> None:
        pct = int(round(float(value)))
        pct = max(int(TORQUE_PCT_MIN), min(int(TORQUE_PCT_MAX), pct))
        alias = LEROBOT_FROM_URDF[joint]
        self._torque_labels[joint].configure(text=f"{joint} ({alias})  {pct}%")
        prev = self._torque_after.get(joint)
        if prev is not None:
            self.root.after_cancel(prev)
        self._torque_after[joint] = self.root.after(
            120, lambda j=joint, p=pct: self._apply_torque(j, p)
        )

    def _apply_torque(self, joint: str, pct: int) -> None:
        self._torque_after[joint] = None
        self._ctrl.set_joint_torque_pct(joint, pct)

    def _on_rot_frame(self, value: str) -> None:
        frame = ROT_FRAME_TCP if str(value).upper() == "TCP" else ROT_FRAME_BASE
        self._ctrl.set_rot_frame(frame)
        self._apply_rpy_label_colors(frame)
        self._update_rot_frame_label(frame)

    def _rpy_colors(self, frame: str) -> dict[str, str]:
        return _RPY_COLOR_TCP if frame == ROT_FRAME_TCP else _RPY_COLOR_BASE

    def _apply_rpy_label_colors(self, frame: str) -> None:
        colors = self._rpy_colors(frame)
        for title, lab in self._rpy_jog_labels.items():
            lab.configure(text_color=colors[title])
        for key, lab in self._rpy_ee_labels.items():
            lab.configure(text_color=colors[key])

    def _update_rot_frame_label(self, frame: str) -> None:
        if frame == ROT_FRAME_TCP:
            text = "rot: TCP (그리퍼 로컬 X/Y/Z · 빨/초/파)"
        else:
            text = "rot: base (project Rz180°)"
        self.rot_frame_label.configure(text=text)

    def _toggle_connect(self) -> None:
        st = self._ctrl.snapshot()
        if st.connected:
            self._ctrl.disconnect()
            self.connect_btn.configure(text="Connect")
            return
        self.connect_btn.configure(state="disabled")
        try:
            self._ctrl.connect()
            self.connect_btn.configure(text="Disconnect")
            self._fill_joints()
            self._fill_ee()
        except Exception as exc:
            messagebox.showerror("Connect", str(exc))
            self.connect_btn.configure(text="Connect")
        finally:
            self.connect_btn.configure(state="normal")

    def _schedule_display(self) -> None:
        st = self._ctrl.snapshot()
        self._viz.display(st.q)
        xyz = st.pose.xyz_mm
        rpy = st.pose.rpy_deg
        self.tcp_label.configure(text=f"TCP:  x={xyz[0]:8.2f}  y={xyz[1]:8.2f}  z={xyz[2]:8.2f}  mm")
        self.rpy_label.configure(text=f"RPY:  r={rpy[0]:8.2f}  p={rpy[1]:8.2f}  y={rpy[2]:8.2f}  deg")
        if st.ee_err_mm is None or st.err_xyz_mm is None:
            self.err_label.configure(text="err:  (virtual)")
        else:
            d = st.err_xyz_mm
            self.err_label.configure(
                text=(
                    f"err:  |Δ|={st.ee_err_mm:7.2f}  "
                    f"Δx={d[0]:+7.2f}  Δy={d[1]:+7.2f}  Δz={d[2]:+7.2f}  mm"
                )
            )
        torque = "torque on" if st.torque else "torque off"
        if st.connected:
            self.mode_label.configure(text=f"mode: real · {torque}")
            self.connect_btn.configure(text="Disconnect")
        else:
            self.mode_label.configure(text="mode: virtual")
            if self.connect_btn.cget("state") != "disabled":
                self.connect_btn.configure(text="Connect")
        self._update_rot_frame_label(st.rot_frame)
        seg_val = "TCP" if st.rot_frame == ROT_FRAME_TCP else "base"
        if self._rot_frame_seg.get() != seg_val:
            self._rot_frame_seg.set(seg_val)
            self._apply_rpy_label_colors(st.rot_frame)
        self.fault_label.configure(text=st.fault)
        for name, deg in st.joints_deg.items():
            alias = LEROBOT_FROM_URDF[name]
            shown = self._joint_display(name, deg)
            unit = "" if name == GRIPPER_JOINT else "°"
            self.joint_labels[name].configure(text=f"{name} ({alias}): {shown:8.2f}{unit}")
        self.root.after(DISPLAY_MS, self._schedule_display)

    def on_close(self) -> None:
        self._ctrl.stop()
        self.root.destroy()
