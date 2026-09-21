"""Joint + Cartesian jog and absolute go-to."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

import time
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk
import numpy as np

from motion.controller import (
    DEFAULT_SPEED_PCT,
    JOG_ROT_DEG_S,
    JOG_ROT_MAX_DEG_S,
    JOG_ROT_MIN_DEG_S,
    JOG_VEL_MAX_MPS,
    JOG_VEL_MIN_MPS,
    JOG_VEL_MPS,
    ROT_FRAME_BASE,
    ROT_FRAME_TCP,
    SPEED_PCT_MAX,
    SPEED_PCT_MIN,
    SPEED_PRESETS,
    Controller,
    speed_pct_to_joint_vel,
    speed_pct_to_scale,
)
from motion.hw_controller import grip_100_to_user, grip_user_to_100
from motion.robot_kinematics import GRIPPER_JOINT, LEROBOT_FROM_URDF, SHUTDOWN_JOINTS_DEG, URDF_JOINT_NAMES
from visualizer import Visualizer

DISPLAY_MS = 33

def _joint_ui_name(name: str) -> str:
    """Pendant label: S1–S6 → J1–J6. Gripper stays S7."""
    if name.startswith("S") and name[1:].isdigit() and 1 <= int(name[1:]) <= 6:
        return f"J{name[1:]}"
    return name


_RPY_COLOR_WHITE = "#FFFFFF"
_RPY_COLOR = {
    "Rx": _RPY_COLOR_WHITE,
    "Ry": _RPY_COLOR_WHITE,
    "Rz": _RPY_COLOR_WHITE,
    "roll": _RPY_COLOR_WHITE,
    "pitch": _RPY_COLOR_WHITE,
    "yaw": _RPY_COLOR_WHITE,
}


class PendantGui:
    def __init__(self, root: ctk.CTk, controller: Controller, visualizer: Visualizer, *, meshcat_url: str = "") -> None:
        self.root = root
        self._ctrl = controller
        self._viz = visualizer
        self._closing = False
        self._closed = False
        self.root.title("로봇 컨트롤러")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        status = ctk.CTkLabel(
            root,
            text=f"+/− 조그 · 속도% 공통 · Meshcat: {meshcat_url or 'printed URL'}",
            anchor="w",
        )
        status.pack(side="top", fill="x", padx=12, pady=(8, 2))

        self.tcp_label = ctk.CTkLabel(root, text="TCP: —", font=ctk.CTkFont(family="monospace", size=14), anchor="w")
        self.rpy_label = ctk.CTkLabel(root, text="RxRyRz: —", font=ctk.CTkFont(family="monospace", size=14), anchor="w")
        self.err_label = ctk.CTkLabel(
            root, text="err:  (virtual)", font=ctk.CTkFont(family="monospace", size=14), anchor="w"
        )
        self.mode_label = ctk.CTkLabel(root, text="mode: virtual", font=ctk.CTkFont(family="monospace"), anchor="w")
        self.rot_frame_label = ctk.CTkLabel(
            root,
            text="rot: base (+X forward)",
            font=ctk.CTkFont(family="monospace", size=14),
            anchor="w",
        )
        self.fault_label = ctk.CTkLabel(root, text="", text_color="#f87171", anchor="w")
        self.tcp_label.pack(side="top", anchor="w", padx=12)
        self.rpy_label.pack(side="top", anchor="w", padx=12)
        self.err_label.pack(side="top", anchor="w", padx=12)
        self.mode_label.pack(side="top", anchor="w", padx=12)
        self.rot_frame_label.pack(side="top", anchor="w", padx=12)
        self.fault_label.pack(side="top", anchor="w", padx=12, pady=(0, 2))

        bar = ctk.CTkFrame(root, fg_color="transparent")
        body = ctk.CTkFrame(root, fg_color="transparent")
        body.grid_columnconfigure(0, weight=1)
        body.grid_columnconfigure(1, weight=1)

        self.root.minsize(900, 700)
        self.root.geometry("980x780")
        left = ctk.CTkFrame(body)
        right = ctk.CTkFrame(body)
        left.grid(row=0, column=0, sticky="new", padx=(0, 6))
        right.grid(row=0, column=1, sticky="new", padx=(6, 0))

        ctk.CTkLabel(left, text="Joints  (−/+ 조그 · 숫자는 Go-to)", anchor="w").pack(
            anchor="w", padx=8, pady=(8, 4)
        )
        self.joint_labels: dict[str, ctk.CTkLabel] = {}
        self.joint_entries: dict[str, ctk.CTkEntry] = {}
        for name in URDF_JOINT_NAMES:
            row = ctk.CTkFrame(left)
            row.pack(fill="x", pady=3, padx=6)
            alias = LEROBOT_FROM_URDF[name]
            unit = "0–100" if name == GRIPPER_JOINT else "deg"
            ctk.CTkLabel(
                row,
                text=f"{_joint_ui_name(name)} ({alias})",
                width=132,
                anchor="w",
                font=ctk.CTkFont(family="monospace"),
            ).pack(side="left", padx=(4, 2))
            lab = ctk.CTkLabel(
                row,
                text="—",
                width=64,
                anchor="e",
                font=ctk.CTkFont(family="monospace"),
            )
            lab.pack(side="left", padx=(0, 4))
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
            ent = ctk.CTkEntry(row, width=72)
            ent.insert(0, "50" if name == GRIPPER_JOINT else "0")
            ent.pack(side="left", padx=(8, 4))
            self.joint_entries[name] = ent
            ctk.CTkLabel(row, text=unit, width=40, anchor="w").pack(side="left")
        goto_btn = ctk.CTkFrame(left, fg_color="transparent")
        goto_btn.pack(fill="x", padx=6, pady=(4, 2))
        ctk.CTkButton(goto_btn, text="현재값 넣기", width=120, command=self._fill_joints).pack(
            side="left", padx=(0, 6)
        )
        ctk.CTkButton(goto_btn, text="관절 이동", width=110, command=self._move_joints).pack(
            side="left"
        )

        ctk.CTkLabel(left, text="속도  (조그 · 관절 이동 · TCP 이동 공통)", anchor="w").pack(
            anchor="w", padx=8, pady=(12, 4)
        )
        speed = ctk.CTkFrame(left)
        speed.pack(fill="x", padx=6, pady=(0, 4))
        preset_row = ctk.CTkFrame(speed, fg_color="transparent")
        preset_row.pack(fill="x", padx=4, pady=(6, 2))
        for name, pct in SPEED_PRESETS:
            ctk.CTkButton(
                preset_row,
                text=name,
                width=70,
                command=lambda p=pct: self._set_speed_pct(p),
            ).pack(side="left", padx=(0, 6))
        self._speed_label = ctk.CTkLabel(
            speed,
            text=self._speed_label_text(DEFAULT_SPEED_PCT),
            anchor="w",
            font=ctk.CTkFont(family="monospace"),
        )
        self._speed_label.pack(anchor="w", padx=4, pady=(4, 0))
        self._speed_slider = ctk.CTkSlider(
            speed,
            from_=SPEED_PCT_MIN,
            to=SPEED_PCT_MAX,
            number_of_steps=int(SPEED_PCT_MAX - SPEED_PCT_MIN),
            command=self._on_speed_slider,
        )
        self._speed_slider.set(DEFAULT_SPEED_PCT)
        self._speed_slider.pack(fill="x", padx=4, pady=(4, 8))
        self._ctrl.set_speed_pct(DEFAULT_SPEED_PCT)

        ctk.CTkLabel(right, text="TCP  (−/+ 조그 · 숫자는 Go-to)", anchor="w").pack(
            anchor="w", padx=8, pady=(8, 4)
        )
        self.ee_entries: dict[str, ctk.CTkEntry] = {}
        self._tcp_live_labels: dict[str, ctk.CTkLabel] = {}
        self._rpy_jog_labels: dict[str, ctk.CTkLabel] = {}
        self._cart_row(right, "X", "x", entry_key="x", unit="mm")
        self._cart_row(right, "Y", "y", entry_key="y", unit="mm")
        self._cart_row(right, "Z", "z", entry_key="z", unit="mm")

        rot_bar = ctk.CTkFrame(right, fg_color="transparent")
        rot_bar.pack(fill="x", padx=6, pady=(10, 2))
        ctk.CTkLabel(rot_bar, text="Rx/Ry/Rz jog frame", anchor="w").pack(side="left", padx=(4, 8))
        self._rot_frame_seg = ctk.CTkSegmentedButton(
            rot_bar,
            values=["base", "TCP"],
            command=self._on_rot_frame,
            width=160,
        )
        self._rot_frame_seg.set("base")
        self._rot_frame_seg.pack(side="left")

        self._cart_row(right, "Rx", "wx", entry_key="roll", unit="deg", color=_RPY_COLOR["Rx"], store_label=True)
        self._cart_row(right, "Ry", "wy", entry_key="pitch", unit="deg", color=_RPY_COLOR["Ry"], store_label=True)
        self._cart_row(right, "Rz", "wz", entry_key="yaw", unit="deg", color=_RPY_COLOR["Rz"], store_label=True)
        ee_btn = ctk.CTkFrame(right, fg_color="transparent")
        ee_btn.pack(fill="x", padx=6, pady=(4, 2))
        ctk.CTkButton(ee_btn, text="현재값 넣기", width=120, command=self._fill_ee).pack(
            side="left", padx=(0, 6)
        )
        ctk.CTkButton(ee_btn, text="TCP 이동", width=110, command=self._move_ee).pack(side="left")

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
            text=f"Rx/Ry/Rz jog  {JOG_ROT_DEG_S:.0f} deg/s",
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
        self._rpy_vel_slider.pack(fill="x", padx=4, pady=(4, 6))
        self._ctrl.set_rpy_jog_deg_s(JOG_ROT_DEG_S)

        bar.pack(side="bottom", fill="x", padx=12, pady=(4, 10))
        body.pack(side="top", fill="both", expand=True, padx=12, pady=(2, 4))
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
        self._trail_var = ctk.BooleanVar(value=True)
        ctk.CTkCheckBox(
            bar,
            text="TCP 경로",
            variable=self._trail_var,
            command=self._on_trail_toggle,
            width=90,
        ).pack(side="left", padx=(8, 0))

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
        entry_key: str,
        unit: str,
        color: str | None = None,
        store_label: bool = False,
    ) -> None:
        row = ctk.CTkFrame(parent)
        row.pack(fill="x", pady=3, padx=6)
        label_kw: dict = {"text": title, "width": 36, "anchor": "w"}
        if color is not None:
            label_kw["text_color"] = color
        lab = ctk.CTkLabel(row, **label_kw)
        lab.pack(side="left", padx=(4, 2))
        if store_label:
            self._rpy_jog_labels[title] = lab
        live = ctk.CTkLabel(
            row,
            text="—",
            width=72,
            anchor="e",
            font=ctk.CTkFont(family="monospace"),
        )
        live.pack(side="left", padx=(0, 4))
        self._tcp_live_labels[entry_key] = live
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
        ent = ctk.CTkEntry(row, width=72)
        ent.insert(0, "0")
        ent.pack(side="left", padx=(8, 4))
        self.ee_entries[entry_key] = ent
        unit_kw: dict = {"text": unit, "width": 36, "anchor": "w"}
        if color is not None:
            unit_kw["text_color"] = color
        ctk.CTkLabel(row, **unit_kw).pack(side="left")

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
        self._speed_label.configure(text=self._speed_label_text(self._ctrl.speed_pct()))

    def _on_rpy_jog_speed(self, value: float) -> None:
        deg = int(round(float(value)))
        deg = max(int(JOG_ROT_MIN_DEG_S), min(int(JOG_ROT_MAX_DEG_S), deg))
        self._ctrl.set_rpy_jog_deg_s(deg)
        self._rpy_vel_label.configure(text=f"Rx/Ry/Rz jog  {deg} deg/s")

    def _speed_label_text(self, pct: float) -> str:
        joint = speed_pct_to_joint_vel(pct)
        xyz = self._ctrl.tcp_jog_mm_s() * speed_pct_to_scale(pct)
        return f"속도 {pct:.0f}%  ·  관절 {joint:.0f}°/s  ·  XYZ {xyz:.1f} mm/s"

    def _set_speed_pct(self, pct: float) -> None:
        value = max(SPEED_PCT_MIN, min(SPEED_PCT_MAX, float(pct)))
        self._speed_slider.set(value)
        self._apply_speed_pct(value)

    def _on_speed_slider(self, value: float) -> None:
        self._apply_speed_pct(value)

    def _apply_speed_pct(self, value: float) -> None:
        pct = int(round(float(value)))
        pct = max(int(SPEED_PCT_MIN), min(int(SPEED_PCT_MAX), pct))
        self._ctrl.set_speed_pct(pct)
        self._speed_label.configure(text=self._speed_label_text(pct))

    def _on_trail_toggle(self) -> None:
        self._viz.set_tcp_trail_enabled(bool(self._trail_var.get()))

    def _on_rot_frame(self, value: str) -> None:
        frame = ROT_FRAME_TCP if str(value).upper() == "TCP" else ROT_FRAME_BASE
        self._ctrl.set_rot_frame(frame)
        self._apply_rpy_label_colors(frame)
        self._update_rot_frame_label(frame)

    def _rpy_colors(self, frame: str) -> dict[str, str]:
        return _RPY_COLOR

    def _apply_rpy_label_colors(self, frame: str) -> None:
        colors = self._rpy_colors(frame)
        for title, lab in self._rpy_jog_labels.items():
            lab.configure(text_color=colors[title])

    def _update_rot_frame_label(self, frame: str) -> None:
        if frame == ROT_FRAME_TCP:
            text = "rot: TCP (그리퍼 로컬 X/Y/Z)"
        else:
            text = "rot: base (+X forward)"
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
        if self._closed:
            return
        st = self._ctrl.snapshot()
        self._viz.display(st.q)
        xyz = st.pose.xyz_mm
        rpy = st.pose.rpy_deg
        self.tcp_label.configure(text=f"TCP:  x={xyz[0]:8.2f}  y={xyz[1]:8.2f}  z={xyz[2]:8.2f}  mm")
        self.rpy_label.configure(text=f"rot:  Rx={rpy[0]:8.2f}  Ry={rpy[1]:8.2f}  Rz={rpy[2]:8.2f}  deg")
        for key, val in (
            ("x", xyz[0]),
            ("y", xyz[1]),
            ("z", xyz[2]),
            ("roll", rpy[0]),
            ("pitch", rpy[1]),
            ("yaw", rpy[2]),
        ):
            self._tcp_live_labels[key].configure(text=f"{val:7.2f}")
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
            shown = self._joint_display(name, deg)
            unit = "" if name == GRIPPER_JOINT else "°"
            self.joint_labels[name].configure(text=f"{shown:7.2f}{unit}")
        if not self._closed:
            self.root.after(DISPLAY_MS, self._schedule_display)

    def on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        st = self._ctrl.snapshot()
        if st.connected and st.torque:
            duration_s = self._ctrl.start_joint_goto(dict(SHUTDOWN_JOINTS_DEG))
            self.fault_label.configure(text="종료: rest 자세로 이동 후 연결 해제")
            self._poll_shutdown(time.monotonic() + float(duration_s) + 1.5)
            return
        self._finish_close()

    def _poll_shutdown(self, deadline: float) -> None:
        if self._closed:
            return
        if not self._ctrl.goto_active() or time.monotonic() >= deadline:
            self._finish_close()
            return
        self.root.after(50, lambda: self._poll_shutdown(deadline))

    def _finish_close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._ctrl.stop()
        except Exception:
            pass
        self.root.destroy()
