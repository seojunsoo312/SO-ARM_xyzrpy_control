"""Joint + Cartesian jog and absolute go-to."""

from __future__ import annotations

import sys
from pathlib import Path

_PENDANT = Path(__file__).resolve().parent
_PROJECT = _PENDANT.parent
for path in (_PROJECT, _PENDANT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ui_style import apply_root_fonts, apply_ui_theme, enable_xft_tk, mono_font, ui_font

enable_xft_tk()

import time
import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk
import numpy as np

apply_ui_theme()

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
)
from motion.hw_controller import grip_100_to_user, grip_user_to_100
from motion.robot_kinematics import GRIPPER_JOINT, LEROBOT_FROM_URDF, SHUTDOWN_JOINTS_DEG, URDF_JOINT_NAMES
from motion.visualizer import Visualizer

DISPLAY_MS = 33


def _set_text(widget, text: str) -> None:
    """글자가 바뀔 때만 configure. CTk 위젯은 configure 마다 다시 그린다."""
    if widget.cget("text") != text:
        widget.configure(text=text)

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
    def __init__(
        self,
        root: ctk.CTk,
        controller: Controller,
        visualizer: Visualizer,
        *,
        meshcat_url: str = "",
        grasp: bool = False,
    ) -> None:
        self.root = root
        self._ctrl = controller
        self._viz = visualizer
        self._closing = False
        self._closed = False
        self._grasp_enabled = bool(grasp)
        self._teach = None
        self.root.title("로봇 컨트롤러")
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        apply_root_fonts(self.root)

        status = ctk.CTkLabel(
            root,
            text=f"+/− 조그 · 속도% 공통 · Meshcat: {meshcat_url or 'printed URL'}",
            anchor="w",
        )
        status.pack(side="top", fill="x", padx=12, pady=(8, 2))

        self.tcp_label = ctk.CTkLabel(root, text="TCP: —", font=mono_font(14), anchor="w")
        self.rpy_label = ctk.CTkLabel(root, text="RxRyRz: —", font=mono_font(14), anchor="w")
        self.err_label = ctk.CTkLabel(
            root, text="err:  (virtual)", font=mono_font(14), anchor="w"
        )
        self.mode_label = ctk.CTkLabel(root, text="mode: virtual", font=mono_font(), anchor="w")
        self.rot_frame_label = ctk.CTkLabel(
            root,
            text="xyz: base (+X forward) · rot: base (+X forward)",
            font=mono_font(14),
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
        body.grid_rowconfigure(0, weight=1)
        body.grid_columnconfigure(2, weight=0, minsize=500)
        body.grid_columnconfigure(3, weight=0, minsize=440)

        self.root.minsize(1760, 700)
        self.root.geometry("1980x820")
        left = ctk.CTkFrame(body)
        right = ctk.CTkFrame(body)
        self._grasp_col = ctk.CTkFrame(body)
        self._grasp_act_col = ctk.CTkFrame(body)
        left.grid(row=0, column=0, sticky="new", padx=(0, 6))
        right.grid(row=0, column=1, sticky="new", padx=(6, 6))
        self._grasp_col.grid(row=0, column=2, sticky="new", padx=(0, 6))
        self._grasp_act_col.grid(row=0, column=3, sticky="new")

        ctk.CTkLabel(left, text="Joints", anchor="w").pack(
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
                font=mono_font(),
            ).pack(side="left", padx=(4, 2))
            lab = ctk.CTkLabel(
                row,
                text="—",
                width=64,
                anchor="e",
                font=mono_font(),
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
        # TCP의 현재값/이동 버튼과 같은 높이가 되도록 아래에서 늘린다.
        self._joint_btn_spacer = ctk.CTkFrame(left, fg_color="transparent", height=0)
        self._joint_btn_spacer.pack(fill="x")
        self._joint_btn_spacer.pack_propagate(False)
        goto_btn = ctk.CTkFrame(left, fg_color="transparent")
        self._joint_goto_row = goto_btn
        goto_btn.pack(fill="x", padx=6, pady=(4, 2))
        ctk.CTkButton(goto_btn, text="현재값 넣기", width=120, command=self._fill_joints).pack(
            side="left", padx=(0, 6)
        )
        ctk.CTkButton(goto_btn, text="관절 이동", width=110, command=self._move_joints).pack(
            side="left"
        )

        ctk.CTkLabel(left, text="관절 속도", anchor="w").pack(
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
            font=mono_font(),
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

        ctk.CTkLabel(right, text="TCP", anchor="w").pack(
            anchor="w", padx=8, pady=(8, 4)
        )
        self.ee_entries: dict[str, ctk.CTkEntry] = {}
        self._xyz_frame_seg = self._jog_frame_bar(right, "XYZ jog frame", self._on_xyz_frame)
        self._tcp_live_labels: dict[str, ctk.CTkLabel] = {}
        self._rpy_jog_labels: dict[str, ctk.CTkLabel] = {}
        self._cart_row(right, "X", "x", entry_key="x", unit="mm")
        self._cart_row(right, "Y", "y", entry_key="y", unit="mm")
        self._cart_row(right, "Z", "z", entry_key="z", unit="mm")

        self._rot_frame_seg = self._jog_frame_bar(
            right, "Rx/Ry/Rz jog frame", self._on_rot_frame, top=10
        )

        self._cart_row(right, "Rx", "wx", entry_key="roll", unit="deg", color=_RPY_COLOR["Rx"], store_label=True)
        self._cart_row(right, "Ry", "wy", entry_key="pitch", unit="deg", color=_RPY_COLOR["Ry"], store_label=True)
        self._cart_row(right, "Rz", "wz", entry_key="yaw", unit="deg", color=_RPY_COLOR["Rz"], store_label=True)
        ee_btn = ctk.CTkFrame(right, fg_color="transparent")
        self._ee_goto_row = ee_btn
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
        jog_preset = ctk.CTkFrame(vel, fg_color="transparent")
        jog_preset.pack(fill="x", padx=4, pady=(6, 2))
        for name, mm_s, deg_s in (
            ("느림", JOG_VEL_MIN_MPS * 1000.0, JOG_ROT_MIN_DEG_S),
            ("보통", JOG_VEL_MPS * 1000.0, JOG_ROT_DEG_S),
            ("빠름", JOG_VEL_MAX_MPS * 1000.0, JOG_ROT_MAX_DEG_S),
        ):
            ctk.CTkButton(
                jog_preset,
                text=name,
                width=70,
                command=lambda m=mm_s, d=deg_s: self._set_jog_preset(m, d),
            ).pack(side="left", padx=(0, 6))
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
            number_of_steps=int(JOG_VEL_MAX_MPS * 1000.0 - JOG_VEL_MIN_MPS * 1000.0),
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
        self._mount_grasp_panel()
        self._fill_joints()
        self._fill_ee()
        self._schedule_display()
        self.root.after(80, self._align_joint_goto_row)
        self.root.bind("<Map>", lambda _e: self.root.after(50, self._align_joint_goto_row), add="+")

    @property
    def grasp_ready(self) -> bool:
        return self._teach is not None and self._grasp_enabled

    def _mount_grasp_panel(self) -> None:
        from teach_grasp import attach_teach_panel

        self._teach = attach_teach_panel(
            self._grasp_col,
            self._viz,
            meshcat_url="",
            controller=self._ctrl,
            enabled=self._grasp_enabled,
            actions_parent=self._grasp_act_col,
        )
        if self._teach is None:
            ctk.CTkLabel(
                self._grasp_col,
                text="사물 위치 패널을 열지 못했습니다",
                text_color="#6b7280",
                anchor="w",
            ).pack(anchor="w", padx=12, pady=12)

    def _align_joint_goto_row(self) -> None:
        """관절 현재값/이동 버튼을 TCP 현재값/이동 버튼과 같은 높이에 둔다."""
        self.root.update_idletasks()
        dy = self._ee_goto_row.winfo_rooty() - self._joint_goto_row.winfo_rooty()
        if dy > 1:
            self._joint_btn_spacer.configure(height=int(dy))

    def _jog_frame_bar(self, parent: ctk.CTkFrame, text: str, command, *, top: int = 0) -> ctk.CTkSegmentedButton:
        """base/TCP 전환을 Go-to 입력칸과 같은 x에 둔다."""
        bar = ctk.CTkFrame(parent, fg_color="transparent")
        bar.pack(fill="x", padx=6, pady=(top, 2))
        # _cart_row에서 입력칸이 시작되는 x와 맞춘다. (축 이름 + 현재값 + −/+)
        ctk.CTkLabel(bar, text=text, width=200, anchor="w").pack(side="left", padx=(4, 2))
        seg = ctk.CTkSegmentedButton(bar, values=["base", "TCP"], command=command, width=148)
        seg.set("base")
        seg.pack(side="left", padx=(8, 4))
        return seg

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
            font=mono_font(),
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
        # width <= 4 is the old Tk character width for +/−. Larger values are text buttons.
        px = 40 if width <= 4 else 88
        btn = ctk.CTkButton(
            parent,
            text=text,
            width=px,
            height=32,
            corner_radius=8,
            font=ui_font(16 if width <= 4 else 13),
        )
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
        self._rpy_vel_label.configure(text=f"Rx/Ry/Rz jog  {deg} deg/s")

    def _set_jog_preset(self, mm_s: float, deg_s: float) -> None:
        self._tcp_vel_slider.set(mm_s)
        self._rpy_vel_slider.set(deg_s)
        self._on_tcp_jog_speed(mm_s)
        self._on_rpy_jog_speed(deg_s)

    def _speed_label_text(self, pct: float) -> str:
        joint = speed_pct_to_joint_vel(pct)
        return f"관절 {joint:.0f}°/s"

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

    def _on_xyz_frame(self, value: str) -> None:
        frame = ROT_FRAME_TCP if str(value).upper() == "TCP" else ROT_FRAME_BASE
        self._ctrl.set_xyz_frame(frame)
        self._update_frame_label(frame, self._ctrl.rot_frame())

    def _on_rot_frame(self, value: str) -> None:
        frame = ROT_FRAME_TCP if str(value).upper() == "TCP" else ROT_FRAME_BASE
        self._ctrl.set_rot_frame(frame)
        self._apply_rpy_label_colors(frame)
        self._update_frame_label(self._ctrl.xyz_frame(), frame)

    def _rpy_colors(self, frame: str) -> dict[str, str]:
        return _RPY_COLOR

    def _apply_rpy_label_colors(self, frame: str) -> None:
        colors = self._rpy_colors(frame)
        for title, lab in self._rpy_jog_labels.items():
            lab.configure(text_color=colors[title])

    def _frame_text(self, frame: str) -> str:
        if frame == ROT_FRAME_TCP:
            return "TCP (그리퍼 로컬 X/Y/Z)"
        return "base (+X forward)"

    def _update_frame_label(self, xyz_frame: str, rot_frame: str) -> None:
        _set_text(
            self.rot_frame_label,
            f"xyz: {self._frame_text(xyz_frame)} · rot: {self._frame_text(rot_frame)}",
        )

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
        _set_text(self.tcp_label, f"TCP:  x={xyz[0]:8.2f}  y={xyz[1]:8.2f}  z={xyz[2]:8.2f}  mm")
        _set_text(self.rpy_label, f"rot:  Rx={rpy[0]:8.2f}  Ry={rpy[1]:8.2f}  Rz={rpy[2]:8.2f}  deg")
        for key, val in (
            ("x", xyz[0]),
            ("y", xyz[1]),
            ("z", xyz[2]),
            ("roll", rpy[0]),
            ("pitch", rpy[1]),
            ("yaw", rpy[2]),
        ):
            _set_text(self._tcp_live_labels[key], f"{val:7.2f}")
        if st.ee_err_mm is None or st.err_xyz_mm is None:
            _set_text(self.err_label, "err:  (virtual)")
        else:
            d = st.err_xyz_mm
            _set_text(
                self.err_label,
                f"err:  |Δ|={st.ee_err_mm:7.2f}  "
                f"Δx={d[0]:+7.2f}  Δy={d[1]:+7.2f}  Δz={d[2]:+7.2f}  mm",
            )
        torque = "torque on" if st.torque else "torque off"
        if st.connected:
            _set_text(self.mode_label, f"mode: real · {torque}")
            _set_text(self.connect_btn, "Disconnect")
        else:
            _set_text(self.mode_label, "mode: virtual")
            if self.connect_btn.cget("state") != "disabled":
                _set_text(self.connect_btn, "Connect")
        self._update_frame_label(st.xyz_frame, st.rot_frame)
        xyz_val = "TCP" if st.xyz_frame == ROT_FRAME_TCP else "base"
        if self._xyz_frame_seg.get() != xyz_val:
            self._xyz_frame_seg.set(xyz_val)
        seg_val = "TCP" if st.rot_frame == ROT_FRAME_TCP else "base"
        if self._rot_frame_seg.get() != seg_val:
            self._rot_frame_seg.set(seg_val)
            self._apply_rpy_label_colors(st.rot_frame)
        _set_text(self.fault_label, st.fault)
        for name, deg in st.joints_deg.items():
            shown = self._joint_display(name, deg)
            unit = "" if name == GRIPPER_JOINT else "°"
            _set_text(self.joint_labels[name], f"{shown:7.2f}{unit}")
        if not self._closed:
            self.root.after(DISPLAY_MS, self._schedule_display)

    def on_close(self) -> None:
        if self._closing:
            return
        self._closing = True
        if self._teach is not None:
            self._teach.abort_user_pick_place()
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
        try:
            self._viz.close()
        except Exception:
            pass
        if self._teach is not None:
            self._teach._cancel_register_poll()
        self.root.destroy()
