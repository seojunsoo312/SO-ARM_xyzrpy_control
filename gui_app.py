"""Joint + Cartesian jog and absolute go-to."""

from __future__ import annotations

import tkinter as tk
from tkinter import messagebox

import customtkinter as ctk
import numpy as np

from controller import GOTO_DURATION_S, GRIPPER_VEL_UNIT_S, JOG_VEL_MPS, JOINT_VEL_DEG_S, Controller
from hw_controller import grip_100_to_user, grip_user_to_100
from robot_kinematics import GRIPPER_JOINT, LEROBOT_FROM_URDF, URDF_JOINT_NAMES
from visualizer import Visualizer

DISPLAY_MS = 33


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
                f"TCP {JOG_VEL_MPS * 1000:.0f} mm/s · go-to {GOTO_DURATION_S:.1f}s. "
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
        self.fault_label = ctk.CTkLabel(root, text="", text_color="#f87171", anchor="w")
        self.tcp_label.pack(anchor="w", padx=12)
        self.rpy_label.pack(anchor="w", padx=12)
        self.err_label.pack(anchor="w", padx=12)
        self.mode_label.pack(anchor="w", padx=12)
        self.fault_label.pack(anchor="w", padx=12, pady=(0, 4))

        bar = ctk.CTkFrame(root, fg_color="transparent")
        bar.pack(side="bottom", fill="x", padx=12, pady=(4, 12))
        ctk.CTkButton(bar, text="HOME", width=80, command=self._ctrl.home).pack(side="left", padx=(0, 8))
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

        ctk.CTkLabel(right, text="TCP (jaw-tip midpoint, base frame)", anchor="w").pack(anchor="w", padx=8, pady=(8, 4))
        self._cart_row(right, "X", "x")
        self._cart_row(right, "Y", "y")
        self._cart_row(right, "Z", "z")
        self._cart_row(right, "Roll", "wx")
        self._cart_row(right, "Pitch", "wy")
        self._cart_row(right, "Yaw", "wz")

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

        ctk.CTkLabel(right, text="Go-to TCP (mm / deg)", anchor="w").pack(anchor="w", padx=8, pady=(12, 4))
        self.ee_entries: dict[str, ctk.CTkEntry] = {}
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
            ctk.CTkLabel(row, text=label, width=70, anchor="w").pack(side="left")
            ent = ctk.CTkEntry(row, width=90)
            ent.insert(0, "0")
            ent.pack(side="left", padx=6)
            self.ee_entries[key] = ent
        btn = ctk.CTkFrame(right, fg_color="transparent")
        btn.pack(fill="x", padx=6, pady=6)
        ctk.CTkButton(btn, text="Use current", width=110, command=self._fill_ee).pack(side="left", padx=(0, 6))
        ctk.CTkButton(btn, text="Move EE", width=110, command=self._move_ee).pack(side="left")

        self.root.bind_all("<ButtonRelease-1>", self._on_global_release, add="+")
        self._fill_joints()
        self._fill_ee()
        self._schedule_display()

    def _cart_row(self, parent: ctk.CTkFrame, title: str, axis: str) -> None:
        row = ctk.CTkFrame(parent)
        row.pack(fill="x", pady=3, padx=6)
        ctk.CTkLabel(row, text=title, width=60, anchor="w").pack(side="left", padx=(4, 8))
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
        if getattr(event.widget, "_is_jog", None):
            return
        # Mouse-up outside a jog button stops jog, but must not cancel go-to.
        # CTkButton command also fires on ButtonRelease, then this bind_all runs.
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
        self.tcp_label.configure(text=f"TCP:  x={xyz[0]:7.1f}  y={xyz[1]:7.1f}  z={xyz[2]:7.1f}  mm")
        self.rpy_label.configure(text=f"RPY:  r={rpy[0]:7.1f}  p={rpy[1]:7.1f}  y={rpy[2]:7.1f}  deg")
        if st.ee_err_mm is None or st.err_xyz_mm is None:
            self.err_label.configure(text="err:  (virtual)")
        else:
            d = st.err_xyz_mm
            self.err_label.configure(
                text=(
                    f"err:  |Δ|={st.ee_err_mm:6.1f}  "
                    f"Δx={d[0]:+6.1f}  Δy={d[1]:+6.1f}  Δz={d[2]:+6.1f}  mm"
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
        self.fault_label.configure(text=st.fault)
        for name, deg in st.joints_deg.items():
            alias = LEROBOT_FROM_URDF[name]
            shown = self._joint_display(name, deg)
            unit = "" if name == GRIPPER_JOINT else "°"
            self.joint_labels[name].configure(text=f"{name} ({alias}): {shown:7.1f}{unit}")
        self.root.after(DISPLAY_MS, self._schedule_display)

    def on_close(self) -> None:
        self._ctrl.stop()
        self.root.destroy()
