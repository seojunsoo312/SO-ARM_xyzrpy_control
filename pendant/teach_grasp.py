#!/usr/bin/env python3
"""물체 배치 + grasp yaml 값을 Meshcat에서 확인·티칭하는 GUI.

권장: 펜던트와 한 프로세스·한 Meshcat.

  python pendant/main.py              # 펜던트만 (사물 위치 칸은 잠김)
  python pendant/main.py --grasp      # 같은 창에서 사물 위치 칸 조작
  python pendant/teach_grasp.py       # 물체 창만 (가상 팔로 P/G 이동)

접근 자세·랜덤 배치·RPY 변환 계산은 motion/grasp.py, yaml·STL 읽기는 cad/model.py.
랜덤 배치 세 자세의 Rx/Ry/Rz 설명도 motion/grasp.py 맨 위에 있다.
"""

from __future__ import annotations

import argparse
import importlib.util
import sys
import threading
from pathlib import Path

import numpy as np

PENDANT = Path(__file__).resolve().parent
PROJECT = PENDANT.parent
for path in (PROJECT, PENDANT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from ui_style import apply_root_fonts, apply_ui_theme, enable_xft_tk, ui_font  # noqa: E402

enable_xft_tk()
apply_ui_theme()

from cad.model import (  # noqa: E402
    CAD_YAML,
    DEFAULT_DROP_XY_MM,
    approach_mm,
    drop_xy_mm,
    load_mesh_m,
    mesh_path as cad_mesh_path,
)
from yolo.config import quiet_gtk  # noqa: E402
from yolo.pose.register_link import (  # noqa: E402
    REG_MSG_FAIL,
    REG_MSG_NO_BOX,
    REG_MSG_NO_REPLY,
    REG_MSG_RUN,
    REG_POLL_MS,
    RegisterRequest,
)
from motion.grasp import (  # noqa: E402
    N_CAD,
    RANDOM_PLACE_MAX_TRIES,
    RANDOM_PLACE_MODES,
    AutoPreResult,
    GraspSpec,
    PlacePose,
    canonicalize_extrinsic_rpy,
    compute_auto_pre,
    load_specs,
    object_collides_robot,
    place_region_overlay_m,
    place_rpy_base_delta,
    place_rpy_body_delta,
    pose_to_T,
    rpy_body_xyz_to_extrinsic,
    rpy_extrinsic_to_body_xyz,
    sample_random_place,
    T_to_xyzrpy,
)
from motion.pick_place import PickPlaceRunner, build_pick_place_steps  # noqa: E402

APPROACH_MARKER_RADIUS_M = 0.004  # 4 mm
# TCP vs orange pre sphere: turn green when close (debug go-to).
PRE_TCP_MATCH_POS_MM = 1.5
APPROACH_PRE_COLOR = 0xFB923C
APPROACH_PRE_MATCH_COLOR = 0x22C55E
APPROACH_OTHER_COLOR = 0xC2410C
DISPLAY_MS = 33
RPY_SLIDER_MIN = -180.0
RPY_SLIDER_MAX = 180.0
# Place XYZ sliders (project base mm); match random-place workspace box.
PLACE_XY_SLIDER_MIN = -200.0
PLACE_XY_SLIDER_MAX = 200.0
PLACE_Z_SLIDER_MIN = 0.0
PLACE_Z_SLIDER_MAX = 40.0
RPY_LIVE_APPLY_MS = 40


def _T_meshcat(T_user: np.ndarray) -> np.ndarray:
    """Project-base pose → Pinocchio/Meshcat (raw URDF) world."""
    from motion.base_frame import T_urdf_from_user

    return T_urdf_from_user(T_user)


def _load_user_pick_run():
    """프로젝트 루트의 user_pickandplace.py 를 매번 다시 읽는다."""
    path = PROJECT / "user_pickandplace.py"
    if not path.is_file():
        raise FileNotFoundError(f"파일 없음: {path}")
    spec = importlib.util.spec_from_file_location("user_pickandplace_live", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("user_pickandplace.py 를 읽지 못했습니다")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run = getattr(module, "run", None)
    if not callable(run):
        raise RuntimeError("user_pickandplace.py 에 run(arm) 함수가 없습니다")
    return run


class TeachGraspGui:
    def __init__(
        self,
        root,
        *,
        visualizer,
        mesh_path: Path,
        vertices_m: np.ndarray,
        faces: np.ndarray,
        place: PlacePose,
        grasp: GraspSpec,
        meshcat_url: str,
        kinematics=None,
        controller=None,
        own_robot: bool = True,
        drive_viz: bool = False,
        embedded: bool = False,
        enabled: bool = True,
        actions_parent=None,
    ) -> None:
        import customtkinter as ctk

        self.root = root
        self._embedded = bool(embedded)
        self._panel_enabled = bool(enabled) if self._embedded else True
        self._panel_labels: list = []
        self._panel_buttons: list = []
        self._mode_cbs: list = []
        self._place_xyz_frame = "base"
        if not self._embedded:
            apply_root_fonts(root)
        self._viz = visualizer
        self._kin = kinematics
        self._ctrl = controller
        self._own_robot = bool(own_robot) and controller is None
        self._drive_viz = bool(drive_viz) and controller is not None
        self._yaml_path = CAD_YAML
        self._grasp_spec = grasp
        self._drop_xy = DEFAULT_DROP_XY_MM
        self._grasp_hide_var = None
        self._region_hide_var = None
        self._cad_loaded = False
        self._auto_pre: AutoPreResult | None = None
        self._pre_tcp_matched: bool | None = None
        self._status: ctk.CTkLabel | None = None
        self.entries: dict[str, ctk.CTkEntry] = {}
        self.sliders: dict[str, ctk.CTkSlider] = {}
        self._rpy_sync = False
        self._rpy_apply_after: str | None = None
        self._replay_after: str | None = None
        self._pp_after: str | None = None
        self._reg_after: str | None = None
        self._reg_req: RegisterRequest | None = None
        self._pp_runner: PickPlaceRunner | None = (
            PickPlaceRunner(controller) if controller is not None else None
        )
        self._user_pp_thread: threading.Thread | None = None
        verts = np.asarray(vertices_m, dtype=float).reshape(-1, 3)
        self._cad_verts_m = verts
        self._cad_mins_mm = verts.min(axis=0) * 1000.0
        self._cad_maxs_mm = verts.max(axis=0) * 1000.0
        self._rng = np.random.default_rng()
        self._collision_warn = False
        self._warn_label: ctk.CTkLabel | None = None
        self._rand_mode: str | None = None
        self._rand_mode_vars: dict[str, object] = {}
        approach_d0, approach_a0 = approach_mm(CAD_YAML)
        self._approach_d_mm = float(approach_d0)
        self._approach_a_mm = float(approach_a0)
        self._drop_xy = drop_xy_mm(CAD_YAML)
        # place 저장/적용 = 베이스 extrinsic (_place_rpy_prev, canonicalize).
        # 화면 숫자 = 물체 축 또는 베이스. 버튼으로 표기만 바꾼다.
        self._place_rpy_frame = "body"
        self._place_rpy_prev = canonicalize_extrinsic_rpy(place.rpy_deg)
        shown = self._place_rpy_shown()
        self._place_slider_cmd = dict(shown)

        if not self._embedded:
            root.title("물체 집기 티칭")
            root.minsize(980, 720)
            root.geometry("1080x860")
            root.protocol("WM_DELETE_WINDOW", self.on_close)
            split = ctk.CTkFrame(root, fg_color="transparent")
            split.pack(fill="both", expand=True)
            host = ctk.CTkFrame(split)
            actions_host = ctk.CTkFrame(split)
            host.pack(side="left", fill="both", expand=True, padx=(8, 4), pady=8)
            actions_host.pack(side="left", fill="y", padx=(4, 8), pady=8)
        else:
            host = root
            actions_host = actions_parent if actions_parent is not None else root
        self._host = host
        self._actions_host = actions_host

        # TCP 칸과 같은 시작 높이·같은 줄 간격. 박스는 TCP 행과 같은 CTkFrame.
        title = ctk.CTkLabel(host, text="사물의 위치", anchor="w")
        title.pack(anchor="w", padx=8, pady=(8, 4))
        self._panel_labels.append(title)
        self._place_xyz_seg = self._xyz_jog_frame_bar(host)
        for key, label, val in (
            ("place_x", "x (mm)", place.xyz_mm[0]),
            ("place_y", "y (mm)", place.xyz_mm[1]),
            ("place_z", "z (mm)", place.xyz_mm[2]),
        ):
            self._xyz_row(host, key, float(val), label)

        rpy_head = ctk.CTkFrame(host, fg_color="transparent")
        rpy_head.pack(fill="x", padx=6, pady=(10, 2))
        rpy_title = ctk.CTkLabel(rpy_head, text="Rx/Ry/Rz", width=200, anchor="w")
        rpy_title.pack(side="left", padx=(4, 2))
        self._panel_labels.append(rpy_title)
        self._place_rpy_seg = ctk.CTkSegmentedButton(
            rpy_head,
            values=["base", "OBJ"],
            command=self._on_place_rpy_frame,
            width=148,
        )
        self._place_rpy_seg.set("OBJ")
        self._place_rpy_seg.pack(side="left", padx=(8, 4))
        for key, label, axis in (
            ("place_roll", "Rx", "roll"),
            ("place_pitch", "Ry", "pitch"),
            ("place_yaw", "Rz", "yaw"),
        ):
            self._rpy_row(host, key, float(shown[axis]), label)
        self._rpy_hint = ctk.CTkLabel(
            host,
            text="",
            text_color="#9ca3af",
            anchor="w",
        )
        self._rpy_hint.pack(fill="x", padx=8, pady=(0, 2))
        self._update_rpy_hint()

        hide_row = ctk.CTkFrame(host, fg_color="transparent")
        hide_row.pack(fill="x", padx=8, pady=(2, 2))
        self._grasp_hide_var = ctk.BooleanVar(value=False)
        self._grasp_hide_cb = ctk.CTkCheckBox(
            hide_row,
            text="비활성화 (축 숨김)",
            variable=self._grasp_hide_var,
            command=self._on_grasp_hide_toggle,
            width=160,
        )
        self._grasp_hide_cb.pack(side="left")
        self._region_hide_var = ctk.BooleanVar(value=False)
        self._region_hide_cb = ctk.CTkCheckBox(
            hide_row,
            text="비활성화 (영역 숨김)",
            variable=self._region_hide_var,
            command=self._on_region_hide_toggle,
            width=170,
        )
        self._region_hide_cb.pack(side="left", padx=(12, 0))

        self._panel_labels.append(self._rpy_hint)

        actions = ctk.CTkFrame(actions_host, fg_color="transparent")
        actions.pack(fill="x", padx=14, pady=(4, 8))
        actions.grid_columnconfigure(0, weight=1)
        actions.grid_columnconfigure(1, weight=1)
        btn_apply = ctk.CTkButton(actions, text="적용", command=self.apply)
        btn_apply.grid(row=0, column=0, sticky="ew", padx=(0, 4), pady=(0, 6))
        btn_reset = ctk.CTkButton(actions, text="물체 초기화", command=self.reset_place)
        btn_reset.grid(row=0, column=1, sticky="ew", padx=(4, 0), pady=(0, 6))
        self._panel_buttons.extend((btn_apply, btn_reset))
        self._btn_pre = ctk.CTkButton(actions, text="대기 위치로", command=self.goto_pre)
        self._btn_grasp = ctk.CTkButton(actions, text="집기 위치로", command=self.goto_grasp)
        self._btn_pre.grid(row=1, column=0, sticky="ew", padx=(0, 4), pady=(0, 6))
        self._btn_grasp.grid(row=1, column=1, sticky="ew", padx=(4, 0), pady=(0, 6))
        self._btn_seq = ctk.CTkButton(
            actions,
            text="픽앤플레이스",
            command=self.start_pick_place,
            height=56,
            corner_radius=12,
            font=ui_font(18),
        )
        self._btn_seq.grid(row=2, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self._btn_user_seq = ctk.CTkButton(
            actions,
            text="픽앤플레이스(유저생성)",
            command=self.start_user_pick_place,
            height=56,
            corner_radius=12,
            font=ui_font(18),
        )
        self._btn_user_seq.grid(row=3, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        self._btn_init = ctk.CTkButton(
            actions, text="초기자세로 이동", command=self.snap_init_pose
        )
        self._btn_init.grid(row=4, column=0, columnspan=2, sticky="ew")
        self._panel_buttons.extend(
            (
                self._btn_pre,
                self._btn_grasp,
                self._btn_seq,
                self._btn_user_seq,
                self._btn_init,
            )
        )
        # 픽앤플레이스: motion/pick_place.py 시퀀스.
        if self._ctrl is None:
            self._btn_pre.configure(state="disabled")
            self._btn_grasp.configure(state="disabled")
            self._btn_seq.configure(state="disabled")
            self._btn_user_seq.configure(state="disabled")
            self._btn_init.configure(state="disabled")

        rand_row = ctk.CTkFrame(actions_host, fg_color="transparent")
        rand_row.pack(fill="x", padx=14, pady=(0, 2))
        btn_rand = ctk.CTkButton(
            rand_row,
            text="물체 랜덤 생성",
            width=140,
            command=self.randomize_place,
        )
        btn_rand.pack(side="left")
        self._panel_buttons.append(btn_rand)
        self._warn_label = ctk.CTkLabel(
            rand_row,
            text="",
            anchor="w",
            text_color="#f87171",
        )
        self._warn_label.pack(side="left", padx=(12, 0), fill="x", expand=True)

        mode_row = ctk.CTkFrame(actions_host, fg_color="transparent")
        mode_row.pack(fill="x", padx=14, pady=(0, 6))
        mode_specs = (
            ("stand", "세우기"),
            ("lie", "눕히기"),
            ("slant", "비스듬히"),
            ("free", "자유 포즈"),
        )
        for key, label in mode_specs:
            var = ctk.BooleanVar(value=False)
            cb = ctk.CTkCheckBox(
                mode_row,
                text=label,
                variable=var,
                command=lambda k=key: self._on_rand_mode_toggle(k),
                checkbox_width=18,
                checkbox_height=18,
            )
            cb.pack(side="left", padx=(0, 8))
            self._rand_mode_vars[key] = var
            self._mode_cbs.append(cb)

        foot = ctk.CTkFrame(actions_host, fg_color="transparent")
        foot.pack(fill="x", padx=14, pady=(0, 4))
        self._row(foot, "approach_d", f"{approach_d0:.1f}", "d mm (pre)")
        self._row(foot, "approach_a", f"{approach_a0:.1f}", "a mm (grasp)")
        self.entries["approach_d"].bind("<FocusOut>", lambda _e: self._schedule_live_apply())
        self.entries["approach_d"].bind("<Return>", lambda _e: self._schedule_live_apply())
        self.entries["approach_a"].bind("<FocusOut>", lambda _e: self._schedule_live_apply())
        self.entries["approach_a"].bind("<Return>", lambda _e: self._schedule_live_apply())

        self._status = ctk.CTkLabel(actions_host, text="", anchor="w")
        self._status.pack(fill="x", padx=14, pady=(0, 6))
        btn_load = ctk.CTkButton(
            actions_host,
            text="물체 위치 불러오기",
            command=self.load_registered_place,
            fg_color="#86EFAC",
            hover_color="#4ADE80",
            text_color="#14532D",
        )
        btn_load.pack(fill="x", padx=14, pady=(0, 14))
        self._panel_buttons.append(btn_load)

        if self._embedded and not self._panel_enabled:
            self._lock_place_panel()
            return

        if not self._embedded:
            root.bind("<Return>", lambda _e: self.apply())
        self._draw_static(vertices_m, faces)
        self.apply()
        # Color pre sphere while TCP moves; pendant owns display(q) when not drive_viz.
        if self._ctrl is not None:
            self._schedule_display()

    def _xyz_jog_frame_bar(self, parent):
        """TCP 패널의 XYZ jog frame 과 같은 자리. 숫자는 그대로 둔다."""
        import customtkinter as ctk

        bar = ctk.CTkFrame(parent, fg_color="transparent")
        bar.pack(fill="x", padx=6, pady=(0, 2))
        lab = ctk.CTkLabel(bar, text="XYZ jog frame", width=200, anchor="w")
        lab.pack(side="left", padx=(4, 2))
        self._panel_labels.append(lab)
        seg = ctk.CTkSegmentedButton(
            bar,
            values=["base", "TCP"],
            command=self._on_place_xyz_frame,
            width=148,
        )
        seg.set("base")
        seg.pack(side="left", padx=(8, 4))
        return seg

    def _on_place_xyz_frame(self, value: str) -> None:
        if self._embedded and not self._panel_enabled:
            return
        self._place_xyz_frame = "tcp" if str(value) == "TCP" else "base"

    def _axis_row(self, parent):
        """TCP `_cart_row` 와 같은 프레임. 높이는 조그 버튼(32)에 맞춘다."""
        import customtkinter as ctk

        row = ctk.CTkFrame(parent)
        row.pack(fill="x", pady=3, padx=6)
        return row

    def _section(self, parent, title: str) -> None:
        import customtkinter as ctk

        lab = ctk.CTkLabel(parent, text=title, anchor="w", font=ui_font(15, "bold"))
        lab.pack(fill="x", padx=8, pady=(8, 2))
        self._panel_labels.append(lab)

    def _grasp_section_hidden(self) -> bool:
        var = getattr(self, "_grasp_hide_var", None)
        return bool(var.get()) if var is not None else False

    def _on_grasp_hide_toggle(self) -> None:
        self.apply()

    def _region_hidden(self) -> bool:
        var = getattr(self, "_region_hide_var", None)
        return bool(var.get()) if var is not None else False

    def _on_region_hide_toggle(self) -> None:
        if self._region_hidden():
            self._hide_region_overlays()
        else:
            self._show_region_overlays()

    def _hide_grasp_overlays(self) -> None:
        for name in ("grasp", "grasp_marker", "approach"):
            self._viz.clear_overlay(name)

    def _hide_region_overlays(self) -> None:
        for name in ("place_region", "place_region_edge"):
            self._viz.clear_overlay(name)

    def _show_region_overlays(self) -> None:
        region_v, region_f, region_loop = place_region_overlay_m()
        self._viz.set_overlay_mesh(
            "place_region",
            region_v,
            region_f,
            color=0x86EFAC,
            opacity=0.28,
        )
        self._viz.set_overlay_lines(
            "place_region_edge",
            region_loop,
            color=0x4ADE80,
            closed=True,
        )

    def _xyzrpy_block(
        self,
        parent,
        prefix: str,
        xyz: tuple[float, float, float],
        rpy: tuple[float, float, float],
    ) -> None:
        for key, label, val in (
            (f"{prefix}_x", "x mm", xyz[0]),
            (f"{prefix}_y", "y mm", xyz[1]),
            (f"{prefix}_z", "z mm", xyz[2]),
        ):
            if prefix == "place":
                self._xyz_row(parent, key, float(val), label)
            else:
                self._row(parent, key, f"{val:.1f}", label)
        for key, label, val in (
            (f"{prefix}_roll", "Rx deg", rpy[0]),
            (f"{prefix}_pitch", "Ry deg", rpy[1]),
            (f"{prefix}_yaw", "Rz deg", rpy[2]),
        ):
            self._rpy_row(parent, key, float(val), label)

    def _row(self, parent, key: str, value: str, label: str) -> None:
        import customtkinter as ctk

        row = ctk.CTkFrame(parent)
        row.pack(fill="x", padx=8, pady=2)
        lab = ctk.CTkLabel(row, text=label, width=110, anchor="w")
        lab.pack(side="left")
        self._panel_labels.append(lab)
        ent = ctk.CTkEntry(row, width=140)
        ent.insert(0, value)
        ent.pack(side="left", padx=6)
        self.entries[key] = ent

    def _slider_limits(self, key: str) -> tuple[float, float]:
        if key in ("place_x", "place_y"):
            return PLACE_XY_SLIDER_MIN, PLACE_XY_SLIDER_MAX
        if key == "place_z":
            return PLACE_Z_SLIDER_MIN, PLACE_Z_SLIDER_MAX
        return RPY_SLIDER_MIN, RPY_SLIDER_MAX

    def _xyz_row(self, parent, key: str, value: float, label: str) -> None:
        import customtkinter as ctk

        lo, hi = self._slider_limits(key)
        row = self._axis_row(parent)
        lab = ctk.CTkLabel(row, text=label, width=110, anchor="w")
        lab.pack(side="left")
        self._panel_labels.append(lab)
        ent = ctk.CTkEntry(row, width=72, height=32)
        ent.insert(0, f"{value:.1f}")
        ent.pack(side="left", padx=(6, 4))
        steps = max(int(round(hi - lo)), 1)
        slider = ctk.CTkSlider(
            row,
            from_=lo,
            to=hi,
            number_of_steps=steps,
            command=lambda v, k=key: self._on_xyz_slider(k, v),
        )
        self._rpy_sync = True
        try:
            slider.set(float(np.clip(value, lo, hi)))
        finally:
            self._rpy_sync = False
        slider.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.entries[key] = ent
        self.sliders[key] = slider
        ent.bind("<FocusOut>", lambda _e, k=key: self._on_xyz_entry(k))
        ent.bind("<Return>", lambda _e, k=key: self._on_xyz_entry(k))

    def _rpy_row(self, parent, key: str, value: float, label: str) -> None:
        import customtkinter as ctk

        row = self._axis_row(parent)
        lab = ctk.CTkLabel(row, text=label, width=110, anchor="w")
        lab.pack(side="left")
        self._panel_labels.append(lab)
        ent = ctk.CTkEntry(row, width=72, height=32)
        ent.insert(0, f"{value:.1f}")
        ent.pack(side="left", padx=(6, 4))
        slider = ctk.CTkSlider(
            row,
            from_=RPY_SLIDER_MIN,
            to=RPY_SLIDER_MAX,
            number_of_steps=360,
            command=lambda v, k=key: self._on_rpy_slider(k, v),
        )
        self._rpy_sync = True
        try:
            slider.set(float(np.clip(value, RPY_SLIDER_MIN, RPY_SLIDER_MAX)))
        finally:
            self._rpy_sync = False
        slider.pack(side="left", fill="x", expand=True, padx=(0, 4))
        self.entries[key] = ent
        self.sliders[key] = slider
        ent.bind("<FocusOut>", lambda _e, k=key: self._on_rpy_entry(k))
        ent.bind("<Return>", lambda _e, k=key: self._on_rpy_entry(k))

    def _on_xyz_slider(self, key: str, value: float) -> None:
        if self._rpy_sync:
            return
        self._rpy_sync = True
        try:
            ent = self.entries[key]
            ent.delete(0, "end")
            ent.insert(0, f"{float(value):.1f}")
        finally:
            self._rpy_sync = False
        self._schedule_live_apply()

    def _on_xyz_entry(self, key: str) -> None:
        if self._rpy_sync or key not in self.sliders:
            return
        try:
            val = float(self.entries[key].get().strip())
        except ValueError:
            return
        lo, hi = self._slider_limits(key)
        val = float(np.clip(val, lo, hi))
        self._rpy_sync = True
        try:
            self.sliders[key].set(val)
            self.entries[key].delete(0, "end")
            self.entries[key].insert(0, f"{val:.1f}")
        finally:
            self._rpy_sync = False
        self._schedule_live_apply()

    def _place_rpy_shown(self) -> dict[str, float]:
        """현재 버튼 표기로 푼 Rx/Ry/Rz."""
        if self._place_rpy_frame == "base":
            rpy = self._place_rpy_prev
            return {"roll": float(rpy[0]), "pitch": float(rpy[1]), "yaw": float(rpy[2])}
        roll, pitch, yaw = rpy_extrinsic_to_body_xyz(self._place_rpy_prev)
        return {"roll": float(roll), "pitch": float(pitch), "yaw": float(yaw)}

    def _update_rpy_hint(self) -> None:
        if self._place_rpy_frame == "base":
            text = "슬라이더 = 베이스 고정축 회전. 숫자는 저장되는 베이스 각."
        else:
            text = "슬라이더 = 물체에 붙은 축 회전. Rz만 돌리면 Rz 숫자만 바뀌는 건 정상."
        self._rpy_hint.configure(text=text)

    def _on_place_rpy_frame(self, value: str) -> None:
        if self._embedded and not self._panel_enabled:
            return
        frame = "base" if str(value) == "base" else "body"
        if frame == self._place_rpy_frame:
            return
        self._place_rpy_frame = frame
        self._sync_place_rpy_ui()
        self._update_rpy_hint()

    def _sync_place_rpy_ui(self, *, keep_axis: str | None = None) -> None:
        """보이는 칸을 현재 표기로 맞춘다. 드래그 중인 축 슬라이더는 유지."""
        extracted = self._place_rpy_shown()
        self._rpy_sync = True
        try:
            for axis, val in extracted.items():
                key = f"place_{axis}"
                if key in self.entries:
                    self.entries[key].delete(0, "end")
                    self.entries[key].insert(0, f"{val:.1f}")
                if axis == keep_axis:
                    continue
                self._place_slider_cmd[axis] = val
                if key in self.sliders:
                    lo, hi = self._slider_limits(key)
                    self.sliders[key].set(float(np.clip(val, lo, hi)))
        finally:
            self._rpy_sync = False

    def _nudge_place_rpy(
        self, axis: str, new_axis_val: float, *, absolute: bool = False
    ) -> None:
        """Rx/Ry/Rz 조작. xyz(물체 원점)는 건드리지 않는다. 저장은 베이스 각."""
        axis = {"roll": "roll", "pitch": "pitch", "yaw": "yaw"}[axis]
        new_axis_val = float(new_axis_val)
        if absolute:
            cmd = dict(self._place_slider_cmd)
            cmd[axis] = new_axis_val
            if self._place_rpy_frame == "base":
                self._place_rpy_prev = canonicalize_extrinsic_rpy(
                    (float(cmd["roll"]), float(cmd["pitch"]), float(cmd["yaw"]))
                )
            else:
                self._place_rpy_prev = rpy_body_xyz_to_extrinsic(
                    float(cmd["roll"]), float(cmd["pitch"]), float(cmd["yaw"])
                )
            self._place_slider_cmd = cmd
            self._sync_place_rpy_ui()
        else:
            old = float(self._place_slider_cmd[axis])
            delta = new_axis_val - old
            if delta > 180.0:
                delta -= 360.0
            elif delta < -180.0:
                delta += 360.0
            if self._place_rpy_frame == "base":
                self._place_rpy_prev = place_rpy_base_delta(
                    self._place_rpy_prev, axis=axis, delta_deg=delta
                )
            else:
                self._place_rpy_prev = place_rpy_body_delta(
                    self._place_rpy_prev, axis=axis, delta_deg=delta
                )
            self._place_slider_cmd[axis] = new_axis_val
            self._sync_place_rpy_ui(keep_axis=axis)
        self._schedule_live_apply()

    def _on_rpy_slider(self, key: str, value: float) -> None:
        if self._rpy_sync:
            return
        if key.startswith("place_"):
            self._nudge_place_rpy(
                key.removeprefix("place_"), float(value), absolute=False
            )
            return
        self._rpy_sync = True
        try:
            ent = self.entries[key]
            ent.delete(0, "end")
            ent.insert(0, f"{float(value):.1f}")
        finally:
            self._rpy_sync = False
        self._schedule_live_apply()

    def _on_rpy_entry(self, key: str) -> None:
        if self._rpy_sync or key not in self.sliders:
            return
        try:
            val = float(self.entries[key].get().strip())
        except ValueError:
            return
        val = float(np.clip(val, RPY_SLIDER_MIN, RPY_SLIDER_MAX))
        if key.startswith("place_"):
            self._nudge_place_rpy(
                key.removeprefix("place_"), val, absolute=True
            )
            return
        self._rpy_sync = True
        try:
            self.sliders[key].set(val)
            self.entries[key].delete(0, "end")
            self.entries[key].insert(0, f"{val:.1f}")
        finally:
            self._rpy_sync = False
        self._schedule_live_apply()

    def _schedule_live_apply(self) -> None:
        if self._rpy_apply_after is not None:
            try:
                self.root.after_cancel(self._rpy_apply_after)
            except Exception:
                pass
        self._rpy_apply_after = self.root.after(RPY_LIVE_APPLY_MS, self._live_apply)

    def _live_apply(self) -> None:
        self._rpy_apply_after = None
        try:
            self.apply()
        except Exception:
            pass

    def _set_status(self, text: str, *, error: bool = False, info: bool = False) -> None:
        if self._status is None:
            return
        if error:
            color = "#f87171"
        elif info:
            color = "#e5e7eb"
        else:
            color = "#86efac"
        self._status.configure(text=text, text_color=color)

    def _read_float(self, key: str) -> float:
        return float(self.entries[key].get().strip())

    def _read_place_grasp(self) -> tuple[PlacePose, GraspSpec]:
        place = PlacePose(
            xyz_mm=(
                self._read_float("place_x"),
                self._read_float("place_y"),
                self._read_float("place_z"),
            ),
            rpy_deg=tuple(float(x) for x in self._place_rpy_prev),
        )
        return place, self._grasp_spec

    def _fill(self, place: PlacePose, grasp: GraspSpec) -> None:
        self._fill_place(place)
        self._fill_grasp(grasp)

    def _set_entry(self, key: str, val: float) -> None:
        self.entries[key].delete(0, "end")
        self.entries[key].insert(0, f"{val:.1f}")
        if key in self.sliders:
            lo, hi = self._slider_limits(key)
            clipped = float(np.clip(val, lo, hi))
            self._rpy_sync = True
            try:
                self.sliders[key].set(clipped)
            finally:
                self._rpy_sync = False

    def _fill_place(self, place: PlacePose) -> None:
        # 내부=베이스 extrinsic. 칸 숫자는 현재 물체 축/베이스 표기.
        self._place_rpy_prev = canonicalize_extrinsic_rpy(place.rpy_deg)
        shown = self._place_rpy_shown()
        self._place_slider_cmd = dict(shown)
        mapping = {
            "place_x": place.xyz_mm[0],
            "place_y": place.xyz_mm[1],
            "place_z": place.xyz_mm[2],
            "place_roll": shown["roll"],
            "place_pitch": shown["pitch"],
            "place_yaw": shown["yaw"],
        }
        for key, val in mapping.items():
            self._set_entry(key, val)

    def _fill_grasp(self, grasp: GraspSpec) -> None:
        self._grasp_spec = grasp

    def _draw_static(self, vertices_m: np.ndarray, faces: np.ndarray) -> None:
        table_t = 0.004
        T_table = np.eye(4)
        T_table[2, 3] = -table_t / 2.0
        self._viz.set_overlay_box(
            "table",
            (0.80, 0.80, table_t),
            T_table,
            color=0x2F2F35,
            opacity=0.45,
        )
        if not self._region_hidden():
            self._show_region_overlays()
        self._viz.set_overlay_mesh("cad", vertices_m, faces, color=0xC4C4C8, opacity=0.92)
        self._cad_loaded = True

    def _show_robot(self, gripper_100: float) -> None:
        if self._kin is None:
            return
        from motion.hw_controller import grip_100_to_user
        from motion.robot_kinematics import GRIPPER_JOINT, HOME_JOINTS_DEG, URDF_JOINT_NAMES

        joints = {name: float(HOME_JOINTS_DEG[name]) for name in URDF_JOINT_NAMES}
        joints[GRIPPER_JOINT] = grip_100_to_user(gripper_100)
        self._viz.display(self._kin.q_from_deg(joints))

    def _lock_place_panel(self) -> None:
        gray = "#6b7280"
        for lab in self._panel_labels:
            try:
                lab.configure(text_color=gray)
            except Exception:
                pass
        for ent in self.entries.values():
            ent.configure(text_color=gray, state="disabled")
        for slider in self.sliders.values():
            slider.configure(state="disabled")
        self._place_rpy_seg.configure(state="disabled")
        self._place_xyz_seg.configure(state="disabled")
        for cb in (self._grasp_hide_cb, self._region_hide_cb, *self._mode_cbs):
            cb.configure(state="disabled", text_color_disabled=gray)
        for btn in self._panel_buttons:
            btn.configure(state="disabled")

    def _schedule_display(self) -> None:
        if self._ctrl is None:
            return
        try:
            if not int(self.root.winfo_exists()):
                return
        except Exception:
            return
        st = self._ctrl.snapshot()
        if self._drive_viz:
            self._viz.display(st.q)
        self._update_pre_match_color(st)
        self.root.after(DISPLAY_MS, self._schedule_display)

    def _tcp_matches_pre(self, st) -> bool:
        """True if TCP is on the orange pre sphere (project-base position)."""
        auto = self._auto_pre
        if auto is None:
            return False
        err_p = float(
            np.linalg.norm(np.asarray(st.pose.xyz_m, dtype=float) - auto.T_base[:3, 3])
        )
        return err_p * 1000.0 < PRE_TCP_MATCH_POS_MM

    def _draw_approach_markers(self, *, matched: bool) -> None:
        auto = self._auto_pre
        if auto is None:
            return
        T_p = _T_meshcat(auto.T_base)
        centers = [T_p[:3, 3]]
        colors = [APPROACH_PRE_MATCH_COLOR if matched else APPROACH_PRE_COLOR]
        if auto.other_pre_base is not None:
            T_other = np.eye(4)
            T_other[:3, 3] = np.asarray(auto.other_pre_base, dtype=float)
            centers.append(_T_meshcat(T_other)[:3, 3])
            colors.append(APPROACH_OTHER_COLOR)
        self._viz.clear_overlay("approach_markers")
        self._viz.set_overlay_spheres(
            "approach_markers",
            np.stack(centers, axis=0),
            radius_m=APPROACH_MARKER_RADIUS_M,
            colors=colors,
        )

    def _update_pre_match_color(self, st) -> None:
        if self._auto_pre is None:
            return
        matched = self._tcp_matches_pre(st)
        if matched is self._pre_tcp_matched:
            return
        self._pre_tcp_matched = matched
        self._draw_approach_markers(matched=matched)

    def _read_da(self) -> tuple[float, float]:
        try:
            d_mm = self._read_float("approach_d")
        except (KeyError, ValueError):
            d_mm = self._approach_d_mm
        try:
            a_mm = self._read_float("approach_a")
        except (KeyError, ValueError):
            a_mm = self._approach_a_mm
        return float(d_mm), float(a_mm)

    def _compute_auto(self, place: PlacePose) -> AutoPreResult:
        d_mm, a_mm = self._read_da()
        kin = self._kin
        if kin is None and self._ctrl is not None:
            kin = getattr(self._ctrl, "_kin", None)
        q = None
        if self._ctrl is not None:
            q = self._ctrl.snapshot().q
        elif kin is not None:
            q = kin.q_home()
        return compute_auto_pre(
            place,
            self._cad_mins_mm,
            self._cad_maxs_mm,
            d_mm,
            a_mm,
            kin=kin,
            q=q,
        )

    def _cancel_register_poll(self) -> None:
        if self._reg_after is None:
            return
        try:
            self.root.after_cancel(self._reg_after)
        except Exception:
            pass
        self._reg_after = None

    def load_registered_place(self) -> None:
        """카메라 창에 c 와 같은 전체 등록을 요청하고, 끝난 자세만 칸에 넣는다."""
        if self._embedded and not self._panel_enabled:
            return
        self._cancel_register_poll()
        try:
            self._reg_req = RegisterRequest.send()
        except OSError:
            self._reg_req = None
            self._set_status(REG_MSG_NO_REPLY, error=True)
            return
        self._set_status(REG_MSG_RUN, info=True)
        self._reg_after = self.root.after(REG_POLL_MS, self._poll_register)

    def _poll_register(self) -> None:
        self._reg_after = None
        try:
            if not int(self.root.winfo_exists()):
                return
        except Exception:
            return
        req = self._reg_req
        if req is None:
            return
        state = req.poll()
        if state == "ok":
            place = PlacePose(xyz_mm=req.xyz_mm, rpy_deg=req.rpy_deg)
            self._fill_place(place)
            self.apply()
            self._set_status(
                f"불러옴  xyz=({place.xyz_mm[0]:.1f}, {place.xyz_mm[1]:.1f}, {place.xyz_mm[2]:.1f})"
            )
            return
        if state in ("run", "busy"):
            self._set_status(REG_MSG_RUN, info=True)
        elif state == "no_box":
            self._set_status(REG_MSG_NO_BOX, error=True)
        elif state == "fail":
            self._set_status(REG_MSG_FAIL, error=True)
        elif state == "timeout":
            self._set_status(REG_MSG_NO_REPLY, error=True)
        if not req.done:
            self._reg_after = self.root.after(REG_POLL_MS, self._poll_register)

    def apply(self) -> None:
        if self._embedded and not self._panel_enabled:
            return
        try:
            place, grasp = self._read_place_grasp()
        except ValueError as exc:
            self._set_status(f"입력 오류: {exc}", error=True)
            return
        d_mm, a_mm = self._read_da()
        auto = self._compute_auto(place)
        self._auto_pre = auto
        T_base_cad = pose_to_T(np.array(place.xyz_mm), np.array(place.rpy_deg))
        T_cad = _T_meshcat(T_base_cad)
        T_p = _T_meshcat(auto.T_base)
        T_g = _T_meshcat(auto.T_grasp_base)

        if self._cad_loaded:
            self._viz.set_overlay_transform("cad", T_cad)
        self._viz.set_overlay_axes("cad_axes", T_cad, scale=0.02, labels=False)
        self._viz.clear_overlay("base_at_obj")
        self._viz.set_overlay_axes("pregrasp", T_p, scale=0.02)
        # Virtual N axis (x=z, y=0) only while that approach is selected.
        if auto.axis_name == "N":
            n_world = T_cad[:3, :3] @ N_CAD
            o = T_cad[:3, 3]
            half = max(float(d_mm), 20.0) / 1000.0
            self._viz.set_overlay_segment(
                "n_axis",
                o - n_world * half,
                o + n_world * half,
                color=0xC026D3,
            )
        else:
            self._viz.clear_overlay("n_axis")
        matched = False
        if self._ctrl is not None:
            matched = self._tcp_matches_pre(self._ctrl.snapshot())
        self._pre_tcp_matched = matched
        self._draw_approach_markers(matched=matched)
        if self._grasp_section_hidden():
            self._hide_grasp_overlays()
        else:
            self._viz.set_overlay_axes("grasp", T_g, scale=0.0225)
            self._viz.set_overlay_segment("approach", T_p[:3, 3], T_g[:3, 3])
            self._viz.set_overlay_spheres(
                "grasp_marker",
                T_g[:3, 3].reshape(1, 3),
                radius_m=APPROACH_MARKER_RADIUS_M,
                color=0x22C55E,
            )
        if self._own_robot:
            self._show_robot(grasp.gripper)

    def _set_collision_warn(self, text: str) -> None:
        self._collision_warn = bool(text)
        if self._warn_label is not None:
            self._warn_label.configure(text=text)

    def _on_rand_mode_toggle(self, key: str) -> None:
        """One mode at a time; others stay clickable (no lock). Click again to clear."""
        var = self._rand_mode_vars[key]
        selected = bool(var.get())
        if selected:
            self._rand_mode = key
            for k, other in self._rand_mode_vars.items():
                if k != key:
                    other.set(False)
        else:
            self._rand_mode = None

    def _active_rand_mode(self) -> str:
        """Unset checkbox → free (same as 자유 포즈)."""
        m = self._rand_mode
        return m if m in RANDOM_PLACE_MODES else "free"

    def randomize_place(self) -> None:
        """Random place by selected mode (default/unset = free). Floor-safe."""
        self._set_collision_warn("")
        mode = self._active_rand_mode()
        place = sample_random_place(self._cad_verts_m, mode=mode, rng=self._rng)
        if place is None:
            self._set_status(
                f"랜덤 실패({mode}): {RANDOM_PLACE_MAX_TRIES}회 내 바닥/영역 만족 샘플 없음",
                error=True,
            )
            return
        self._fill_place(place)
        self.apply()

        kin = self._kin
        if kin is None and self._ctrl is not None:
            kin = getattr(self._ctrl, "_kin", None)
        q = None
        if self._ctrl is not None:
            q = self._ctrl.snapshot().q
        elif kin is not None:
            q = kin.q_home()

        collided = False
        if kin is not None and q is not None:
            try:
                collided = object_collides_robot(kin, q, self._cad_verts_m, place)
            except Exception as exc:
                self._set_status(
                    f"랜덤({mode}) xyz={list(np.round(place.xyz_mm, 1))}  "
                    f"rpy={list(np.round(place.rpy_deg, 1))}  "
                    f"(충돌검사 실패: {exc})",
                    error=True,
                )
                return

        mode_ko = {
            "stand": "세우기",
            "lie": "눕히기",
            "slant": "비스듬히",
            "free": "자유",
        }.get(mode, mode)
        if collided:
            self._set_collision_warn("경고: 물체가 현재 로봇과 충돌합니다. 다시 랜덤 생성하세요.")
            self._set_status(
                f"랜덤({mode_ko}/충돌) xyz={list(np.round(place.xyz_mm, 1))}  "
                f"rpy={list(np.round(place.rpy_deg, 1))}",
                error=True,
            )
        else:
            self._set_status(
                f"랜덤({mode_ko}) xyz={list(np.round(place.xyz_mm, 1))}  "
                f"rpy={list(np.round(place.rpy_deg, 1))}"
            )

    def reset_place(self) -> None:
        place, _ = load_specs(self._yaml_path)
        self._fill_place(place)
        self.apply()
        self._set_status(f"물체 초기화: {self._yaml_path.name} place")

    def reset_grasp(self) -> None:
        _, grasp = load_specs(self._yaml_path)
        self._fill_grasp(grasp)
        self.apply()
        self._set_status(f"grasp 초기화: {self._yaml_path.name} grasp")

    def _goto_T(self, T: np.ndarray, *, label: str) -> float:
        if self._ctrl is None:
            self._set_status("컨트롤러 없음. python pendant/main.py --grasp", error=True)
            return 0.0
        self.apply()
        xyz_mm, rpy_deg = T_to_xyzrpy(T)
        duration_s = self._ctrl.start_ee_goto(xyz_mm, rpy_deg)
        self._set_status(
            f"{label} 이동  xyz=[{xyz_mm[0]:.1f}, {xyz_mm[1]:.1f}, {xyz_mm[2]:.1f}]  "
            f"rpy=[{rpy_deg[0]:.1f}, {rpy_deg[1]:.1f}, {rpy_deg[2]:.1f}]  "
            f"({duration_s:.1f}s)"
        )
        return float(duration_s)

    def goto_pre(self) -> None:
        try:
            place, _grasp = self._read_place_grasp()
        except ValueError as exc:
            self._set_status(f"입력 오류: {exc}", error=True)
            return
        auto = self._compute_auto(place)
        self._auto_pre = auto
        flip_txt = "pitch−" if auto.pitch_flipped else "pitch+"
        form_txt = f" {auto.form}" if auto.form else ""
        self._goto_T(auto.T_base, label=f"대기 위치 ({auto.axis_name}{form_txt} {flip_txt})")

    def goto_grasp(self) -> None:
        try:
            place, _grasp = self._read_place_grasp()
        except ValueError as exc:
            self._set_status(f"입력 오류: {exc}", error=True)
            return
        auto = self._compute_auto(place)
        self._auto_pre = auto
        flip_txt = "pitch−" if auto.pitch_flipped else "pitch+"
        form_txt = f" {auto.form}" if auto.form else ""
        self._goto_T(
            auto.T_grasp_base,
            label=f"집기 위치 ({auto.axis_name}{form_txt} a={auto.a_mm:.0f} {flip_txt})",
        )

    def replay_pg(self) -> None:
        self.start_pick_place()

    def start_pick_place(self) -> None:
        if self._ctrl is None or self._pp_runner is None:
            self._set_status("컨트롤러 없음. python pendant/main.py --grasp", error=True)
            return
        if self._user_pick_running():
            self._set_status("픽앤플레이스(유저생성) 실행 중", error=True)
            return
        if self._pp_runner.busy():
            self._set_status("픽앤플레이스 이미 실행 중", error=True)
            return
        try:
            place, _grasp = self._read_place_grasp()
            drop_xy = self._drop_xy
        except ValueError as exc:
            self._set_status(f"입력 오류: {exc}", error=True)
            return
        auto = self._compute_auto(place)
        self._auto_pre = auto
        p_xyz, p_rpy = T_to_xyzrpy(auto.T_base)
        g_xyz, g_rpy = T_to_xyzrpy(auto.T_grasp_base)
        steps = build_pick_place_steps(
            p_xyz_mm=p_xyz,
            p_rpy_deg=p_rpy,
            g_xyz_mm=g_xyz,
            g_rpy_deg=g_rpy,
            drop_xy_mm=np.array(drop_xy, dtype=float),
        )
        if not self._pp_runner.start(steps):
            self._set_status(self._pp_runner.status, error=True)
            return
        self._set_status(f"픽앤플레이스  {self._pp_runner.step_name}")
        self._tick_pick_place()

    def _tick_pick_place(self) -> None:
        self._pp_after = None
        runner = self._pp_runner
        if runner is None:
            return
        phase = runner.tick()
        if phase == "running":
            self._set_status(f"픽앤플레이스  {runner.step_name}")
            self._pp_after = self.root.after(50, self._tick_pick_place)
            return
        if phase == "done":
            self._set_status("픽앤플레이스 완료")
            return
        if phase == "fault":
            self._set_status(f"픽앤플레이스 실패: {runner.status}", error=True)

    def _user_pick_running(self) -> bool:
        thread = self._user_pp_thread
        return thread is not None and thread.is_alive()

    def start_user_pick_place(self) -> None:
        if self._ctrl is None:
            self._set_status("컨트롤러 없음. python pendant/main.py --grasp", error=True)
            return
        if self._user_pick_running():
            self._set_status("픽앤플레이스(유저생성) 이미 실행 중", error=True)
            return
        if self._pp_runner is not None and self._pp_runner.busy():
            self._set_status("픽앤플레이스 실행 중", error=True)
            return
        self._user_pp_thread = threading.Thread(
            target=self._user_pick_worker, name="user-pick-place", daemon=True
        )
        self._user_pp_thread.start()
        self._set_status("픽앤플레이스(유저생성) 실행")

    def abort_user_pick_place(self) -> None:
        if not self._user_pick_running() or self._ctrl is None:
            return
        self._ctrl.stop_all()

    def _user_pick_worker(self) -> None:
        try:
            run = _load_user_pick_run()
            from motion.arm import Arm

            run(Arm.attach(self._ctrl))
        except Exception as exc:
            self._post_status(f"픽앤플레이스(유저생성) 실패: {exc}", error=True)
        else:
            self._post_status("픽앤플레이스(유저생성) 완료")

    def _post_status(self, text: str, *, error: bool = False) -> None:
        def apply() -> None:
            try:
                self._set_status(text, error=error)
            except Exception:
                pass

        try:
            self.root.after(0, apply)
        except Exception:
            pass

    def snap_init_pose(self) -> None:
        if self._ctrl is None:
            self._set_status("컨트롤러 없음. python pendant/main.py --grasp", error=True)
            return
        how = self._ctrl.snap_init_pose()
        if how == "snap":
            self._set_status("초기자세로 즉시 이동 (가상)")
        elif how == "goto":
            self._set_status("초기자세로 이동 중 (실기 · 관절 go-to)")
        else:
            self._set_status("초기자세 이동 실패 (E-stop/충돌 등)", error=True)

    def on_close(self) -> None:
        self.abort_user_pick_place()
        if self._pp_runner is not None and self._pp_runner.busy():
            self._pp_runner.abort("창 닫힘")
        if self._pp_after is not None:
            try:
                self.root.after_cancel(self._pp_after)
            except Exception:
                pass
            self._pp_after = None
        if self._replay_after is not None:
            try:
                self.root.after_cancel(self._replay_after)
            except Exception:
                pass
            self._replay_after = None
        self._cancel_register_poll()
        self.root.destroy()


def attach_teach_panel(
    parent,
    visualizer,
    *,
    meshcat_url: str,
    controller=None,
    enabled: bool = False,
    actions_parent=None,
) -> TeachGraspGui | None:
    """로봇 펜던트 열에 사물 위치 블록을 붙인다. enabled 가 아니면 표시만 하고 잠근다."""
    try:
        mesh_path = cad_mesh_path()
        vertices_m, faces = load_mesh_m(mesh_path)
    except Exception as exc:
        print(f"사물 위치 패널: CAD 로드 실패 ({exc})")
        if enabled:
            return None
        mesh_path = Path("cad")
        vertices_m = np.zeros((1, 3))
        faces = np.zeros((0, 3), dtype=np.uint32)
    try:
        place, grasp = load_specs(CAD_YAML)
    except Exception as exc:
        print(f"사물 위치 패널: yaml 로드 실패 ({exc})")
        return None
    return TeachGraspGui(
        parent,
        visualizer=visualizer,
        mesh_path=mesh_path,
        vertices_m=vertices_m,
        faces=faces,
        place=place,
        grasp=grasp,
        meshcat_url=meshcat_url,
        kinematics=getattr(controller, "_kin", None) if controller is not None else None,
        controller=controller,
        own_robot=False,
        drive_viz=False,
        embedded=True,
        enabled=enabled,
        actions_parent=actions_parent,
    )


def attach_teach_window(
    parent,
    visualizer,
    *,
    meshcat_url: str,
    controller=None,
    geometry: str = "560x1200+1000+40",
) -> TeachGraspGui | None:
    """펜던트와 같은 Meshcat에 teach 창을 붙인다. 팔 display(q) 는 펜던트가 담당."""
    import customtkinter as ctk

    try:
        mesh_path = cad_mesh_path()
        vertices_m, faces = load_mesh_m(mesh_path)
    except Exception as exc:
        print(f"teach 창 생략: CAD 로드 실패 ({exc})")
        return None
    place, grasp = load_specs(CAD_YAML)
    win = ctk.CTkToplevel(parent)
    win.geometry(geometry)
    win.minsize(520, 1100)
    gui = TeachGraspGui(
        win,
        visualizer=visualizer,
        mesh_path=mesh_path,
        vertices_m=vertices_m,
        faces=faces,
        place=place,
        grasp=grasp,
        meshcat_url=meshcat_url,
        kinematics=getattr(controller, "_kin", None) if controller is not None else None,
        controller=controller,
        own_robot=False,
        drive_viz=False,
    )
    win.geometry(geometry)
    win.after(80, win.lift)
    return gui


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="물체 배치·grasp 값을 Meshcat에서 확인·이동")
    p.add_argument("--no-open", action="store_true", help="Meshcat 브라우저를 자동으로 열지 않음")
    return p.parse_args()


def main() -> None:
    quiet_gtk()
    args = parse_args()
    try:
        import customtkinter as ctk  # noqa: F401
        import open3d as o3d  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "customtkinter / open3d 가 필요합니다. conda activate AIvision"
        ) from exc

    from motion import Controller, DEFAULT_URDF, RobotKinematics
    from motion.visualizer import Visualizer

    mesh_path = cad_mesh_path()
    vertices_m, faces = load_mesh_m(mesh_path)
    place, grasp = load_specs(CAD_YAML)

    kin = RobotKinematics(DEFAULT_URDF)
    viz = Visualizer(kin, open_browser=not args.no_open)
    url = viz.url or "(meshcat server running)"
    print(f"CAD      {mesh_path}")
    print(f"Meshcat  {url}")

    ctrl = Controller(kin)
    ctrl.start()

    import customtkinter as ctk

    ctk.set_appearance_mode("dark")
    root = ctk.CTk()
    gui = TeachGraspGui(
        root,
        visualizer=viz,
        kinematics=kin,
        controller=ctrl,
        own_robot=False,
        drive_viz=True,
        mesh_path=mesh_path,
        vertices_m=vertices_m,
        faces=faces,
        place=place,
        grasp=grasp,
        meshcat_url=url,
    )

    def _on_close() -> None:
        gui._cancel_register_poll()
        ctrl.stop()
        try:
            viz.close()
        except Exception:
            pass
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
