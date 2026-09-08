#!/usr/bin/env python3
"""물체 배치 + grasp yaml 값을 Meshcat에서 확인·티칭하는 GUI.

권장: 펜던트와 한 프로세스·한 Meshcat.

  python pendant/main.py              # 펜던트만
  python pendant/main.py --grasp      # 조그 + 물체 집기 창 (같은 Meshcat)
  python yolo/teach_grasp.py          # 물체 창만 (가상 팔로 P/G 이동)
"""

from __future__ import annotations

import argparse
import ast
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
PENDANT = PROJECT / "pendant"
for path in (PROJECT, PENDANT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from yolo.config import CAD_YAML, cad_mesh_path, cad_unit, quiet_gtk  # noqa: E402
from yolo.register import _to_mm  # noqa: E402

# 시뮬에서 물체를 책상에 올려 두는 기본값 (베이스 mm / deg).
# Rx(180): CAD −Z 세로판이 위로. XY판이 z≈0 근처.
DEFAULT_PLACE_XYZ_MM = (200.0, 0.0, 2.0)
DEFAULT_PLACE_RPY_DEG = (180.0, 0.0, 0.0)
DEFAULT_GRASP_XYZ_MM = (20.0, 0.0, 1.0)
DEFAULT_GRASP_RPY_DEG = (180.0, 0.0, 0.0)
DEFAULT_PRE_GRASP_MM = 45.0
DEFAULT_GRIPPER = 50.0
DISPLAY_MS = 33


@dataclass(frozen=True)
class PlacePose:
    xyz_mm: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]


@dataclass(frozen=True)
class GraspSpec:
    xyz_mm: tuple[float, float, float]
    rpy_deg: tuple[float, float, float]
    pre_grasp_mm: float
    gripper: float


def _vec3(raw: object, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    if isinstance(raw, (list, tuple)) and len(raw) == 3:
        return (float(raw[0]), float(raw[1]), float(raw[2]))
    if isinstance(raw, str):
        try:
            parsed = ast.literal_eval(raw)
        except (SyntaxError, ValueError):
            return fallback
        return _vec3(parsed, fallback)
    return fallback


def _scalar(raw: object, fallback: float) -> float:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return fallback


def _parse_model_yaml(path: Path) -> dict:
    """들여쓰기 2칸짜리 작은 yaml. PyYAML 없이 grasp/place 만 읽는다."""
    if not path.is_file():
        return {}
    root: dict = {}
    section: str | None = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        if "#" in raw:
            raw = raw.split("#", 1)[0]
        if not raw.strip():
            continue
        indent = len(raw) - len(raw.lstrip(" "))
        line = raw.strip()
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if indent == 0:
            section = None
            if value == "":
                section = key
                root[key] = {}
            else:
                root[key] = value.strip("'\"")
            continue
        if section is None or not isinstance(root.get(section), dict):
            continue
        root[section][key] = ast.literal_eval(value) if value else None
    return root


def load_specs(path: Path) -> tuple[PlacePose, GraspSpec]:
    data = _parse_model_yaml(path)
    place_raw = data.get("place") if isinstance(data.get("place"), dict) else {}
    grasp_raw = data.get("grasp") if isinstance(data.get("grasp"), dict) else {}
    place = PlacePose(
        xyz_mm=_vec3(place_raw.get("xyz_mm"), DEFAULT_PLACE_XYZ_MM),
        rpy_deg=_vec3(place_raw.get("rpy_deg"), DEFAULT_PLACE_RPY_DEG),
    )
    grasp = GraspSpec(
        xyz_mm=_vec3(grasp_raw.get("xyz_mm"), DEFAULT_GRASP_XYZ_MM),
        rpy_deg=_vec3(grasp_raw.get("rpy_deg"), DEFAULT_GRASP_RPY_DEG),
        pre_grasp_mm=_scalar(grasp_raw.get("pre_grasp_mm"), DEFAULT_PRE_GRASP_MM),
        gripper=_scalar(grasp_raw.get("gripper"), DEFAULT_GRIPPER),
    )
    return place, grasp


def pose_to_T(xyz_mm: np.ndarray, rpy_deg: np.ndarray) -> np.ndarray:
    from motion.robot_kinematics import rpy_deg_to_rotmat

    T = np.eye(4)
    T[:3, :3] = rpy_deg_to_rotmat(float(rpy_deg[0]), float(rpy_deg[1]), float(rpy_deg[2]))
    T[:3, 3] = np.asarray(xyz_mm, dtype=float).reshape(3) / 1000.0
    return T


def T_to_xyzrpy(T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    from motion.robot_kinematics import rotmat_to_rpy_deg

    M = np.asarray(T, dtype=float)
    xyz_mm = M[:3, 3] * 1000.0
    rpy_deg = rotmat_to_rpy_deg(M[:3, :3])
    return xyz_mm, rpy_deg


def frames_from_specs(place: PlacePose, grasp: GraspSpec) -> tuple[np.ndarray, np.ndarray]:
    """Return (T_base_grasp, T_base_pre)."""
    T_base_cad = pose_to_T(np.array(place.xyz_mm), np.array(place.rpy_deg))
    T_cad_grasp = pose_to_T(np.array(grasp.xyz_mm), np.array(grasp.rpy_deg))
    T_base_grasp = T_base_cad @ T_cad_grasp
    T_base_pre = T_base_grasp.copy()
    T_base_pre[:3, 3] = T_base_grasp[:3, 3] - T_base_grasp[:3, 2] * (grasp.pre_grasp_mm / 1000.0)
    return T_base_grasp, T_base_pre


def load_cad_mesh_m(path: Path) -> tuple[np.ndarray, np.ndarray]:
    import open3d as o3d

    mesh = o3d.io.read_triangle_mesh(str(path))
    if not mesh.has_triangles() or len(mesh.triangles) == 0:
        raise RuntimeError(f"삼각형 없음: {path}")
    verts_mm = _to_mm(np.asarray(mesh.vertices), cad_unit(), path)
    faces = np.asarray(mesh.triangles, dtype=np.uint32)
    return verts_mm / 1000.0, faces


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
    ) -> None:
        import customtkinter as ctk

        self.root = root
        self._viz = visualizer
        self._kin = kinematics
        self._ctrl = controller
        self._own_robot = bool(own_robot) and controller is None
        self._drive_viz = bool(drive_viz) and controller is not None
        self._yaml_path = CAD_YAML
        self._cad_loaded = False
        self._status: ctk.CTkLabel | None = None
        self.entries: dict[str, ctk.CTkEntry] = {}
        self._replay_after: str | None = None

        root.title("물체 집기 티칭")
        root.minsize(460, 820)
        root.geometry("500x880")
        root.protocol("WM_DELETE_WINDOW", self.on_close)

        if self._ctrl is not None:
            robot_hint = "→P / →G / P→G 로 가상·실물 팔을 목표로 보냄. 조그는 펜던트."
        elif self._own_robot:
            robot_hint = "단독 모드: 적용 시 팔은 HOME, 그리퍼만 반영. 이동은 펜던트 --grasp 권장."
        else:
            robot_hint = "팔 컨트롤러 없음. 오버레이만."
        ctk.CTkLabel(
            root,
            text=(
                f"CAD {mesh_path.name}  ·  Meshcat: {meshcat_url or 'printed URL'}\n"
                f"물체 xyzrpy = 시뮬 베이스. grasp = CAD 축 기준.\n"
                f"{robot_hint}"
            ),
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=14, pady=(14, 8))

        body = ctk.CTkScrollableFrame(root, fg_color="transparent")
        body.pack(fill="both", expand=True, padx=10, pady=4)

        self._section(body, "시뮬 물체 (베이스 mm / deg)")
        self._xyzrpy_block(body, "place", place.xyz_mm, place.rpy_deg)
        ctk.CTkLabel(
            body,
            text="책상에 올려 두는 자세. yaml 집기 값이 아니다.",
            text_color="#9ca3af",
            anchor="w",
        ).pack(fill="x", padx=8, pady=(0, 10))

        self._section(body, "grasp (CAD mm / deg)")
        self._xyzrpy_block(body, "grasp", grasp.xyz_mm, grasp.rpy_deg)
        self._row(body, "pre_grasp_mm", f"{grasp.pre_grasp_mm:.1f}", "오프셋")
        self._row(body, "gripper", f"{grasp.gripper:.1f}", "벌림 0–100")
        grip_note = (
            "이동 전에 그리퍼 값을 반영하려면 →P 직전 펜던트에서 벌리거나, 아래 이동 버튼 사용."
            if self._ctrl is not None
            else "단독 모드에서만 적용 시 턱 벌림에 반영."
        )
        ctk.CTkLabel(
            body,
            text=(
                "P = G에서 집기 프레임 −Z 로 오프셋. 노란 선 = 접근.\n"
                "축: 빨강=X 초록=Y 파랑=Z. 끝 글자 X/Y/Z. 청록 C=CAD G=grasp P=pre.\n"
                + grip_note
            ),
            text_color="#9ca3af",
            anchor="w",
            justify="left",
        ).pack(fill="x", padx=8, pady=(0, 10))

        bar = ctk.CTkFrame(root, fg_color="transparent")
        bar.pack(fill="x", padx=14, pady=(4, 4))
        ctk.CTkButton(bar, text="적용", width=90, command=self.apply).pack(side="left", padx=(0, 6))
        ctk.CTkButton(bar, text="초기화", width=90, command=self.reset).pack(side="left", padx=(0, 6))

        move = ctk.CTkFrame(root, fg_color="transparent")
        move.pack(fill="x", padx=14, pady=(0, 8))
        self._btn_pre = ctk.CTkButton(move, text="→ P", width=90, command=self.goto_pre)
        self._btn_grasp = ctk.CTkButton(move, text="→ G", width=90, command=self.goto_grasp)
        self._btn_seq = ctk.CTkButton(move, text="P → G", width=90, command=self.replay_pg)
        self._btn_pre.pack(side="left", padx=(0, 6))
        self._btn_grasp.pack(side="left", padx=(0, 6))
        self._btn_seq.pack(side="left")
        if self._ctrl is None:
            for btn in (self._btn_pre, self._btn_grasp, self._btn_seq):
                btn.configure(state="disabled")

        self._status = ctk.CTkLabel(root, text="", anchor="w")
        self._status.pack(fill="x", padx=14, pady=(0, 14))

        root.bind("<Return>", lambda _e: self.apply())
        self._draw_static(vertices_m, faces)
        self.apply()
        if self._drive_viz:
            self._schedule_display()

    def _section(self, parent, title: str) -> None:
        import customtkinter as ctk

        ctk.CTkLabel(parent, text=title, anchor="w", font=ctk.CTkFont(size=15, weight="bold")).pack(
            fill="x", padx=8, pady=(12, 4)
        )

    def _xyzrpy_block(
        self,
        parent,
        prefix: str,
        xyz: tuple[float, float, float],
        rpy: tuple[float, float, float],
    ) -> None:
        labels = (
            (f"{prefix}_x", "x mm", xyz[0]),
            (f"{prefix}_y", "y mm", xyz[1]),
            (f"{prefix}_z", "z mm", xyz[2]),
            (f"{prefix}_roll", "roll deg", rpy[0]),
            (f"{prefix}_pitch", "pitch deg", rpy[1]),
            (f"{prefix}_yaw", "yaw deg", rpy[2]),
        )
        for key, label, val in labels:
            self._row(parent, key, f"{val:.1f}", label)

    def _row(self, parent, key: str, value: str, label: str) -> None:
        import customtkinter as ctk

        row = ctk.CTkFrame(parent)
        row.pack(fill="x", padx=8, pady=2)
        ctk.CTkLabel(row, text=label, width=110, anchor="w").pack(side="left")
        ent = ctk.CTkEntry(row, width=140)
        ent.insert(0, value)
        ent.pack(side="left", padx=6)
        self.entries[key] = ent

    def _set_status(self, text: str, *, error: bool = False) -> None:
        if self._status is None:
            return
        self._status.configure(text=text, text_color="#f87171" if error else "#86efac")

    def _read_float(self, key: str) -> float:
        return float(self.entries[key].get().strip())

    def _read_place_grasp(self) -> tuple[PlacePose, GraspSpec]:
        place = PlacePose(
            xyz_mm=(
                self._read_float("place_x"),
                self._read_float("place_y"),
                self._read_float("place_z"),
            ),
            rpy_deg=(
                self._read_float("place_roll"),
                self._read_float("place_pitch"),
                self._read_float("place_yaw"),
            ),
        )
        gripper = self._read_float("gripper")
        if not 0.0 <= gripper <= 100.0:
            raise ValueError("gripper 는 0–100")
        grasp = GraspSpec(
            xyz_mm=(
                self._read_float("grasp_x"),
                self._read_float("grasp_y"),
                self._read_float("grasp_z"),
            ),
            rpy_deg=(
                self._read_float("grasp_roll"),
                self._read_float("grasp_pitch"),
                self._read_float("grasp_yaw"),
            ),
            pre_grasp_mm=self._read_float("pre_grasp_mm"),
            gripper=gripper,
        )
        return place, grasp

    def _fill(self, place: PlacePose, grasp: GraspSpec) -> None:
        mapping = {
            "place_x": place.xyz_mm[0],
            "place_y": place.xyz_mm[1],
            "place_z": place.xyz_mm[2],
            "place_roll": place.rpy_deg[0],
            "place_pitch": place.rpy_deg[1],
            "place_yaw": place.rpy_deg[2],
            "grasp_x": grasp.xyz_mm[0],
            "grasp_y": grasp.xyz_mm[1],
            "grasp_z": grasp.xyz_mm[2],
            "grasp_roll": grasp.rpy_deg[0],
            "grasp_pitch": grasp.rpy_deg[1],
            "grasp_yaw": grasp.rpy_deg[2],
            "pre_grasp_mm": grasp.pre_grasp_mm,
            "gripper": grasp.gripper,
        }
        for key, val in mapping.items():
            self.entries[key].delete(0, "end")
            self.entries[key].insert(0, f"{val:.1f}")

    def _draw_static(self, vertices_m: np.ndarray, faces: np.ndarray) -> None:
        table_t = 0.004
        T_table = np.eye(4)
        T_table[2, 3] = -table_t / 2.0
        self._viz.set_overlay_box(
            "table",
            (0.45, 0.45, table_t),
            T_table,
            color=0x2F2F35,
            opacity=0.45,
        )
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

    def _schedule_display(self) -> None:
        if self._ctrl is None:
            return
        st = self._ctrl.snapshot()
        self._viz.display(st.q)
        self.root.after(DISPLAY_MS, self._schedule_display)

    def apply(self) -> None:
        try:
            place, grasp = self._read_place_grasp()
        except ValueError as exc:
            self._set_status(f"입력 오류: {exc}", error=True)
            return
        T_base_cad = pose_to_T(np.array(place.xyz_mm), np.array(place.rpy_deg))
        T_base_grasp, T_base_pre = frames_from_specs(place, grasp)

        if self._cad_loaded:
            self._viz.set_overlay_transform("cad", T_base_cad)
        self._viz.set_overlay_axes("cad_axes", T_base_cad, scale=0.03, tag="C")
        self._viz.set_overlay_axes("grasp", T_base_grasp, scale=0.05, tag="G")
        self._viz.set_overlay_axes("pregrasp", T_base_pre, scale=0.035, tag="P")
        self._viz.set_overlay_segment("approach", T_base_pre[:3, 3], T_base_grasp[:3, 3])
        if self._own_robot:
            self._show_robot(grasp.gripper)
        self._set_status(
            f"적용  CAD xyz={list(np.round(place.xyz_mm, 1))}  "
            f"grasp xyz={list(np.round(grasp.xyz_mm, 1))}  "
            f"pre={grasp.pre_grasp_mm:.1f}  gripper={grasp.gripper:.1f}"
        )

    def reset(self) -> None:
        place, grasp = load_specs(self._yaml_path)
        self._fill(place, grasp)
        self.apply()
        self._set_status(f"초기화: {self._yaml_path.name} 값")

    def _goto_T(self, T: np.ndarray, *, label: str) -> bool:
        if self._ctrl is None:
            self._set_status("컨트롤러 없음. python pendant/main.py --grasp", error=True)
            return False
        self.apply()
        xyz_mm, rpy_deg = T_to_xyzrpy(T)
        self._ctrl.start_ee_goto(xyz_mm, rpy_deg)
        self._set_status(
            f"{label} 이동  xyz=[{xyz_mm[0]:.1f}, {xyz_mm[1]:.1f}, {xyz_mm[2]:.1f}]  "
            f"rpy=[{rpy_deg[0]:.1f}, {rpy_deg[1]:.1f}, {rpy_deg[2]:.1f}]"
        )
        return True

    def goto_pre(self) -> None:
        try:
            place, grasp = self._read_place_grasp()
        except ValueError as exc:
            self._set_status(f"입력 오류: {exc}", error=True)
            return
        _, T_pre = frames_from_specs(place, grasp)
        self._goto_T(T_pre, label="→ P")

    def goto_grasp(self) -> None:
        try:
            place, grasp = self._read_place_grasp()
        except ValueError as exc:
            self._set_status(f"입력 오류: {exc}", error=True)
            return
        T_g, _ = frames_from_specs(place, grasp)
        self._goto_T(T_g, label="→ G")

    def replay_pg(self) -> None:
        if self._ctrl is None:
            self._set_status("컨트롤러 없음. python pendant/main.py --grasp", error=True)
            return
        from motion.controller import GOTO_DURATION_S

        try:
            place, grasp = self._read_place_grasp()
        except ValueError as exc:
            self._set_status(f"입력 오류: {exc}", error=True)
            return
        T_g, T_pre = frames_from_specs(place, grasp)
        if not self._goto_T(T_pre, label="P → G (1/2 P)"):
            return
        delay_ms = int(GOTO_DURATION_S * 1000) + 200
        if self._replay_after is not None:
            try:
                self.root.after_cancel(self._replay_after)
            except Exception:
                pass

        def _second() -> None:
            self._replay_after = None
            self._goto_T(T_g, label="P → G (2/2 G)")

        self._replay_after = self.root.after(delay_ms, _second)

    def on_close(self) -> None:
        if self._replay_after is not None:
            try:
                self.root.after_cancel(self._replay_after)
            except Exception:
                pass
            self._replay_after = None
        self.root.destroy()


def attach_teach_window(
    parent,
    visualizer,
    *,
    meshcat_url: str,
    controller=None,
    geometry: str = "500x880+1000+40",
) -> TeachGraspGui | None:
    """펜던트와 같은 Meshcat에 teach 창을 붙인다. 팔 display(q) 는 펜던트가 담당."""
    import customtkinter as ctk

    try:
        mesh_path = cad_mesh_path()
        vertices_m, faces = load_cad_mesh_m(mesh_path)
    except Exception as exc:
        print(f"teach 창 생략: CAD 로드 실패 ({exc})")
        return None
    place, grasp = load_specs(CAD_YAML)
    win = ctk.CTkToplevel(parent)
    win.geometry(geometry)
    gui = TeachGraspGui(
        win,
        visualizer=visualizer,
        mesh_path=mesh_path,
        vertices_m=vertices_m,
        faces=faces,
        place=place,
        grasp=grasp,
        meshcat_url=meshcat_url,
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
            "customtkinter / open3d 가 필요합니다. conda activate lerobot"
        ) from exc

    from motion import Controller, DEFAULT_URDF, RobotKinematics
    from visualizer import Visualizer

    mesh_path = cad_mesh_path()
    vertices_m, faces = load_cad_mesh_m(mesh_path)
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
    TeachGraspGui(
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
        ctrl.stop()
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", _on_close)
    root.mainloop()


if __name__ == "__main__":
    main()
