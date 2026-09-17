"""Meshcat debug view. display(q) only — no IK, no input."""

from __future__ import annotations

import atexit
import html
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

import meshcat.geometry as g
import meshcat.visualizer as meshcat_visualizer
import numpy as np
from pinocchio.visualize import MeshcatVisualizer

from motion.base_frame import T_urdf_from_user_matrix
from motion.robot_kinematics import RobotKinematics

# Browser tab title (meshcat hardcodes "MeshCat" in viewer/dist/index.html).
PAGE_TITLE = "시뮬레이터"

# Meshcat OrbitControls always orbit world origin. Do not translate
# /Cameras/default — that parent offset makes pan/orbit/dolly fight the target.
# Camera lives in an Rx(+90°) frame; default convention is (x, z, 0) like (3, 1, 0).
# Closer copy of that so the ~0.4 m arm fills the view.
# Project base axes (overlay): +X forward; see motion/base_frame.py.
CAM_POSITION = (0.70, 0.32, 0.0)
OVERLAY_ROOT = "teach"

# RGB matching meshcat.geometry.triad (X red, Y green, Z blue).
_AXIS_COLORS = (0xE53935, 0x43A047, 0x1E88E8)

# Unit-box stroke glyphs in XY (normalized ±0.5). Scaled later.
_GLYPHS: dict[str, tuple[tuple[float, float, float, float], ...]] = {
    "X": ((-0.5, -0.5, 0.5, 0.5), (-0.5, 0.5, 0.5, -0.5)),
    "Y": ((-0.5, 0.5, 0.0, 0.0), (0.5, 0.5, 0.0, 0.0), (0.0, 0.0, 0.0, -0.5)),
    "Z": ((-0.5, 0.5, 0.5, 0.5), (0.5, 0.5, -0.5, -0.5), (-0.5, -0.5, 0.5, -0.5)),
    "C": ((0.45, 0.5, -0.5, 0.5), (-0.5, 0.5, -0.5, -0.5), (-0.5, -0.5, 0.45, -0.5)),
    "G": (
        (0.45, 0.5, -0.5, 0.5),
        (-0.5, 0.5, -0.5, -0.5),
        (-0.5, -0.5, 0.5, -0.5),
        (0.5, -0.5, 0.5, 0.0),
        (0.5, 0.0, 0.0, 0.0),
    ),
    "P": (
        (-0.5, -0.5, -0.5, 0.5),
        (-0.5, 0.5, 0.45, 0.5),
        (0.45, 0.5, 0.45, 0.05),
        (0.45, 0.05, -0.5, 0.05),
    ),
    "B": (  # base
        (-0.5, -0.5, -0.5, 0.5),
        (-0.5, 0.5, 0.35, 0.5),
        (0.35, 0.5, 0.35, 0.05),
        (0.35, 0.05, -0.5, 0.05),
        (-0.5, 0.05, 0.45, 0.05),
        (0.45, 0.05, 0.45, -0.5),
        (0.45, -0.5, -0.5, -0.5),
    ),
}


def _glyph_points(letter: str, *, size: float) -> np.ndarray:
    """Stroke glyph → (3, N) LineSegments positions in local XY."""
    key = letter.upper()
    segs = _GLYPHS.get(key)
    if segs is None:
        segs = _GLYPHS["X"]
    s = float(size)
    pts: list[list[float]] = []
    for x0, y0, x1, y1 in segs:
        pts.append([x0 * s, y0 * s, 0.0])
        pts.append([x1 * s, y1 * s, 0.0])
    return np.asarray(pts, dtype=np.float32).T


def _axis_label_transform(axis: int, scale: float, letter_size: float) -> np.ndarray:
    """Letter just past the axis tip, drawn in a plane readable from that tip."""
    T = np.eye(4)
    tip = float(scale) + 0.55 * float(letter_size)
    T[axis, 3] = tip
    if axis == 0:  # +X → letter in YZ (Ry(-90°))
        T[:3, :3] = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
    elif axis == 1:  # +Y → letter in XZ (Rx(+90°))
        T[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]])
    # +Z: letter stays in XY
    return T


def _meshcat_viewer_root(title: str) -> str:
    """Copy meshcat's viewer dist and replace the HTML <title>."""
    import meshcat

    src = Path(meshcat.__file__).resolve().parent / "viewer" / "dist"
    dst = Path(tempfile.mkdtemp(prefix="meshcat_viewer_"))
    atexit.register(shutil.rmtree, dst, True)
    for path in src.iterdir():
        target = dst / path.name
        if path.name == "index.html":
            text = path.read_text(encoding="utf-8")
            text = text.replace("<title>MeshCat</title>", f"<title>{html.escape(title)}</title>", 1)
            target.write_text(text, encoding="utf-8")
        elif path.is_file():
            os.symlink(path, target)
    return str(dst)


def _start_meshcat_server(zmq_url=None, server_args=None):
    """Same as meshcat's launcher, but serve our titled index.html."""
    from meshcat.servers.zmqserver import match_web_url, match_zmq_url

    root = _meshcat_viewer_root(PAGE_TITLE)
    code = (
        "import meshcat.servers.zmqserver as s;"
        f"s.VIEWER_ROOT={root!r};"
        "s.main()"
    )
    args = [sys.executable, "-u", "-c", code]
    if zmq_url is not None:
        args.extend(["--zmq-url", zmq_url])
    if server_args:
        args.extend(server_args)
    env = dict(os.environ)
    import meshcat

    env["PYTHONPATH"] = str(Path(meshcat.__file__).resolve().parent.parent)
    server_proc = subprocess.Popen(
        args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        start_new_session=True,
    )
    line = ""
    while "zmq_url" not in line:
        line = server_proc.stdout.readline().strip().decode("utf-8")
        if server_proc.poll() is not None:
            outs, errs = server_proc.communicate()
            print(outs.decode("utf-8"))
            print(errs.decode("utf-8"))
            raise RuntimeError(
                "the meshcat server process exited prematurely with exit code "
                + str(server_proc.poll())
            )
    zmq_url = match_zmq_url(line)
    web_url = match_web_url(server_proc.stdout.readline().strip().decode("utf-8"))

    def cleanup(proc):
        proc.kill()
        proc.wait()

    atexit.register(cleanup, server_proc)
    return server_proc, zmq_url, web_url


meshcat_visualizer.start_zmq_server_as_subprocess = _start_meshcat_server


class Visualizer:
    def __init__(self, kinematics: RobotKinematics, *, open_browser: bool = True) -> None:
        self._kin = kinematics
        self._viz = MeshcatVisualizer(
            kinematics.model,
            kinematics.collision_model,
            kinematics.visual_model,
        )
        self._viz.initViewer(open=open_browser)
        self._viz.loadViewerModel(rootNodeName="SO101_6DOF")
        self._viz.displayCollisions(False)
        self._viz.displayVisuals(True)
        # TCP axes (body-fixed on L6 / wrist_roll, not the moving jaw).
        self._viz.displayFrames(True, frame_ids=[kinematics.ee_frame_id], axis_length=0.05)
        self._set_initial_camera()
        self._align_floor_axes()
        self._mark_origin()
        # Project-base triad (same +X-forward frame as /Axes).
        self.set_overlay_axes("base", T_urdf_from_user_matrix(), scale=0.06)

    def _mark_origin(self) -> None:
        """Red sphere at shared URDF / project-base origin (0,0,0)."""
        node = self._viz.viewer["origin"]
        node.set_object(
            g.Sphere(0.002),
            g.MeshLambertMaterial(color=0xE53935, reflectivity=0.1),
        )
        node.set_transform(np.eye(4))

    def _set_initial_camera(self) -> None:
        vis = self._viz.viewer
        vis["/Cameras/default"].set_transform(np.eye(4))
        vis["/Cameras/default/rotated/<object>"].set_property("position", list(CAM_POSITION))

    def _align_floor_axes(self) -> None:
        """Meshcat `/Axes` defaults to URDF world — rotate to project base (+X forward)."""
        self._viz.viewer["/Axes"].set_transform(T_urdf_from_user_matrix())

    def _overlay(self, name: str):
        return self._viz.viewer[OVERLAY_ROOT][name]

    def set_overlay_mesh(
        self,
        name: str,
        vertices_m: np.ndarray,
        faces: np.ndarray,
        *,
        color: int = 0xB8B8BC,
        opacity: float = 0.95,
    ) -> None:
        verts = np.asarray(vertices_m, dtype=np.float32).reshape(-1, 3)
        tris = np.asarray(faces, dtype=np.uint32).reshape(-1, 3)
        self._overlay(name).set_object(
            g.TriangularMeshGeometry(verts, tris),
            g.MeshLambertMaterial(
                color=int(color),
                opacity=float(opacity),
                transparent=float(opacity) < 1.0,
            ),
        )

    def set_overlay_box(
        self,
        name: str,
        lengths_m: tuple[float, float, float],
        T: np.ndarray,
        *,
        color: int = 0x3A3A40,
        opacity: float = 0.5,
    ) -> None:
        node = self._overlay(name)
        node.set_object(
            g.Box([float(lengths_m[0]), float(lengths_m[1]), float(lengths_m[2])]),
            g.MeshLambertMaterial(
                color=int(color),
                opacity=float(opacity),
                transparent=True,
            ),
        )
        node.set_transform(np.asarray(T, dtype=float))

    def set_overlay_axes(
        self,
        name: str,
        T: np.ndarray,
        *,
        scale: float = 0.04,
        labels: bool = False,
        tag: str | None = None,
    ) -> None:
        """RGB triad. labels → tip X/Y/Z (off by default). tag → cyan frame letter."""
        root = self._overlay(name)
        root.set_transform(np.asarray(T, dtype=float))
        root["axes"].set_object(g.triad(float(scale)))
        for child in ("label_X", "label_Y", "label_Z", "tag"):
            root[child].delete()
        if labels:
            letter_size = max(0.008, 0.28 * float(scale))
            for axis, letter, color in zip((0, 1, 2), ("X", "Y", "Z"), _AXIS_COLORS, strict=True):
                node = root[f"label_{letter}"]
                node.set_object(
                    g.LineSegments(
                        g.PointsGeometry(position=_glyph_points(letter, size=letter_size)),
                        g.LineBasicMaterial(color=int(color), linewidth=2),
                    )
                )
                node.set_transform(_axis_label_transform(axis, float(scale), letter_size))
        if tag:
            tag_size = max(0.007, 0.22 * float(scale))
            T_tag = np.eye(4)
            T_tag[0, 3] = 0.22 * float(scale)
            T_tag[1, 3] = 0.22 * float(scale)
            T_tag[2, 3] = 0.22 * float(scale)
            node = root["tag"]
            node.set_object(
                g.LineSegments(
                    g.PointsGeometry(
                        position=_glyph_points(tag.strip()[:1] or "X", size=tag_size)
                    ),
                    g.LineBasicMaterial(color=0x22D3EE, linewidth=2),
                )
            )
            node.set_transform(T_tag)

    def set_overlay_transform(self, name: str, T: np.ndarray) -> None:
        self._overlay(name).set_transform(np.asarray(T, dtype=float))

    def clear_overlay(self, name: str) -> None:
        """Remove an overlay subtree (axes, mesh, spheres, …)."""
        try:
            self._overlay(name).delete()
        except Exception:
            pass

    def set_overlay_segment(
        self,
        name: str,
        p0_m: np.ndarray,
        p1_m: np.ndarray,
        *,
        color: int = 0xFACC15,
    ) -> None:
        p0 = np.asarray(p0_m, dtype=np.float32).reshape(3)
        p1 = np.asarray(p1_m, dtype=np.float32).reshape(3)
        node = self._overlay(name)
        node.set_object(
            g.LineSegments(
                g.PointsGeometry(position=np.column_stack([p0, p1])),
                g.LineBasicMaterial(color=int(color), linewidth=3),
            )
        )
        node.set_transform(np.eye(4))

    def set_overlay_spheres(
        self,
        name: str,
        centers_m: np.ndarray,
        *,
        radius_m: float = 0.004,
        color: int = 0x1E88E8,
        colors: list[int] | tuple[int, ...] | None = None,
    ) -> None:
        """Spheres at world positions (m). Optional per-sphere colors."""
        root = self._overlay(name)
        centers = np.asarray(centers_m, dtype=float).reshape(-1, 3)
        r = float(radius_m)
        for i, c in enumerate(centers):
            col = int(colors[i]) if colors is not None else int(color)
            node = root[f"p{i}"]
            node.set_object(
                g.Sphere(r),
                g.MeshLambertMaterial(color=col, reflectivity=0.2),
            )
            T = np.eye(4)
            T[:3, 3] = np.asarray(c, dtype=float).reshape(3)
            node.set_transform(T)

    def display(self, q: np.ndarray) -> None:
        self._viz.display(q)

    @property
    def url(self) -> str:
        try:
            return str(self._viz.viewer.url())
        except Exception:
            return ""
