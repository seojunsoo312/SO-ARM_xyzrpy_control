"""Meshcat debug view. display(q) only — no IK, no input."""

from __future__ import annotations

import sys
from pathlib import Path

_PROJECT = Path(__file__).resolve().parent.parent
if str(_PROJECT) not in sys.path:
    sys.path.insert(0, str(_PROJECT))

import meshcat.geometry as g
import numpy as np
from pinocchio.visualize import MeshcatVisualizer

from motion.robot_kinematics import RobotKinematics

# Meshcat OrbitControls always orbit world origin. Do not translate
# /Cameras/default — that parent offset makes pan/orbit/dolly fight the target.
# Camera lives in an Rx(+90°) frame; default convention is (x, z, 0) like (3, 1, 0).
# Closer copy of that so the ~0.4 m arm fills the view.
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
        # Base frame at world origin with X/Y/Z letters.
        self.set_overlay_axes("base", np.eye(4), scale=0.06, tag="B")

    def _set_initial_camera(self) -> None:
        vis = self._viz.viewer
        vis["/Cameras/default"].set_transform(np.eye(4))
        vis["/Cameras/default/rotated/<object>"].set_property("position", list(CAM_POSITION))

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
        labels: bool = True,
        tag: str | None = None,
    ) -> None:
        """RGB triad. labels → tip X/Y/Z. tag → cyan frame letter (C/G/P/B…)."""
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

    def display(self, q: np.ndarray) -> None:
        self._viz.display(q)

    @property
    def url(self) -> str:
        try:
            return str(self._viz.viewer.url())
        except Exception:
            return ""
