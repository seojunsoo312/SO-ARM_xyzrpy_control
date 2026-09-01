"""Meshcat debug view. display(q) only — no IK, no input."""

from __future__ import annotations

import numpy as np
from pinocchio.visualize import MeshcatVisualizer

from robot_kinematics import RobotKinematics

# Meshcat OrbitControls always orbit world origin. Do not translate
# /Cameras/default — that parent offset makes pan/orbit/dolly fight the target.
# Camera lives in an Rx(+90°) frame; default convention is (x, z, 0) like (3, 1, 0).
# Closer copy of that so the ~0.4 m arm fills the view.
CAM_POSITION = (0.70, 0.32, 0.0)


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

    def _set_initial_camera(self) -> None:
        vis = self._viz.viewer
        vis["/Cameras/default"].set_transform(np.eye(4))
        vis["/Cameras/default/rotated/<object>"].set_property("position", list(CAM_POSITION))

    def display(self, q: np.ndarray) -> None:
        self._viz.display(q)

    @property
    def url(self) -> str:
        try:
            return str(self._viz.viewer.url())
        except Exception:
            return ""
