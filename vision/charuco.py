"""ChArUco board: OpenCV (7, 5), DICT_4X4_50.

화면상 가로 7칸 / 세로 5칸, 마커는 4x4. 검은 칸 20 mm, 마커 15 mm.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

SQUARES_X = 7
SQUARES_Y = 5
ARUCO_DICT_NAME = "DICT_4X4_50"
SQUARE_MM = 20.0
MARKER_MM = 15.0
MIN_CORNERS = 16


@dataclass
class DetectedBoard:
    marker_corners: np.ndarray | None
    marker_ids: np.ndarray | None
    charuco_corners: np.ndarray | None
    charuco_ids: np.ndarray | None

    @property
    def n_markers(self) -> int:
        if self.marker_ids is None:
            return 0
        return int(len(self.marker_ids))

    @property
    def n_corners(self) -> int:
        if self.charuco_corners is None or self.charuco_ids is None:
            return 0
        ids = np.asarray(self.charuco_ids).reshape(-1)
        corners = np.asarray(self.charuco_corners).reshape(-1, 2)
        n = len(ids)
        return n if len(corners) == n else 0

    @property
    def ok(self) -> bool:
        return self.n_corners >= MIN_CORNERS


def board_label() -> str:
    return f"{SQUARES_X}x{SQUARES_Y} {ARUCO_DICT_NAME}"


def aruco_dictionary(name: str = ARUCO_DICT_NAME) -> cv2.aruco.Dictionary:
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def make_board(square_mm: float | None = None) -> cv2.aruco.CharucoBoard:
    sq = float(SQUARE_MM if square_mm is None else square_mm)
    if sq <= 0:
        raise ValueError("square_mm은 0보다 커야 합니다.")
    marker = MARKER_MM * (sq / SQUARE_MM)
    return cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        sq,
        marker,
        aruco_dictionary(),
    )


def detect_board(gray, board: cv2.aruco.CharucoBoard | None = None) -> DetectedBoard:
    if board is None:
        board = make_board()
    if gray.ndim == 3:
        gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    detector = cv2.aruco.CharucoDetector(board)
    ch_c, ch_ids, m_c, m_ids = detector.detectBoard(gray)
    return DetectedBoard(m_c, m_ids, ch_c, ch_ids)


def draw_detection(bgr, det: DetectedBoard) -> np.ndarray:
    vis = bgr.copy()
    if det.marker_ids is not None:
        cv2.aruco.drawDetectedMarkers(vis, det.marker_corners, det.marker_ids)
    if det.n_corners > 0:
        cc = np.asarray(det.charuco_corners, dtype=np.float32).reshape(-1, 1, 2)
        ci = np.asarray(det.charuco_ids, dtype=np.int32).reshape(-1, 1)
        cv2.aruco.drawDetectedCornersCharuco(vis, cc, ci)
    color = (0, 255, 0) if det.ok else (0, 0, 255)
    expect = (SQUARES_X - 1) * (SQUARES_Y - 1)
    cv2.putText(
        vis,
        f"markers={det.n_markers}  corners={det.n_corners}/{expect}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    return vis


def match_points(det: DetectedBoard, board: cv2.aruco.CharucoBoard):
    if not det.ok:
        return None, None
    obj, img = board.matchImagePoints(det.charuco_corners, det.charuco_ids)
    if obj is None or img is None or len(obj) < MIN_CORNERS:
        return None, None
    return obj, img
