"""Orbbec color. V4L2 노드가 있으면 그걸 쓰고, 없으면 SDK."""

from __future__ import annotations

from pathlib import Path

import cv2

CAMERA_INDEX = None
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
ROTATE_180 = True
ORBBEC_VID = "2bc5"


def _usb_vid(sys_video: Path) -> str | None:
    device = (sys_video / "device").resolve()
    for _ in range(10):
        vid = device / "idVendor"
        if vid.is_file():
            return vid.read_text().strip().lower()
        parent = device.parent
        if parent == device:
            break
        device = parent
    return None


def find_orbbec_v4l_index() -> int | None:
    root = Path("/sys/class/video4linux")
    if not root.is_dir():
        return None
    for node in sorted(root.glob("video*"), key=lambda p: p.name):
        try:
            index = int(node.name.replace("video", ""))
        except ValueError:
            continue
        if _usb_vid(node) != ORBBEC_VID:
            continue
        name = ""
        name_path = node / "name"
        if name_path.is_file():
            name = name_path.read_text().strip().lower()
        if "metadata" in name:
            continue
        return index
    return None


class OrbbecCapture:
    """detect/capture/vision_test 가 쓰는 VideoCapture 비슷한 래퍼."""

    def __init__(self, width: int = FRAME_WIDTH, height: int = FRAME_HEIGHT):
        from vision.rgbd import OrbbecV1

        self._cam = OrbbecV1(width=width, height=height)
        self._w = width
        self._h = height
        bgr = None
        for _ in range(30):
            bgr, _ = self._cam.grab()
            if bgr is not None:
                break
        if bgr is None:
            self._cam.close()
            raise RuntimeError(
                "Orbbec 컬러 프레임이 없습니다. Viewer를 끄고, 가능하면 USB3에 직접 연결하세요."
            )
        self._h, self._w = bgr.shape[:2]

    def isOpened(self) -> bool:
        return getattr(self._cam, "pipe", None) is not None

    def read(self):
        bgr, _ = self._cam.grab()
        if bgr is None:
            return False, None
        self._h, self._w = bgr.shape[:2]
        return True, bgr

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self._w)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self._h)
        if prop == cv2.CAP_PROP_FPS:
            return 30.0
        return 0.0

    def set(self, _prop: int, _value) -> bool:
        return False

    def release(self) -> None:
        self._cam.close()


def _open_v4l2(index: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"카메라를 열 수 없습니다: index={index} (/dev/video{index})")
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap


def open_camera(index: int | None = CAMERA_INDEX):
    if index is not None:
        return _open_v4l2(index)
    v4l = find_orbbec_v4l_index()
    if v4l is not None:
        return _open_v4l2(v4l)
    return OrbbecCapture(FRAME_WIDTH, FRAME_HEIGHT)


def grab_bgr(cap, rotate_180: bool = ROTATE_180):
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    if rotate_180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    return frame


def rotate_image(img, rotate_180: bool = ROTATE_180):
    if img is None or not rotate_180:
        return img
    return cv2.rotate(img, cv2.ROTATE_180)


def rotate_rgbd(bgr, depth_mm, rotate_180: bool = ROTATE_180):
    return rotate_image(bgr, rotate_180), rotate_image(depth_mm, rotate_180)
