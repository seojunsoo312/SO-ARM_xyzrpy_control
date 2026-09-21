"""Orbbec SDK v1.10: 컬러 MJPG 640x480 + 소프트웨어 D2C 뎁스."""

from __future__ import annotations

import os
from ctypes import (
    CDLL,
    POINTER,
    RTLD_GLOBAL,
    Structure,
    byref,
    c_bool,
    c_char_p,
    c_float,
    c_int,
    c_int16,
    c_int32,
    c_uint8,
    c_uint16,
    c_uint32,
    c_void_p,
)
from pathlib import Path

import cv2
import numpy as np

ALIGN_D2C_SW = 2
OB_PROP_LDP_BOOL = 2  # Viewer LDP enable. 공장 기본 켜짐.
OB_PROP_DEPTH_HOLEFILTER_BOOL = 17  # 펌웨어 HoleFilter. ON/OFF만.
OB_PROP_DEPTH_SOFT_FILTER_BOOL = 24  # Viewer NoiseRemovalFilter
OB_PROP_DEPTH_MAX_DIFF_INT = 40  # Viewer Min Diff
OB_PROP_DEPTH_MAX_SPECKLE_SIZE_INT = 41  # Viewer Max Size
OB_PROP_COLOR_AUTO_EXPOSURE_BOOL = 2000
OB_PROP_COLOR_EXPOSURE_INT = 2001
OB_PROP_COLOR_GAIN_INT = 2002
OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL = 2016
OB_PROP_DEPTH_EXPOSURE_INT = 2017
OB_PROP_DEPTH_GAIN_INT = 2018
OB_PERMISSION_WRITE = 2
OB_PERMISSION_ANY = 255
OB_FORMAT_MJPG = 5
OB_FORMAT_RGB = 22
OB_FORMAT_BGR = 23


class OBIntRange(Structure):
    _fields_ = [
        ("cur", c_int32),
        ("max", c_int32),
        ("min", c_int32),
        ("step", c_int32),
        ("def", c_int32),
    ]


class OBFloatRange(Structure):
    _fields_ = [
        ("cur", c_float),
        ("max", c_float),
        ("min", c_float),
        ("step", c_float),
        ("def", c_float),
    ]


def _jpeg_complete(raw: np.ndarray) -> bool:
    """SOI/EOI 없으면 USB에서 잘린 MJPEG. libjpeg 경고를 내지 않고 버린다."""
    if raw.size < 4 or int(raw[0]) != 0xFF or int(raw[1]) != 0xD8:
        return False
    tail = raw[-64:] if raw.size > 64 else raw
    return bool(np.any((tail[:-1] == 0xFF) & (tail[1:] == 0xD9)))
EP = POINTER(c_void_p)
_SDK_SO_NAMES = (
    "libOrbbecSDK.so.1.10",
    "libOrbbecSDK.so.1.10.27",
    "libOrbbecSDK.so",
)


def _sdk_so(sdk: Path) -> Path | None:
    for name in _SDK_SO_NAMES:
        path = sdk / name
        if path.is_file():
            return path
    return None


def resolve_sdk_dir() -> Path:
    env = os.environ.get("ORBBEC_SDK_DIR")
    if env:
        return Path(env).expanduser()
    downloads = Path.home() / "Downloads"
    candidates: list[Path] = []
    if downloads.is_dir():
        candidates.extend(sorted(downloads.glob("OrbbecViewer_*"), reverse=True))
    for candidate in candidates:
        if _sdk_so(candidate) is not None:
            return candidate
    return candidates[0] if candidates else downloads


def _require_sdk(sdk: Path) -> Path:
    so = _sdk_so(sdk)
    if so is None:
        raise RuntimeError(
            f"SDK 없음: {sdk}\n"
            "OrbbecViewer 폴더를 풀었으면 ORBBEC_SDK_DIR 로 그 경로를 지정하세요."
        )
    return so


def _bind(lib, name, restype, *argtypes):
    fn = getattr(lib, name)
    fn.restype = restype
    fn.argtypes = list(argtypes)
    return fn


class OBCameraIntrinsic(Structure):
    _fields_ = [
        ("fx", c_float),
        ("fy", c_float),
        ("cx", c_float),
        ("cy", c_float),
        ("width", c_int16),
        ("height", c_int16),
    ]


class OBCameraDistortion(Structure):
    _fields_ = [
        ("k1", c_float),
        ("k2", c_float),
        ("k3", c_float),
        ("k4", c_float),
        ("k5", c_float),
        ("k6", c_float),
        ("p1", c_float),
        ("p2", c_float),
    ]


class OBD2CTransform(Structure):
    _fields_ = [
        ("rot", c_float * 9),
        ("trans", c_float * 3),
    ]


class OBCameraParam(Structure):
    _fields_ = [
        ("depthIntrinsic", OBCameraIntrinsic),
        ("rgbIntrinsic", OBCameraIntrinsic),
        ("depthDistortion", OBCameraDistortion),
        ("rgbDistortion", OBCameraDistortion),
        ("transform", OBD2CTransform),
        ("isMirrored", c_bool),
    ]


def _chk(lib, err, where: str) -> None:
    errmsg = _bind(lib, "ob_error_message", c_char_p, c_void_p)
    delerr = _bind(lib, "ob_delete_error", None, c_void_p)
    if err:
        msg = errmsg(err) or b""
        delerr(err)
        raise RuntimeError(f"{where}: {msg.decode(errors='replace')}")


def _intrinsic_dict(intr: OBCameraIntrinsic) -> dict:
    return {
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "cx": float(intr.cx),
        "cy": float(intr.cy),
        "width": int(intr.width),
        "height": int(intr.height),
    }


def _dist_dict(dist: OBCameraDistortion) -> dict:
    return {
        "k1": float(dist.k1),
        "k2": float(dist.k2),
        "k3": float(dist.k3),
        "k4": float(dist.k4),
        "k5": float(dist.k5),
        "k6": float(dist.k6),
        "p1": float(dist.p1),
        "p2": float(dist.p2),
    }


def opencv_K(intr: OBCameraIntrinsic) -> list[list[float]]:
    return [
        [float(intr.fx), 0.0, float(intr.cx)],
        [0.0, float(intr.fy), float(intr.cy)],
        [0.0, 0.0, 1.0],
    ]


def opencv_dist(dist: OBCameraDistortion) -> list[float]:
    """OpenCV Brown-Conrady: k1, k2, p1, p2, k3, k4, k5, k6."""
    return [
        float(dist.k1),
        float(dist.k2),
        float(dist.p1),
        float(dist.p2),
        float(dist.k3),
        float(dist.k4),
        float(dist.k5),
        float(dist.k6),
    ]


def _rgb_ok(param: OBCameraParam) -> bool:
    rgb = param.rgbIntrinsic
    return float(rgb.fx) >= 1.0 and int(rgb.width) > 0 and int(rgb.height) > 0


def _payload_from_param(
    param: OBCameraParam,
    *,
    method: str,
    color_w: int,
    color_h: int,
    depth_w: int,
    depth_h: int,
) -> dict:
    rgb = param.rgbIntrinsic
    K = opencv_K(rgb)
    dist = opencv_dist(param.rgbDistortion)
    rgb_info = _intrinsic_dict(rgb)
    rgb_info["distortion"] = _dist_dict(param.rgbDistortion)
    depth_info = _intrinsic_dict(param.depthIntrinsic)
    depth_info["distortion"] = _dist_dict(param.depthDistortion)
    return {
        "image_size": [int(rgb.width), int(rgb.height)],
        "camera_matrix": K,
        "dist_coeffs": dist,
        "source": "orbbec_sdk_factory",
        "sdk": "Orbbec SDK v1.10",
        "method": method,
        "requested_color": [color_w, color_h],
        "requested_depth": [depth_w, depth_h],
        "factory_camera_matrix": K,
        "factory_dist_coeffs": dist,
        "rgb": rgb_info,
        "depth": depth_info,
    }


def read_factory_intrinsics(
    color_size: tuple[int, int] = (640, 480),
    depth_size: tuple[int, int] = (640, 400),
    sdk_dir: Path | None = None,
) -> dict:
    """장치에 저장된 공장 캘리브를 SDK로 읽는다. Viewer는 끄고 실행."""
    sdk = Path(sdk_dir) if sdk_dir is not None else resolve_sdk_dir()
    so = _require_sdk(sdk)
    color_w, color_h = (int(color_size[0]), int(color_size[1]))
    depth_w, depth_h = (int(depth_size[0]), int(depth_size[1]))
    cwd = os.getcwd()
    os.chdir(sdk)
    pipe = None
    lib = None
    stop = None
    delete = None
    param_list = None
    method = ""
    param: OBCameraParam | None = None
    try:
        lib = CDLL(f"./{so.name}", mode=RTLD_GLOBAL)
        create = _bind(lib, "ob_create_pipeline", c_void_p, EP)
        get_dev = _bind(lib, "ob_pipeline_get_device", c_void_p, c_void_p, EP)
        get_list = _bind(lib, "ob_device_get_calibration_camera_param_list", c_void_p, c_void_p, EP)
        list_count = _bind(lib, "ob_camera_param_list_count", c_uint32, c_void_p, EP)
        list_get = _bind(lib, "ob_camera_param_list_get_param", OBCameraParam, c_void_p, c_uint32, EP)
        list_del = _bind(lib, "ob_delete_camera_param_list", None, c_void_p, EP)
        get_cfg = _bind(lib, "ob_pipeline_get_config", c_void_p, c_void_p, EP)
        set_align = _bind(lib, "ob_config_set_align_mode", None, c_void_p, c_int, EP)
        set_d2c = _bind(
            lib, "ob_config_set_d2c_target_resolution", None, c_void_p, c_uint32, c_uint32, EP
        )
        start = _bind(lib, "ob_pipeline_start_with_config", None, c_void_p, c_void_p, EP)
        wait = _bind(lib, "ob_pipeline_wait_for_frameset", c_void_p, c_void_p, c_uint32, EP)
        del_frame = _bind(lib, "ob_delete_frame", None, c_void_p, EP)
        get_live = _bind(lib, "ob_pipeline_get_camera_param", OBCameraParam, c_void_p, EP)
        get_profile = _bind(
            lib,
            "ob_pipeline_get_camera_param_with_profile",
            OBCameraParam,
            c_void_p,
            c_uint32,
            c_uint32,
            c_uint32,
            c_uint32,
            EP,
        )
        stop = _bind(lib, "ob_pipeline_stop", None, c_void_p, EP)
        delete = _bind(lib, "ob_delete_pipeline", None, c_void_p, EP)

        err = c_void_p()
        pipe = create(byref(err))
        _chk(lib, err, "create_pipeline — 카메라 연결, Viewer 종료 확인")
        if not pipe:
            raise RuntimeError("pipeline 생성 실패")

        err = c_void_p()
        dev = get_dev(pipe, byref(err))
        if err:
            _chk(lib, err, "get_device")
        if dev:
            err = c_void_p()
            param_list = get_list(dev, byref(err))
            if err:
                try:
                    _chk(lib, err, "calibration_camera_param_list")
                except RuntimeError:
                    param_list = None
            if param_list:
                err = c_void_p()
                n = int(list_count(param_list, byref(err)))
                chosen = None
                for i in range(n):
                    err = c_void_p()
                    cand = list_get(param_list, i, byref(err))
                    if err:
                        continue
                    if not _rgb_ok(cand):
                        continue
                    rgb = cand.rgbIntrinsic
                    if int(rgb.width) == color_w and int(rgb.height) == color_h:
                        chosen = cand
                        break
                    if chosen is None:
                        chosen = cand
                if chosen is not None:
                    param = chosen
                    method = "ob_device_get_calibration_camera_param_list"

        if param is None or not _rgb_ok(param):
            err = c_void_p()
            cfg = get_cfg(pipe, byref(err))
            _chk(lib, err, "get_config")
            err = c_void_p()
            set_align(cfg, ALIGN_D2C_SW, byref(err))
            _chk(lib, err, "set_align")
            err = c_void_p()
            set_d2c(cfg, color_w, color_h, byref(err))
            if err:
                lib.ob_delete_error(err)
            err = c_void_p()
            start(pipe, cfg, byref(err))
            _chk(lib, err, "start — Viewer 끄기")
            for _ in range(8):
                err = c_void_p()
                fs = wait(pipe, 800, byref(err))
                if err:
                    lib.ob_delete_error(err)
                    continue
                if fs:
                    e = c_void_p()
                    del_frame(fs, byref(e))
                    break
            err = c_void_p()
            live = get_live(pipe, byref(err))
            _chk(lib, err, "get_camera_param")
            if _rgb_ok(live):
                param = live
                method = "ob_pipeline_get_camera_param"

        if param is None or not _rgb_ok(param):
            err = c_void_p()
            profile = get_profile(pipe, color_w, color_h, depth_w, depth_h, byref(err))
            _chk(lib, err, "get_camera_param_with_profile")
            param = profile
            method = "ob_pipeline_get_camera_param_with_profile"
    finally:
        if param_list and lib is not None:
            e = c_void_p()
            try:
                lib.ob_delete_camera_param_list(param_list, byref(e))
            except Exception:
                pass
        if pipe and lib is not None:
            e = c_void_p()
            try:
                if stop is not None:
                    stop(pipe, byref(e))
            except Exception:
                pass
            e = c_void_p()
            try:
                if delete is not None:
                    delete(pipe, byref(e))
            except Exception:
                pass
        os.chdir(cwd)

    if param is None or not _rgb_ok(param):
        rgb = param.rgbIntrinsic if param is not None else None
        fx = float(rgb.fx) if rgb is not None else 0.0
        wh = f"{int(rgb.width)}x{int(rgb.height)}" if rgb is not None else "0x0"
        raise RuntimeError(
            f"공장 intrinsic이 비정상입니다: fx={fx} size={wh}\n"
            "OpenNI Gemini는 스트림을 켠 뒤에만 K가 나옵니다. Viewer를 끄고 다시 실행하세요."
        )
    return _payload_from_param(
        param,
        method=method,
        color_w=color_w,
        color_h=color_h,
        depth_w=depth_w,
        depth_h=depth_h,
    )


class OrbbecV1:
    def __init__(
        self,
        width=640,
        height=480,
        sdk_dir: Path | None = None,
        *,
        noise_filter: bool | None = True,
        noise_min_diff: int | None = 10000,
        noise_max_size: int | None = 1,
        hole_filter: bool | None = True,
        ldp: bool = False,
    ):
        sdk = Path(sdk_dir) if sdk_dir is not None else resolve_sdk_dir()
        so = _require_sdk(sdk)
        cwd = os.getcwd()
        os.chdir(sdk)
        try:
            self.lib = CDLL(f"./{so.name}", mode=RTLD_GLOBAL)
            L = self.lib
            self._create = _bind(L, "ob_create_pipeline", c_void_p, EP)
            self._get_cfg = _bind(L, "ob_pipeline_get_config", c_void_p, c_void_p, EP)
            self._set_align = _bind(L, "ob_config_set_align_mode", None, c_void_p, c_int, EP)
            self._set_d2c = _bind(
                L, "ob_config_set_d2c_target_resolution", None, c_void_p, c_uint32, c_uint32, EP
            )
            self._get_dev = _bind(L, "ob_pipeline_get_device", c_void_p, c_void_p, EP)
            self._set_bool = _bind(
                L, "ob_device_set_bool_property", None, c_void_p, c_int, c_bool, EP
            )
            self._get_bool = _bind(
                L, "ob_device_get_bool_property", c_bool, c_void_p, c_int, EP
            )
            self._set_int = _bind(
                L, "ob_device_set_int_property", None, c_void_p, c_int, c_int32, EP
            )
            self._get_int = _bind(
                L, "ob_device_get_int_property", c_int32, c_void_p, c_int, EP
            )
            self._prop_ok = _bind(
                L, "ob_device_is_property_supported", c_bool, c_void_p, c_int, c_int, EP
            )
            self._get_int_range = _bind(
                L,
                "ob_device_get_int_property_range",
                OBIntRange,
                c_void_p,
                c_int,
                EP,
            )
            self._create_temporal = _bind(L, "ob_create_temporal_filter", c_void_p, EP)
            self._temporal_w_range = _bind(
                L, "ob_temporal_filter_get_weight_range", OBFloatRange, c_void_p, EP
            )
            self._temporal_d_range = _bind(
                L, "ob_temporal_filter_get_diffscale_range", OBFloatRange, c_void_p, EP
            )
            self._temporal_set_w = _bind(
                L, "ob_temporal_filter_set_weight_value", None, c_void_p, c_float, EP
            )
            self._temporal_set_d = _bind(
                L, "ob_temporal_filter_set_diffscale_value", None, c_void_p, c_float, EP
            )
            self._create_holefill = _bind(L, "ob_create_holefilling_filter", c_void_p, EP)
            self._holefill_set_mode = _bind(
                L, "ob_holefilling_filter_set_mode", None, c_void_p, c_int, EP
            )
            self._filter_enable = _bind(L, "ob_filter_enable", None, c_void_p, c_bool, EP)
            self._filter_reset = _bind(L, "ob_filter_reset", None, c_void_p, EP)
            self._filter_process = _bind(
                L, "ob_filter_process", c_void_p, c_void_p, c_void_p, EP
            )
            self._del_filter = _bind(L, "ob_delete_filter", None, c_void_p, EP)
            self._start = _bind(L, "ob_pipeline_start_with_config", None, c_void_p, c_void_p, EP)
            self._wait = _bind(
                L, "ob_pipeline_wait_for_frameset", c_void_p, c_void_p, c_uint32, EP
            )
            self._color_f = _bind(L, "ob_frameset_color_frame", c_void_p, c_void_p, EP)
            self._depth_f = _bind(L, "ob_frameset_depth_frame", c_void_p, c_void_p, EP)
            self._w = _bind(L, "ob_video_frame_width", c_uint32, c_void_p, EP)
            self._h = _bind(L, "ob_video_frame_height", c_uint32, c_void_p, EP)
            self._fmt = _bind(L, "ob_frame_format", c_int, c_void_p, EP)
            self._nbytes = _bind(L, "ob_frame_data_size", c_uint32, c_void_p, EP)
            self._data = _bind(L, "ob_frame_data", c_void_p, c_void_p, EP)
            self._scale = _bind(L, "ob_depth_frame_get_value_scale", c_float, c_void_p, EP)
            self._del_frame = _bind(L, "ob_delete_frame", None, c_void_p, EP)
            self._stop = _bind(L, "ob_pipeline_stop", None, c_void_p, EP)
            self._del_pipe = _bind(L, "ob_delete_pipeline", None, c_void_p, EP)
            self._set_log = _bind(L, "ob_set_logger_severity", None, c_int, EP)
            self._errmsg = _bind(L, "ob_error_message", c_char_p, c_void_p)
            self._delerr = _bind(L, "ob_delete_error", None, c_void_p)

            err = c_void_p()
            self.pipe = self._create(byref(err))
            self._chk(err, "create_pipeline")
            err = c_void_p()
            self.cfg = self._get_cfg(self.pipe, byref(err))
            self._chk(err, "get_config")
            err = c_void_p()
            self._set_align(self.cfg, ALIGN_D2C_SW, byref(err))
            self._chk(err, "align")
            err = c_void_p()
            self._set_d2c(self.cfg, width, height, byref(err))
            if err:
                self._delerr(err)
            self._apply_ldp(ldp)
            self._apply_noise_filter(noise_filter, noise_min_diff, noise_max_size)
            self._apply_hole_filter(hole_filter)
            self._temporal = None
            self._temporal_on = False
            self._holefill = None
            self._holefill_on = False
            err = c_void_p()
            self._start(self.pipe, self.cfg, byref(err))
            self._chk(err, "start — Viewer 끄기")
            self._init_temporal_filter()
            self._init_holefill_filter()
        finally:
            os.chdir(cwd)
        self._last_bgr = None
        self._last_depth = None

    def _chk(self, err, where):
        if err:
            msg = self._errmsg(err) or b""
            self._delerr(err)
            raise RuntimeError(f"{where}: {msg.decode(errors='replace')}")

    def _prop_supported(self, dev, prop_id: int, permission: int = OB_PERMISSION_WRITE) -> bool:
        err = c_void_p()
        ok = bool(self._prop_ok(dev, prop_id, permission, byref(err)))
        if err:
            self._delerr(err)
            return False
        return ok

    def _set_bool_prop(self, dev, prop_id: int, value: bool, name: str) -> bool:
        if not self._prop_supported(dev, prop_id):
            print(f"{name}: 이 장치는 지원하지 않음")
            return False
        err = c_void_p()
        self._set_bool(dev, prop_id, bool(value), byref(err))
        if err:
            msg = self._errmsg(err) or b""
            self._delerr(err)
            print(f"{name}: 설정 실패 ({msg.decode(errors='replace')})")
            return False
        return True

    def _set_int_prop(self, dev, prop_id: int, value: int, name: str) -> bool:
        if not self._prop_supported(dev, prop_id):
            print(f"{name}: 이 장치는 지원하지 않음")
            return False
        err = c_void_p()
        self._set_int(dev, prop_id, int(value), byref(err))
        if err:
            msg = self._errmsg(err) or b""
            self._delerr(err)
            print(f"{name}: 설정 실패 ({msg.decode(errors='replace')})")
            return False
        return True

    def _get_bool_prop(self, dev, prop_id: int) -> bool | None:
        err = c_void_p()
        value = bool(self._get_bool(dev, prop_id, byref(err)))
        if err:
            self._delerr(err)
            return None
        return value

    def _get_int_prop(self, dev, prop_id: int) -> int | None:
        err = c_void_p()
        value = int(self._get_int(dev, prop_id, byref(err)))
        if err:
            self._delerr(err)
            return None
        return value

    def _apply_ldp(self, enable: bool) -> None:
        """Laser Distance Protection. 켜면 가까울 때 프로젝터를 끔. 기본 끔."""
        dev = self._device_ptr()
        if not dev:
            print("LDP: device 없음")
            return
        ok = self._set_bool_prop(dev, OB_PROP_LDP_BOOL, bool(enable), "LDP")
        if not ok:
            return
        cur = self._get_bool_prop(dev, OB_PROP_LDP_BOOL)
        print(f"LDP on={cur}")

    def _device_ptr(self):
        err = c_void_p()
        dev = self._get_dev(self.pipe, byref(err))
        if err:
            self._delerr(err)
            return None
        return dev

    def get_noise_filter(self) -> dict:
        dev = self._device_ptr()
        if not dev:
            return {"on": None, "min_diff": None, "max_size": None}
        return {
            "on": self._get_bool_prop(dev, OB_PROP_DEPTH_SOFT_FILTER_BOOL),
            "min_diff": self._get_int_prop(dev, OB_PROP_DEPTH_MAX_DIFF_INT),
            "max_size": self._get_int_prop(dev, OB_PROP_DEPTH_MAX_SPECKLE_SIZE_INT),
        }

    def set_noise_filter(
        self,
        enable: bool | None = True,
        min_diff: int | None = None,
        max_size: int | None = None,
        *,
        quiet: bool = False,
    ) -> dict:
        self._apply_noise_filter(enable, min_diff, max_size, quiet=quiet)
        return self.get_noise_filter()

    def _apply_noise_filter(
        self,
        enable: bool | None,
        min_diff: int | None,
        max_size: int | None,
        *,
        quiet: bool = False,
    ) -> None:
        if enable is None and min_diff is None and max_size is None:
            return
        dev = self._device_ptr()
        if not dev:
            print("NoiseRemovalFilter: device 없음")
            return
        on = False if enable is False else True
        if enable is not None or min_diff is not None or max_size is not None:
            self._set_bool_prop(dev, OB_PROP_DEPTH_SOFT_FILTER_BOOL, on, "NoiseRemovalFilter")
        if on and min_diff is not None:
            self._set_int_prop(dev, OB_PROP_DEPTH_MAX_DIFF_INT, min_diff, "Min Diff")
        if on and max_size is not None:
            self._set_int_prop(
                dev, OB_PROP_DEPTH_MAX_SPECKLE_SIZE_INT, max_size, "Max Size"
            )
        if quiet:
            return
        cur = self.get_noise_filter()
        print(
            "NoiseRemovalFilter "
            f"on={cur['on']}  min_diff={cur['min_diff']}  max_size={cur['max_size']}"
        )

    def _int_range(self, prop_id: int, fallback: tuple[int, int]) -> tuple[int, int]:
        dev = self._device_ptr()
        if not dev:
            return fallback
        err = c_void_p()
        try:
            rng = self._get_int_range(dev, prop_id, byref(err))
        except Exception:
            if err:
                self._delerr(err)
            return fallback
        if err:
            self._delerr(err)
            return fallback
        lo, hi = int(rng.min), int(rng.max)
        if hi <= lo:
            return fallback
        return lo, hi

    def has_property(self, prop_id: int) -> bool:
        dev = self._device_ptr()
        if not dev:
            return False
        return self._prop_supported(dev, prop_id, OB_PERMISSION_ANY)

    def get_hole_filter(self) -> bool | None:
        dev = self._device_ptr()
        if not dev:
            return None
        return self._get_bool_prop(dev, OB_PROP_DEPTH_HOLEFILTER_BOOL)

    def set_hole_filter(self, enable: bool, *, quiet: bool = False) -> bool | None:
        self._apply_hole_filter(bool(enable), quiet=quiet)
        return self.get_hole_filter()

    def _apply_hole_filter(self, enable: bool | None, *, quiet: bool = False) -> None:
        if enable is None:
            return
        dev = self._device_ptr()
        if not dev:
            print("HoleFilter: device 없음")
            return
        self._set_bool_prop(dev, OB_PROP_DEPTH_HOLEFILTER_BOOL, bool(enable), "HoleFilter")
        if quiet:
            return
        print(f"HoleFilter on={self.get_hole_filter()}")

    def get_depth_exposure(self) -> dict:
        dev = self._device_ptr()
        if not dev:
            return {"ae": None, "exposure": None, "gain": None}
        ae = None
        if self._prop_supported(dev, OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL, OB_PERMISSION_ANY):
            ae = self._get_bool_prop(dev, OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL)
        return {
            "ae": ae,
            "exposure": self._get_int_prop(dev, OB_PROP_DEPTH_EXPOSURE_INT),
            "gain": self._get_int_prop(dev, OB_PROP_DEPTH_GAIN_INT),
        }

    def set_depth_exposure(
        self,
        *,
        ae: bool | None = None,
        exposure: int | None = None,
        gain: int | None = None,
        quiet: bool = False,
    ) -> dict:
        dev = self._device_ptr()
        if not dev:
            print("Depth exposure: device 없음")
            return self.get_depth_exposure()
        has_ae = self._prop_supported(
            dev, OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL, OB_PERMISSION_WRITE
        )
        if ae is not None and has_ae:
            self._set_bool_prop(dev, OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL, bool(ae), "Depth AE")
        ae_on = False
        if has_ae:
            ae_on = bool(self._get_bool_prop(dev, OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL))
        if ae_on:
            if not quiet:
                cur = self.get_depth_exposure()
                print(f"Depth AE on  exp={cur['exposure']}  gain={cur['gain']}")
            return self.get_depth_exposure()
        if exposure is not None:
            self._set_int_prop(dev, OB_PROP_DEPTH_EXPOSURE_INT, exposure, "Depth Exposure")
        if gain is not None:
            self._set_int_prop(dev, OB_PROP_DEPTH_GAIN_INT, gain, "Depth Gain")
        if not quiet:
            cur = self.get_depth_exposure()
            print(
                f"Depth AE={cur['ae']}  exp={cur['exposure']}  gain={cur['gain']}"
            )
        return self.get_depth_exposure()

    def get_color_exposure(self) -> dict:
        dev = self._device_ptr()
        if not dev:
            return {"ae": None, "exposure": None, "gain": None}
        ae = None
        if self._prop_supported(dev, OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, OB_PERMISSION_ANY):
            ae = self._get_bool_prop(dev, OB_PROP_COLOR_AUTO_EXPOSURE_BOOL)
        return {
            "ae": ae,
            "exposure": self._get_int_prop(dev, OB_PROP_COLOR_EXPOSURE_INT),
            "gain": self._get_int_prop(dev, OB_PROP_COLOR_GAIN_INT),
        }

    def set_color_exposure(
        self,
        *,
        ae: bool | None = None,
        exposure: int | None = None,
        gain: int | None = None,
        quiet: bool = False,
    ) -> dict:
        dev = self._device_ptr()
        if not dev:
            print("Color exposure: device 없음")
            return self.get_color_exposure()
        has_ae = self._prop_supported(
            dev, OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, OB_PERMISSION_WRITE
        )
        if ae is not None and has_ae:
            self._set_bool_prop(dev, OB_PROP_COLOR_AUTO_EXPOSURE_BOOL, bool(ae), "Color AE")
        ae_on = False
        if has_ae:
            ae_on = bool(self._get_bool_prop(dev, OB_PROP_COLOR_AUTO_EXPOSURE_BOOL))
        if ae_on:
            if not quiet:
                cur = self.get_color_exposure()
                print(f"Color AE on  exp={cur['exposure']}  gain={cur['gain']}")
            return self.get_color_exposure()
        if exposure is not None:
            self._set_int_prop(dev, OB_PROP_COLOR_EXPOSURE_INT, exposure, "Color Exposure")
        if gain is not None:
            self._set_int_prop(dev, OB_PROP_COLOR_GAIN_INT, gain, "Color Gain")
        if not quiet:
            cur = self.get_color_exposure()
            print(
                f"Color AE={cur['ae']}  exp={cur['exposure']}  gain={cur['gain']}"
            )
        return self.get_color_exposure()

    def _init_temporal_filter(self) -> None:
        self._temporal = None
        self._temporal_on = False
        self._temporal_weight = 0.4
        self._temporal_diff = 0.1
        self._temporal_weight_range = (0.0, 1.0)
        self._temporal_diff_range = (0.0, 1.0)
        err = c_void_p()
        filt = self._create_temporal(byref(err))
        if err:
            msg = self._errmsg(err) or b""
            self._delerr(err)
            print(f"TemporalFilter: 생성 실패 ({msg.decode(errors='replace')})")
            return
        if not filt:
            print("TemporalFilter: 생성 실패")
            return
        self._temporal = filt
        e = c_void_p()
        wr = self._temporal_w_range(filt, byref(e))
        if e:
            self._delerr(e)
        else:
            lo, hi = float(wr.min), float(wr.max)
            if hi > lo:
                self._temporal_weight_range = (lo, hi)
                default = float(getattr(wr, "def"))
                self._temporal_weight = (
                    default if default == default else (lo + hi) * 0.5
                )
        e = c_void_p()
        dr = self._temporal_d_range(filt, byref(e))
        if e:
            self._delerr(e)
        else:
            lo, hi = float(dr.min), float(dr.max)
            if hi > lo:
                self._temporal_diff_range = (lo, hi)
                default = float(getattr(dr, "def"))
                self._temporal_diff = (
                    default if default == default else (lo + hi) * 0.5
                )
        e = c_void_p()
        self._filter_enable(filt, False, byref(e))
        if e:
            self._delerr(e)
        print(
            "TemporalFilter off  "
            f"weight={self._temporal_weight:.3f} "
            f"({self._temporal_weight_range[0]:.3f}-{self._temporal_weight_range[1]:.3f})  "
            f"diff={self._temporal_diff:.3f} "
            f"({self._temporal_diff_range[0]:.3f}-{self._temporal_diff_range[1]:.3f})"
        )

    def get_temporal_filter(self) -> dict:
        return {
            "ok": self._temporal is not None,
            "on": self._temporal_on,
            "weight": self._temporal_weight,
            "diffscale": self._temporal_diff,
            "weight_range": self._temporal_weight_range,
            "diffscale_range": self._temporal_diff_range,
        }

    def set_temporal_filter(
        self,
        enable: bool | None = None,
        weight: float | None = None,
        diffscale: float | None = None,
        *,
        quiet: bool = False,
    ) -> dict:
        filt = self._temporal
        if filt is None:
            return self.get_temporal_filter()
        wlo, whi = self._temporal_weight_range
        dlo, dhi = self._temporal_diff_range
        if weight is not None:
            self._temporal_weight = float(np.clip(weight, wlo, whi))
            e = c_void_p()
            self._temporal_set_w(filt, c_float(self._temporal_weight), byref(e))
            if e:
                self._delerr(e)
        if diffscale is not None:
            self._temporal_diff = float(np.clip(diffscale, dlo, dhi))
            e = c_void_p()
            self._temporal_set_d(filt, c_float(self._temporal_diff), byref(e))
            if e:
                self._delerr(e)
        if enable is not None:
            on = bool(enable)
            if on != self._temporal_on and not on:
                e = c_void_p()
                self._filter_reset(filt, byref(e))
                if e:
                    self._delerr(e)
            self._temporal_on = on
            e = c_void_p()
            self._filter_enable(filt, on, byref(e))
            if e:
                self._delerr(e)
        if not quiet:
            cur = self.get_temporal_filter()
            print(
                f"TemporalFilter on={cur['on']}  "
                f"weight={cur['weight']:.3f}  diff={cur['diffscale']:.3f}"
            )
        return self.get_temporal_filter()

    def _init_holefill_filter(self) -> None:
        self._holefill = None
        self._holefill_on = False
        self._holefill_mode = 1
        err = c_void_p()
        filt = self._create_holefill(byref(err))
        if err:
            msg = self._errmsg(err) or b""
            self._delerr(err)
            print(f"HoleFillingFilter: 생성 실패 ({msg.decode(errors='replace')})")
            return
        if not filt:
            print("HoleFillingFilter: 생성 실패")
            return
        self._holefill = filt
        e = c_void_p()
        self._holefill_set_mode(filt, int(self._holefill_mode), byref(e))
        if e:
            self._delerr(e)
        e = c_void_p()
        self._filter_enable(filt, False, byref(e))
        if e:
            self._delerr(e)
        print("HoleFillingFilter off  mode=Nearest")

    def set_hole_filling(
        self,
        enable: bool | None = None,
        mode: int | None = None,
        *,
        quiet: bool = False,
    ) -> dict:
        filt = getattr(self, "_holefill", None)
        if filt is None:
            return {"ok": False, "on": False, "mode": None}
        if mode is not None:
            self._holefill_mode = int(np.clip(mode, 0, 2))
            e = c_void_p()
            self._holefill_set_mode(filt, self._holefill_mode, byref(e))
            if e:
                self._delerr(e)
        if enable is not None:
            self._holefill_on = bool(enable)
            e = c_void_p()
            self._filter_enable(filt, self._holefill_on, byref(e))
            if e:
                self._delerr(e)
        names = ("Top", "Nearest", "Farest")
        if not quiet:
            print(
                f"HoleFillingFilter on={self._holefill_on}  "
                f"mode={names[self._holefill_mode]}"
            )
        return self.get_hole_filling()

    def get_hole_filling(self) -> dict:
        ok = getattr(self, "_holefill", None) is not None
        return {
            "ok": ok,
            "on": bool(getattr(self, "_holefill_on", False)),
            "mode": int(getattr(self, "_holefill_mode", 1)),
        }

    def _run_host_filters(self, depth):
        chain = (
            (getattr(self, "_temporal", None), getattr(self, "_temporal_on", False)),
            (getattr(self, "_holefill", None), getattr(self, "_holefill_on", False)),
        )
        for filt, on in chain:
            if not on or filt is None:
                continue
            e = c_void_p()
            out = self._filter_process(filt, depth, byref(e))
            if e:
                self._delerr(e)
                continue
            if out:
                e = c_void_p()
                self._del_frame(depth, byref(e))
                depth = out
        return depth

    def depth_ui_caps(self) -> dict:
        exp_lo, exp_hi = self._int_range(OB_PROP_DEPTH_EXPOSURE_INT, (0, 10000))
        gain_lo, gain_hi = self._int_range(OB_PROP_DEPTH_GAIN_INT, (0, 9999))
        cexp_lo, cexp_hi = self._int_range(OB_PROP_COLOR_EXPOSURE_INT, (1, 10000))
        cgain_lo, cgain_hi = self._int_range(OB_PROP_COLOR_GAIN_INT, (0, 255))
        diff_lo, diff_hi = self._int_range(OB_PROP_DEPTH_MAX_DIFF_INT, (1, 10000))
        size_lo, size_hi = self._int_range(OB_PROP_DEPTH_MAX_SPECKLE_SIZE_INT, (1, 1000))
        return {
            "hole": self.has_property(OB_PROP_DEPTH_HOLEFILTER_BOOL),
            "ae": self.has_property(OB_PROP_DEPTH_AUTO_EXPOSURE_BOOL),
            "exposure": self.has_property(OB_PROP_DEPTH_EXPOSURE_INT),
            "gain": self.has_property(OB_PROP_DEPTH_GAIN_INT),
            "color_ae": self.has_property(OB_PROP_COLOR_AUTO_EXPOSURE_BOOL),
            "color_exposure": self.has_property(OB_PROP_COLOR_EXPOSURE_INT),
            "color_gain": self.has_property(OB_PROP_COLOR_GAIN_INT),
            "temporal": self._temporal is not None,
            "holefill": self._holefill is not None,
            "exposure_range": (exp_lo, exp_hi),
            "gain_range": (gain_lo, gain_hi),
            "color_exposure_range": (cexp_lo, cexp_hi),
            "color_gain_range": (cgain_lo, cgain_hi),
            "min_diff_range": (max(1, diff_lo), max(diff_lo + 1, diff_hi)),
            "max_size_range": (max(1, size_lo), max(size_lo + 1, size_hi)),
        }

    def grab(self, timeout_ms=1000):
        err = c_void_p()
        fs = self._wait(self.pipe, timeout_ms, byref(err))
        if err:
            self._delerr(err)
        if not fs:
            return self._last_bgr, self._last_depth
        e = c_void_p()
        color = self._color_f(fs, byref(e))
        e = c_void_p()
        depth = self._depth_f(fs, byref(e))
        if color:
            img = self._color_to_bgr(color)
            if img is not None:
                self._last_bgr = img
            e = c_void_p()
            self._del_frame(color, byref(e))
        if depth:
            depth = self._run_host_filters(depth)
            self._last_depth = self._depth_to_mm(depth)
            e = c_void_p()
            self._del_frame(depth, byref(e))
        e = c_void_p()
        self._del_frame(fs, byref(e))
        return self._last_bgr, self._last_depth

    def _color_to_bgr(self, frame):
        e = c_void_p()
        w, h = int(self._w(frame, byref(e))), int(self._h(frame, byref(e)))
        e = c_void_p()
        fmt = int(self._fmt(frame, byref(e)))
        e = c_void_p()
        n = int(self._nbytes(frame, byref(e)))
        e = c_void_p()
        ptr = self._data(frame, byref(e))
        raw = np.ctypeslib.as_array((c_uint8 * n).from_address(ptr)).copy()
        if fmt == OB_FORMAT_MJPG:
            if not _jpeg_complete(raw):
                return None
            return cv2.imdecode(raw, cv2.IMREAD_COLOR)
        if fmt == OB_FORMAT_RGB:
            return cv2.cvtColor(raw.reshape(h, w, 3), cv2.COLOR_RGB2BGR)
        if fmt == OB_FORMAT_BGR:
            return raw.reshape(h, w, 3).copy()
        return None

    def _depth_to_mm(self, frame):
        e = c_void_p()
        w, h = int(self._w(frame, byref(e))), int(self._h(frame, byref(e)))
        e = c_void_p()
        n = int(self._nbytes(frame, byref(e)))
        e = c_void_p()
        scale = float(self._scale(frame, byref(e)))
        e = c_void_p()
        ptr = self._data(frame, byref(e))
        raw = np.ctypeslib.as_array((c_uint16 * (n // 2)).from_address(ptr)).copy()
        return raw.reshape(h, w).astype(np.float32) * scale

    def _quiet_log(self) -> None:
        """stop 때 libusb cancel Success 경고가 쏟아지지 않게 한다."""
        fn = getattr(self, "_set_log", None)
        if fn is None:
            return
        err = c_void_p()
        try:
            fn(3, byref(err))  # OB_LOG_SEVERITY_ERROR
        except TypeError:
            try:
                fn(3)
            except Exception:
                return
        if err:
            self._delerr(err)

    def _drain_frames(self) -> None:
        pipe = getattr(self, "pipe", None)
        if not pipe:
            return
        for _ in range(16):
            e = c_void_p()
            fs = self._wait(pipe, 1, byref(e))
            if e:
                self._delerr(e)
            if not fs:
                return
            e = c_void_p()
            self._del_frame(fs, byref(e))

    def close(self):
        self._temporal_on = False
        self._holefill_on = False
        self._quiet_log()
        host = [
            getattr(self, "_temporal", None),
            getattr(self, "_holefill", None),
        ]
        for filt in host:
            if not filt:
                continue
            e = c_void_p()
            try:
                self._filter_enable(filt, False, byref(e))
            except Exception:
                pass
            if e:
                self._delerr(e)
            e = c_void_p()
            try:
                self._filter_reset(filt, byref(e))
            except Exception:
                pass
            if e:
                self._delerr(e)
        self._drain_frames()
        if getattr(self, "pipe", None):
            e = c_void_p()
            self._stop(self.pipe, byref(e))
            if e:
                self._delerr(e)
        for name in ("_temporal", "_holefill"):
            filt = getattr(self, name, None)
            if not filt:
                continue
            e = c_void_p()
            try:
                self._del_filter(filt, byref(e))
            except Exception:
                pass
            if e:
                try:
                    self._delerr(e)
                except Exception:
                    pass
            setattr(self, name, None)
        if getattr(self, "pipe", None):
            e = c_void_p()
            self._del_pipe(self.pipe, byref(e))
            if e:
                try:
                    self._delerr(e)
                except Exception:
                    pass
            self.pipe = None
