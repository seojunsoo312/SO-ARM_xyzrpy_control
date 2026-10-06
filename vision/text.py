"""cv2.putText 대신 쓰는 한글 글자 그리기.

Hershey 글꼴에는 한글이 없어 `?`로 찍힌다. ASCII 는 cv2.putText 그대로,
한글이 섞이면 Noto Sans CJK KR 글자 모양을 PIL 로 한 번 만들어 두고 그 자리에 섞는다.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

NOTO_DIR = Path("/usr/share/fonts/opentype/noto")
# NotoSansCJK-*.ttc 안의 순서: 0 JP, 1 KR, 2 SC, 3 TC, 4 HK
_KR_INDEX = 1
# Hershey SIMPLEX scale 1.0 과 높이가 비슷해지는 픽셀 크기
_PX_PER_SCALE = 30

_FONT_CACHE: dict[tuple[int, bool], object] = {}


def kr_font(size: int, *, bold: bool = False):
    """Noto Sans CJK KR. 없으면 None."""
    from PIL import ImageFont

    key = (int(size), bool(bold))
    if key in _FONT_CACHE:
        return _FONT_CACHE[key]
    font = None
    for name in (("NotoSansCJK-Bold.ttc",) if bold else ()) + ("NotoSansCJK-Regular.ttc",):
        path = NOTO_DIR / name
        if not path.is_file():
            continue
        for index in (_KR_INDEX, 0):
            try:
                font = ImageFont.truetype(str(path), size=int(size), index=index)
                break
            except OSError:
                continue
        if font is not None:
            break
    _FONT_CACHE[key] = font
    return font


@lru_cache(maxsize=512)
def _text_mask(text: str, size: int, bold: bool) -> tuple[np.ndarray, int, int] | None:
    """글자 모양(0~1)과 기준선에서 왼쪽 위까지의 거리. 같은 문구는 다시 그리지 않는다."""
    from PIL import Image, ImageDraw

    font = kr_font(size, bold=bold)
    if font is None:
        return None
    left, top, right, bottom = font.getbbox(text, anchor="ls")
    im = Image.new("L", (max(1, right - left), max(1, bottom - top)), 0)
    ImageDraw.Draw(im).text((-left, -top), text, font=font, fill=255, anchor="ls")
    return np.asarray(im, dtype=np.float32) / 255.0, left, top


def put_text(
    img: np.ndarray,
    text: str,
    org: tuple[int, int],
    scale: float,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> np.ndarray:
    """cv2.putText(img, text, org, FONT_HERSHEY_SIMPLEX, scale, color, thickness, LINE_AA) 와 같은 자리.

    org 는 putText 처럼 글자 왼쪽 아래 기준선이다. img 를 그 자리에서 고친다.
    """
    mask = None
    if not text.isascii():
        mask = _text_mask(text, max(8, round(_PX_PER_SCALE * scale)), thickness >= 2)
    if mask is None:
        cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness, cv2.LINE_AA)
        return img

    alpha, left, top = mask
    h, w = img.shape[:2]
    x0, y0 = int(org[0]) + left, int(org[1]) + top
    mh, mw = alpha.shape
    cx0, cy0 = max(0, x0), max(0, y0)
    cx1, cy1 = min(w, x0 + mw), min(h, y0 + mh)
    if cx1 <= cx0 or cy1 <= cy0:
        return img
    a = alpha[cy0 - y0 : cy1 - y0, cx0 - x0 : cx1 - x0]
    roi = img[cy0:cy1, cx0:cx1]
    if img.ndim == 2:
        col = np.float32(color[0] if isinstance(color, (tuple, list)) else color)
    else:
        a = a[..., None]
        col = np.asarray(color[: roi.shape[2]], dtype=np.float32)
    roi[...] = (roi * (1.0 - a) + col * a).astype(img.dtype)
    return img
