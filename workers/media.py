"""Reading media files: type detection, hashing, decoding (JPEG/HEIC/RAW), EXIF, XMP.

RAW files are decoded from their embedded JPEG preview whenever one exists:
it's 10-50x faster than demosaicing and more than enough for a 512 px
thumbnail and a 224 px CLIP input. Full decode is the fallback.
"""

from __future__ import annotations

import hashlib
import io
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from fractions import Fraction
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps

try:
    import pillow_heif

    pillow_heif.register_heif_opener()
except ImportError:  # pragma: no cover
    pillow_heif = None  # type: ignore[assignment]

Image.MAX_IMAGE_PIXELS = 400_000_000

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".heic", ".heif", ".tif", ".tiff", ".bmp"}
RAW_EXTS = {
    ".cr2", ".cr3", ".nef", ".nrw", ".arw", ".srf", ".sr2", ".dng", ".raf",
    ".orf", ".rw2", ".pef", ".srw", ".x3f", ".3fr", ".iiq", ".rwl",
}  # fmt: skip
VIDEO_EXTS = {".mp4", ".mov", ".m4v", ".avi", ".mkv", ".mts", ".m2ts", ".3gp", ".webm"}
MEDIA_EXTS = IMAGE_EXTS | RAW_EXTS | VIDEO_EXTS

FULL_HASH_LIMIT = 64 * 1024 * 1024
SAMPLE = 4 * 1024 * 1024


def media_kind(path: Path) -> str | None:
    ext = path.suffix.lower()
    if ext in RAW_EXTS:
        return "raw"
    if ext in IMAGE_EXTS:
        return "image"
    if ext in VIDEO_EXTS:
        return "video"
    return None


def file_format(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    return {"jpg": "jpeg", "heif": "heic", "tif": "tiff"}.get(ext, ext)


def hash_file(path: Path, size: int | None = None) -> str:
    """blake2b of the whole file, or of size + head/middle/tail samples for big files
    (videos). Only used for change/move detection, not security."""
    size = path.stat().st_size if size is None else size
    h = hashlib.blake2b(digest_size=16)
    with path.open("rb") as f:
        if size <= FULL_HASH_LIMIT:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        else:
            h.update(str(size).encode())
            for off in (0, size // 2, max(0, size - SAMPLE)):
                f.seek(off)
                h.update(f.read(SAMPLE))
    return h.hexdigest()


# ---------------------------------------------------------------- decoding

_ORIENT_OPS = {
    2: [Image.Transpose.FLIP_LEFT_RIGHT],
    3: [Image.Transpose.ROTATE_180],
    4: [Image.Transpose.FLIP_TOP_BOTTOM],
    5: [Image.Transpose.TRANSPOSE],
    6: [Image.Transpose.ROTATE_270],
    7: [Image.Transpose.TRANSVERSE],
    8: [Image.Transpose.ROTATE_90],
}


def apply_orientation(im: Image.Image, orientation: int | None) -> Image.Image:
    for op in _ORIENT_OPS.get(orientation or 1, []):
        im = im.transpose(op)
    return im


# LibRaw's `flip` code -> PIL transpose, for previews that carry no orientation of their own.
_LIBRAW_FLIP = {
    3: Image.Transpose.ROTATE_180,
    5: Image.Transpose.ROTATE_90,
    6: Image.Transpose.ROTATE_270,
}


def open_raw(path: Path, max_side: int | None = None, full: bool = False) -> Image.Image:
    import rawpy

    with rawpy.imread(str(path)) as raw:
        flip = raw.sizes.flip
        if not full:
            try:
                thumb = raw.extract_thumb()
                if thumb.format == rawpy.ThumbFormat.JPEG:
                    im = Image.open(io.BytesIO(thumb.data))
                    if max_side:
                        im.draft("RGB", (max_side, max_side))
                    im.load()
                    if im.getexif().get(0x0112):
                        return ImageOps.exif_transpose(im)
                    return im.transpose(_LIBRAW_FLIP[flip]) if flip in _LIBRAW_FLIP else im
                if thumb.format == rawpy.ThumbFormat.BITMAP:
                    bmp = Image.fromarray(thumb.data)  # type: ignore[arg-type]
                    return bmp.transpose(_LIBRAW_FLIP[flip]) if flip in _LIBRAW_FLIP else bmp
            except (rawpy.LibRawNoThumbnailError, rawpy.LibRawUnsupportedThumbnailError):
                pass
        # postprocess() already applies the flip
        rgb = raw.postprocess(half_size=True, use_camera_wb=True, no_auto_bright=False)
        return Image.fromarray(rgb)


def open_image(path: Path, max_side: int | None = None) -> Image.Image:
    """Decode any supported still image to an upright RGB PIL image.

    `max_side` lets JPEG use DCT-domain downscaling (Image.draft), which makes
    thumbnailing a 24 MP JPEG several times faster.
    """
    kind = media_kind(path)
    if kind == "raw":
        im = open_raw(path, max_side)
    else:
        im = Image.open(path)
        if max_side and im.format == "JPEG":
            im.draft("RGB", (max_side, max_side))
        im = ImageOps.exif_transpose(im)
    if im.mode != "RGB":
        im = im.convert("RGB")
    return im


def make_thumbnail(im: Image.Image, size: int) -> Image.Image:
    im = im.copy()
    im.thumbnail((size, size), Image.Resampling.LANCZOS)
    return im


def sharpness(im: Image.Image) -> float:
    """Variance of the Laplacian on a fixed-size grayscale copy: higher is sharper."""
    import cv2
    import numpy as np

    g = np.asarray(im.convert("L").resize((512, 512)), dtype=np.float32)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def perceptual_hash(im: Image.Image) -> str:
    import imagehash

    return str(imagehash.phash(im))


# ---------------------------------------------------------------- EXIF


@dataclass
class Exif:
    taken_at: datetime | None = None
    tz_offset_min: int | None = None
    lat: float | None = None
    lon: float | None = None
    camera: str | None = None
    lens: str | None = None
    focal_length: float | None = None
    focal_length_35mm: float | None = None
    aperture: float | None = None
    iso: int | None = None
    shutter: str | None = None
    exposure_s: float | None = None
    width: int | None = None
    height: int | None = None
    orientation: int | None = None
    keywords: list[str] = field(default_factory=list)


def _ratio(v: Any) -> float | None:
    try:
        if hasattr(v, "num") and hasattr(v, "den"):
            return float(v.num) / float(v.den) if v.den else None
        if isinstance(v, tuple) and len(v) == 2:
            return float(v[0]) / float(v[1]) if v[1] else None
        if isinstance(v, (tuple, list)) and len(v) == 1:
            return _ratio(v[0])
        return float(v)
    except (TypeError, ValueError, ZeroDivisionError):
        return None


def _first(tag: Any) -> Any:
    if tag is None:
        return None
    vals = getattr(tag, "values", tag)
    if isinstance(vals, (list, tuple)):
        return vals[0] if vals else None
    return vals


def _str(tag: Any) -> str | None:
    if tag is None:
        return None
    s = str(getattr(tag, "printable", tag)).strip().strip("\x00").strip()
    return s or None


def _dms(tag: Any, ref: Any) -> float | None:
    vals = getattr(tag, "values", tag)
    if not vals or len(vals) < 3:
        return None
    d, m, s = (_ratio(x) for x in vals[:3])
    if d is None or m is None or s is None:
        return None
    deg = d + m / 60 + s / 3600
    if _str(ref) in ("S", "W"):
        deg = -deg
    return deg


_DT = re.compile(r"(\d{4})[:\-](\d{2})[:\-](\d{2})[ T](\d{2}):(\d{2}):(\d{2})")
_OFF = re.compile(r"([+-])(\d{2}):?(\d{2})")


def parse_exif_datetime(
    s: str | None, offset: str | None = None
) -> tuple[datetime | None, int | None]:
    if not s:
        return None, None
    m = _DT.search(s)
    if not m:
        return None, None
    try:
        y, mo, d, h, mi, se = (int(x) for x in m.groups())
        if y < 1900:
            return None, None
        dt = datetime(y, mo, d, h, mi, se, tzinfo=UTC)
    except ValueError:
        return None, None
    off_min = None
    if offset and (om := _OFF.search(offset)):
        sign = -1 if om.group(1) == "-" else 1
        off_min = sign * (int(om.group(2)) * 60 + int(om.group(3)))
    return dt, off_min


def camera_name(make: str | None, model: str | None) -> str | None:
    if not model:
        return make
    if not make:
        return model
    brand = make.split()[0]
    if model.lower().startswith(brand.lower()):
        return model
    # "NIKON CORPORATION" + "NIKON Z 6" / "Apple" + "iPhone 15 Pro"
    return f"{brand.title() if brand.isupper() else brand} {model}"


def format_shutter(seconds: float | None) -> str | None:
    if not seconds:
        return None
    if seconds >= 1:
        return f"{seconds:g}s"
    frac = Fraction(seconds).limit_denominator(16000)
    return (
        f"1/{round(1 / seconds)}" if frac.numerator != 1 else f"{frac.numerator}/{frac.denominator}"
    )


def read_exif(path: Path) -> Exif:
    """EXIF via exifread (JPEG, HEIC, TIFF-based RAW), falling back to Pillow."""
    import exifread

    tags: dict[str, Any] = {}
    try:
        with path.open("rb") as f:
            tags = exifread.process_file(f, details=False, extract_thumbnail=False)
    except Exception:
        tags = {}
    if not tags and media_kind(path) != "video":
        tags = _pillow_tags(path)

    def g(*names: str) -> Any:
        for n in names:
            if n in tags:
                return tags[n]
        return None

    ex = Exif()
    ex.taken_at, ex.tz_offset_min = parse_exif_datetime(
        _str(g("EXIF DateTimeOriginal", "EXIF DateTimeDigitized", "Image DateTime")),
        _str(g("EXIF OffsetTimeOriginal", "EXIF OffsetTime")),
    )
    lat = g("GPS GPSLatitude")
    lon = g("GPS GPSLongitude")
    if lat is not None and lon is not None:
        ex.lat = _dms(lat, g("GPS GPSLatitudeRef"))
        ex.lon = _dms(lon, g("GPS GPSLongitudeRef"))
        if ex.lat is not None and ex.lon is not None and ex.lat == 0 and ex.lon == 0:
            ex.lat = ex.lon = None  # null island: GPS fix never happened
    ex.camera = camera_name(_str(g("Image Make")), _str(g("Image Model")))
    ex.lens = _str(g("EXIF LensModel", "MakerNote LensType", "EXIF LensSpecification"))
    if ex.lens and ex.lens.startswith("["):
        ex.lens = None
    ex.focal_length = _ratio(_first(g("EXIF FocalLength")))
    ex.focal_length_35mm = _ratio(_first(g("EXIF FocalLengthIn35mmFilm")))
    ex.aperture = _ratio(_first(g("EXIF FNumber")))
    iso = _first(g("EXIF ISOSpeedRatings", "EXIF PhotographicSensitivity"))
    ex.iso = int(iso) if isinstance(iso, (int, float)) and iso > 0 else None
    ex.exposure_s = _ratio(_first(g("EXIF ExposureTime")))
    ex.shutter = format_shutter(ex.exposure_s)
    w = _first(g("EXIF ExifImageWidth", "Image ImageWidth"))
    h = _first(g("EXIF ExifImageLength", "Image ImageLength"))
    ex.width = int(w) if isinstance(w, int) else None
    ex.height = int(h) if isinstance(h, int) else None
    o = _first(g("Image Orientation"))
    ex.orientation = int(o) if isinstance(o, int) else None
    ex.keywords = read_keywords(path)
    return ex


class _PT:
    """Adapter so Pillow EXIF values look like exifread tags."""

    def __init__(self, v: Any):
        self.values = list(v) if isinstance(v, tuple) and not hasattr(v, "numerator") else [v]
        if isinstance(v, (str, bytes)):
            self.values = [v]
        self.printable = v.decode(errors="ignore") if isinstance(v, bytes) else str(v)

    def __str__(self) -> str:
        return self.printable


def _pillow_tags(path: Path) -> dict[str, Any]:
    from PIL import ExifTags

    try:
        with Image.open(path) as im:
            exif = im.getexif()
    except Exception:
        return {}
    out: dict[str, Any] = {}
    for k, v in exif.items():
        out[f"Image {ExifTags.TAGS.get(k, k)}"] = _PT(v)
    for k, v in exif.get_ifd(ExifTags.IFD.Exif).items():
        out[f"EXIF {ExifTags.TAGS.get(k, k)}"] = _PT(v)
    for k, v in exif.get_ifd(ExifTags.IFD.GPSInfo).items():
        name = ExifTags.GPSTAGS.get(k, k)
        out[f"GPS {name}"] = _PT(v)
    return out


# ---------------------------------------------------------------- XMP keywords

_XMP_BLOCK = re.compile(rb"<x:xmpmeta.*?</x:xmpmeta>", re.S)
_SUBJECT = re.compile(r"<dc:subject>(.*?)</dc:subject>", re.S)
_HIER = re.compile(r"<lr:hierarchicalSubject>(.*?)</lr:hierarchicalSubject>", re.S)
_LI = re.compile(r"<rdf:li[^>]*>(.*?)</rdf:li>", re.S)


def parse_xmp_keywords(xmp: str) -> list[str]:
    kws: list[str] = []
    for block in _SUBJECT.findall(xmp) + _HIER.findall(xmp):
        for li in _LI.findall(block):
            for part in li.split("|"):  # hierarchical: "Places|USA|Chicago"
                part = part.strip()
                if part and part not in kws:
                    kws.append(part)
    return kws


def read_keywords(path: Path) -> list[str]:
    """Lightroom keywords from an XMP sidecar (RAW) or the file's embedded XMP packet."""
    for sidecar in (path.with_suffix(".xmp"), path.with_suffix(path.suffix + ".xmp")):
        if sidecar.exists():
            try:
                return parse_xmp_keywords(sidecar.read_text(errors="ignore"))
            except OSError:
                pass
    if media_kind(path) == "video":
        return []
    try:
        with path.open("rb") as f:
            head = f.read(512 * 1024)
    except OSError:
        return []
    m = _XMP_BLOCK.search(head)
    return parse_xmp_keywords(m.group(0).decode("utf-8", "ignore")) if m else []


def local_now() -> datetime:
    return datetime.now(UTC)


def to_wallclock(dt: datetime, offset_min: int | None) -> datetime:
    """Shift a true-UTC timestamp (e.g. from a video container) to wall-clock-as-UTC."""
    return dt + timedelta(minutes=offset_min or 0)
