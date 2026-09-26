"""Video support: probing with ffmpeg and scene-based frame sampling.

Each sampled frame becomes a `photos` row (is_video_frame = TRUE,
source_video_id, frame_ts) so it is embedded, captioned and searched exactly
like a still. Search collapses frames back to their video and reports the
best-matching timestamp.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import tempfile
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from PIL import Image

SCENE_THRESHOLD = 0.3
MAX_GAP_S = 20.0  # take a frame at least this often even without a scene cut
MAX_FRAMES = 60


def ffmpeg_exe() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    import imageio_ffmpeg

    return imageio_ffmpeg.get_ffmpeg_exe()


@dataclass
class VideoInfo:
    duration_s: float | None = None
    width: int | None = None
    height: int | None = None
    created_at: datetime | None = None  # true UTC from the container
    lat: float | None = None
    lon: float | None = None
    rotation: int = 0


_DUR = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_RES = re.compile(r"Stream #\S+.*?Video:.*?(\d{2,5})x(\d{2,5})")
_CREATED = re.compile(r"creation_time\s*:\s*(\S+)")
_ISO6709 = re.compile(r"(?:location|ISO6709)\s*:\s*([+-]\d+(?:\.\d+)?)([+-]\d+(?:\.\d+)?)")
_ROT = re.compile(r"(?:rotate\s*:\s*|rotation of )(-?\d+(?:\.\d+)?)")


def parse_probe(stderr: str) -> VideoInfo:
    info = VideoInfo()
    if m := _DUR.search(stderr):
        h, mi, s = m.groups()
        info.duration_s = int(h) * 3600 + int(mi) * 60 + float(s)
    if m := _RES.search(stderr):
        info.width, info.height = int(m.group(1)), int(m.group(2))
    if m := _CREATED.search(stderr):
        try:
            info.created_at = datetime.fromisoformat(m.group(1).replace("Z", "+00:00"))
            if info.created_at.tzinfo is None:
                info.created_at = info.created_at.replace(tzinfo=UTC)
        except ValueError:
            pass
    if m := _ISO6709.search(stderr):
        info.lat, info.lon = float(m.group(1)), float(m.group(2))
    if m := _ROT.search(stderr):
        info.rotation = round(float(m.group(1))) % 360
        if info.rotation in (90, 270) and info.width and info.height:
            info.width, info.height = info.height, info.width
    return info


def probe(path: Path) -> VideoInfo:
    proc = subprocess.run(
        [ffmpeg_exe(), "-hide_banner", "-i", str(path)],
        capture_output=True,
        text=True,
        errors="ignore",
        timeout=60,
    )
    return parse_probe(proc.stderr)


def frame_at(path: Path, ts: float = 0.0, max_side: int = 1024) -> Image.Image | None:
    with tempfile.TemporaryDirectory() as d:
        out = Path(d) / "f.jpg"
        subprocess.run(
            [
                ffmpeg_exe(), "-hide_banner", "-loglevel", "error", "-ss", f"{ts:.3f}",
                "-i", str(path), "-frames:v", "1",
                "-vf", f"scale={max_side}:{max_side}:force_original_aspect_ratio=decrease",
                "-q:v", "3", str(out),
            ],
            capture_output=True,
            timeout=120,
        )  # fmt: skip
        if not out.exists():
            return None
        with Image.open(out) as im:
            return im.convert("RGB").copy()


_PTS = re.compile(r"pts_time:\s*(\d+(?:\.\d+)?)")


def scene_frames(
    path: Path,
    threshold: float = SCENE_THRESHOLD,
    max_gap_s: float = MAX_GAP_S,
    max_frames: int = MAX_FRAMES,
    max_side: int = 768,
) -> list[tuple[float, Image.Image]]:
    """Frames at scene cuts (ffmpeg scene score > threshold), plus the first frame and
    one at least every `max_gap_s` seconds so long static shots are still covered."""
    select = f"select='eq(n,0)+gt(scene,{threshold})+gte(t-prev_selected_t,{max_gap_s})'"
    vf = f"{select},showinfo,scale={max_side}:{max_side}:force_original_aspect_ratio=decrease"
    with tempfile.TemporaryDirectory() as d:
        proc = subprocess.run(
            [
                ffmpeg_exe(), "-hide_banner", "-i", str(path), "-vf", vf,
                "-vsync", "vfr", "-frames:v", str(max_frames), "-q:v", "3",
                str(Path(d) / "f_%05d.jpg"),
            ],
            capture_output=True,
            text=True,
            errors="ignore",
            timeout=1800,
        )  # fmt: skip
        stamps = [float(x) for x in _PTS.findall(proc.stderr)]
        files = sorted(Path(d).glob("f_*.jpg"))
        out = []
        for ts, f in zip(stamps, files, strict=False):
            with Image.open(f) as im:
                out.append((ts, im.convert("RGB").copy()))
        return out


def index_videos(limit: int | None = None) -> int:
    """Extract scene frames for every video that hasn't been processed yet."""
    from api.config import get_settings
    from api.db.session import get_conn
    from api.paths import stored
    from workers.media import perceptual_hash, sharpness

    s = get_settings()
    n_frames = 0
    with get_conn() as conn:
        videos = conn.execute(
            "SELECT * FROM photos WHERE media_type = 'video' AND frames_indexed_at IS NULL "
            "ORDER BY path LIMIT %s",
            (limit,),
        ).fetchall()
    for v in videos:
        frames = scene_frames(Path(v["path"]))
        rows = []
        for ts, im in frames:
            fid = uuid.uuid4()
            thumb = im.copy()
            thumb.thumbnail((s.thumb_size, s.thumb_size))
            tp = s.frames_dir / str(v["id"])[:2] / f"{v['id']}_{ts:09.3f}.jpg"
            tp.parent.mkdir(parents=True, exist_ok=True)
            thumb.save(tp, quality=85)
            taken = v["taken_at"] + timedelta(seconds=ts) if v["taken_at"] else None
            rows.append(
                {
                    "id": fid,
                    "path": f"{v['path']}#t={ts:.2f}",
                    "file_hash": f"{v['file_hash']}:{ts:.2f}",
                    "taken_at": taken,
                    "frame_ts": ts,
                    "source_video_id": v["id"],
                    "thumb_path": stored(tp),
                    "phash": perceptual_hash(thumb),
                    "sharpness": sharpness(thumb),
                    "width": im.width,
                    "height": im.height,
                }
            )
        with get_conn() as conn:
            for r in rows:
                conn.execute(
                    """
                    INSERT INTO photos (id, path, file_hash, media_type, format, taken_at,
                        lat, lon, place_name, admin1, country, camera, is_video_frame,
                        source_video_id, frame_ts, thumb_path, phash, sharpness, width, height)
                    SELECT %(id)s, %(path)s, %(file_hash)s, 'image', 'video_frame', %(taken_at)s,
                        v.lat, v.lon, v.place_name, v.admin1, v.country, v.camera, TRUE,
                        v.id, %(frame_ts)s, %(thumb_path)s, %(phash)s, %(sharpness)s,
                        %(width)s, %(height)s
                    FROM photos v WHERE v.id = %(source_video_id)s
                    ON CONFLICT (path) DO NOTHING
                    """,
                    r,
                )
            conn.execute("UPDATE photos SET frames_indexed_at = now() WHERE id = %s", (v["id"],))
            conn.commit()
        n_frames += len(rows)
    return n_frames
