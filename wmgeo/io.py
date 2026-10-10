"""Frame I/O shared by every stage. A clip is a folder of numbered PNGs plus meta.json."""
import json
from pathlib import Path

import cv2
import numpy as np

# Every clip (real, generated, control) is evaluated at this size and rate so the
# comparison does not depend on generator output resolution or fps.
EVAL_W, EVAL_H = 320, 180
EVAL_FPS = 15


def write_clip(frames, out_dir, meta=None):
    out_dir = Path(out_dir)
    (out_dir / "frames").mkdir(parents=True, exist_ok=True)
    for i, f in enumerate(frames):
        cv2.imwrite(str(out_dir / "frames" / f"{i:05d}.png"), cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
    if meta is not None:
        (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))


def read_clip(clip_dir):
    paths = sorted((Path(clip_dir) / "frames").glob("*.png"))
    return [cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB) for p in paths]


def read_meta(clip_dir):
    p = Path(clip_dir) / "meta.json"
    return json.loads(p.read_text()) if p.exists() else {}


def to_eval_res(frames):
    return [cv2.resize(f, (EVAL_W, EVAL_H), interpolation=cv2.INTER_AREA) for f in frames]


def resample_fps(frames, src_fps, dst_fps=EVAL_FPS):
    """Nearest-frame temporal resampling, keeping the clip duration."""
    n_out = max(1, int(round(len(frames) * dst_fps / src_fps)))
    idx = np.clip(np.round(np.arange(n_out) * src_fps / dst_fps).astype(int), 0, len(frames) - 1)
    return [frames[i] for i in idx]


def reencode(frames, quality=8):
    """Round-trip through H.264 so real clips carry the same compression as generated ones."""
    import imageio.v3 as iio
    buf = iio.imwrite("<bytes>", np.stack(frames), extension=".mp4", fps=EVAL_FPS,
                      codec="libx264", quality=quality, macro_block_size=1)
    out = list(iio.imread(buf, extension=".mp4"))
    assert out[0].shape == frames[0].shape, (out[0].shape, frames[0].shape)
    return out
