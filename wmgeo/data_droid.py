"""Sample real conditioning clips from the DROID-100 wrist camera (LeRobot v3 layout).

DROID's exterior cameras are fixed on tripods, so they give SfM no baseline. The wrist camera
moves with the arm, so we use it. Its gripper fingers are fixed in the image and would anchor
SfM to a fake static structure; `build_gripper_mask` finds them once across all clips.
"""
import subprocess
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from .io import EVAL_FPS, write_clip

CAM = "observation.images.wrist_image_left"


def episodes(root):
    e = pd.read_parquet(Path(root) / "meta/episodes/chunk-000/file-000.parquet")
    return [{
        "episode": int(r["episode_index"]),
        "task": str(r["tasks"][0]),
        "length": int(r["length"]),
        "video": str(Path(root) / f"videos/{CAM}/chunk-{int(r[f'videos/{CAM}/chunk_index']):03d}"
                                  f"/file-{int(r[f'videos/{CAM}/file_index']):03d}.mp4"),
        "t0": float(r[f"videos/{CAM}/from_timestamp"]),
        "t1": float(r[f"videos/{CAM}/to_timestamp"]),
    } for _, r in e.iterrows()]


def decode(video, t0, t1, w=320, h=180):
    cmd = ["ffmpeg", "-v", "error", "-ss", f"{t0:.4f}", "-to", f"{t1:.4f}", "-i", video,
           "-vf", f"fps={EVAL_FPS},scale={w}:{h}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    return list(np.frombuffer(raw, np.uint8).reshape(-1, h, w, 3))


GRIPPER_MASK = Path(__file__).resolve().parents[1] / "data/droid_gripper_mask.png"


def build_gripper_mask(clip_dirs, thr=3.0):
    """255 = keep. The gripper is the region that barely changes across many moving-camera clips."""
    var = []
    for d in clip_dirs:
        g = np.stack([cv2.imread(str(f), 0).astype(np.float32) for f in sorted(Path(d, "frames").glob("*.png"))[::3]])
        var.append(np.median(np.abs(np.diff(g, axis=0)), axis=0))
    keep = (np.median(np.stack(var), axis=0) > thr).astype(np.uint8) * 255
    keep = cv2.morphologyEx(keep, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    keep = cv2.morphologyEx(keep, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
    keep = cv2.erode(keep, np.ones((15, 15), np.uint8))
    cv2.imwrite(str(GRIPPER_MASK), keep)
    return keep


def _motion(frames, step=8):
    """Median sparse-flow magnitude between frames `step` apart: a cheap camera-motion score."""
    mags = []
    for i in range(0, len(frames) - step, step):
        a = cv2.cvtColor(frames[i], cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(frames[i + step], cv2.COLOR_RGB2GRAY)
        p = cv2.goodFeaturesToTrack(a, 300, 0.01, 6)
        if p is None:
            continue
        q, st, _ = cv2.calcOpticalFlowPyrLK(a, b, p, None)
        ok = st.ravel() == 1
        if ok.sum() > 20:
            mags.append(np.median(np.linalg.norm((q - p)[ok].reshape(-1, 2), axis=1)))
    return float(np.median(mags)) if mags else 0.0


def sample_clips(root, out_dir, horizon_s=5.0, n_clips=50, min_motion=3.0):
    """One clip per episode: the window with the most camera motion, if it clears `min_motion` px."""
    n = int(horizon_s * EVAL_FPS) + 1  # frame 0 is the conditioning frame
    picked = []
    for ep in episodes(root):
        frames = decode(ep["video"], ep["t0"], ep["t1"])
        if len(frames) < n + 1:
            continue
        starts = range(0, len(frames) - n, max(1, n // 3))
        scored = [(s, _motion(frames[s:s + n])) for s in starts]
        s, m = max(scored, key=lambda x: x[1])
        if m >= min_motion:
            picked.append((m, ep, frames[s:s + n], s))
    picked = sorted(picked, key=lambda x: -x[0])[:n_clips]
    for m, ep, clip, s in picked:
        d = Path(out_dir) / f"droid_ep{ep['episode']:03d}"
        write_clip(clip, d, {"source": "droid_100", "episode": ep["episode"], "task": ep["task"],
                             "start_frame": s, "fps": EVAL_FPS, "motion_px": m})
    return len(picked)
