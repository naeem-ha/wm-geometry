"""Control clips that bracket the metrics.

- real:  the true continuation, round-tripped through H.264 (upper bound)
- warp:  the real continuation with smooth non-rigid per-frame warps of growing amplitude;
         a dose-response curve that shows each metric responds to geometric distortion
- mix:   frames drawn from unrelated clips (geometry-free lower bound)
- slow:  the real continuation slowed by a factor (nearest-frame repeats), matching the reduced
         camera motion of generated clips, so low motion alone can be told apart from bad geometry
- blur:  the real continuation Gaussian-blurred to the sharpness of soft generators (Ctrl-World,
         LTX), so lost detail (fewer features) can be told apart from inconsistent geometry
"""
import cv2
import numpy as np

from .io import reencode


def real(frames):
    return reencode(frames)


def _smooth_field(h, w, amp_px, rng, cells=4):
    coarse = rng.normal(0, 1, (cells, cells, 2)).astype(np.float32)
    field = cv2.resize(coarse, (w, h), interpolation=cv2.INTER_CUBIC)
    return field / (np.abs(field).max() + 1e-6) * amp_px


def warp(frames, amp_px, seed=0):
    """Each frame gets an independent smooth displacement field, so no rigid scene explains them all."""
    rng = np.random.default_rng(seed)
    h, w = frames[0].shape[:2]
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    out = []
    for f in frames:
        d = _smooth_field(h, w, amp_px, rng)
        out.append(cv2.remap(f, gx + d[..., 0], gy + d[..., 1], cv2.INTER_LINEAR,
                             borderMode=cv2.BORDER_REFLECT))
    return reencode(out)


def mix(pool, n, seed=0):
    """pool: list of clips (lists of frames). Returns n frames sampled across different clips."""
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n):
        c = pool[rng.integers(len(pool))]
        out.append(c[rng.integers(len(c))])
    return reencode(out)


def slow(frames, factor):
    """First len/factor seconds of the real clip stretched over the full length."""
    n = len(frames)
    idx = np.minimum((np.arange(n) / factor).round().astype(int), n - 1)
    return reencode([frames[i] for i in idx])


def blur(frames, sigma):
    return reencode([cv2.GaussianBlur(f, (0, 0), sigma) for f in frames])
