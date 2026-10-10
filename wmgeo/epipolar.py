"""Two-view epipolar consistency. Needs no full reconstruction, so it still scores clips where SfM fails."""
import cv2
import numpy as np

GAPS = (1, 4, 8)  # frame gaps at EVAL_FPS
_sift = cv2.SIFT_create(nfeatures=2000)


def _features(frame, mask):
    gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    return _sift.detectAndCompute(gray, mask)


def _sampson(F, x1, x2):
    x1h = np.c_[x1, np.ones(len(x1))]
    x2h = np.c_[x2, np.ones(len(x2))]
    Fx1 = x1h @ F.T
    Ftx2 = x2h @ F
    num = np.sum(x2h * Fx1, axis=1) ** 2
    den = Fx1[:, 0] ** 2 + Fx1[:, 1] ** 2 + Ftx2[:, 0] ** 2 + Ftx2[:, 1] ** 2
    return num / np.maximum(den, 1e-12)


def pair_stats(f1, f2, mask=None):
    k1, d1 = _features(f1, mask)
    k2, d2 = _features(f2, mask)
    if d1 is None or d2 is None or len(k1) < 16 or len(k2) < 16:
        return None
    knn = cv2.BFMatcher().knnMatch(d1, d2, k=2)
    good = [m for m, n in (p for p in knn if len(p) == 2) if m.distance < 0.8 * n.distance]
    if len(good) < 16:
        return None
    x1 = np.float32([k1[m.queryIdx].pt for m in good])
    x2 = np.float32([k2[m.trainIdx].pt for m in good])
    flow = float(np.median(np.linalg.norm(x2 - x1, axis=1)))
    try:
        F, inl = cv2.findFundamentalMat(x1, x2, cv2.USAC_MAGSAC, 1.0, 0.999, 10000)
    except cv2.error:  # OpenCV 5 asserts instead of returning None on degenerate input
        F = None
    if F is None or F.shape != (3, 3):
        return {"n_matches": len(good), "flow_px": flow, "inlier_ratio": 0.0, "n_inliers": 0,
                "sampson_med": np.nan}
    err = np.sqrt(_sampson(F, x1, x2))
    return {
        "n_matches": len(good),
        "flow_px": flow,  # camera/scene motion proxy; near-static pairs are uninformative
        "inlier_ratio": float(inl.mean()),
        # With few matches some F always fits, so the ratio alone flatters unrelated frames;
        # the absolute inlier count does not.
        "n_inliers": int(inl.sum()),
        "sampson_med": float(np.median(err)),
    }


def clip_stats(frames, mask=None, gaps=GAPS, stride=2):
    """Per-gap averages over frame pairs (i, i+gap) for i on a stride."""
    out = {}
    for g in gaps:
        rows = [pair_stats(frames[i], frames[i + g], mask) for i in range(0, len(frames) - g, stride)]
        rows = [r for r in rows if r is not None]
        out[g] = {
            "n_pairs": len(rows),
            "match_fail_rate": 1 - len(rows) / max(1, len(range(0, len(frames) - g, stride))),
            **{k: float(np.nanmean([r[k] for r in rows])) if rows else np.nan
               for k in ("n_matches", "n_inliers", "flow_px", "inlier_ratio", "sampson_med")},
        }
    return out
