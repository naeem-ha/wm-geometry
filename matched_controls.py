"""Per-clip motion-matched real controls and a frozen-frame baseline.

For every generated clip we search real footage of the *same episode* (same scene and camera) for a
5 s segment whose image motion matches the generated clip's:
  - the true continuation slowed by k in SLOW_K (k <= 3, so the 5 fps track has no repeated frames)
  - every other 5 s window of the episode at natural speed (natural low-motion moments)
The candidate closest in log image motion is kept if its relative error is <= MAX_REL_ERR.
Two independent image-motion estimators are used for matching: the SIFT-match displacement proxy
used throughout the paper, and dense Farneback optical flow, which does not depend on which features
survive matching. Every clip (generated, matched control, frozen baseline) is scored here with the
same code: COLMAP and epipolar metrics at 0.2/0.4/0.8 s, VGGT depth disagreement (default and
visibility-filtered) at 0.2/0.4/0.8 s. Writes work/matched.json.
"""
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from vggt_sensitivity import pair_scores
from wmgeo import colmap_eval, epipolar, io, vggt_eval
from wmgeo.data_droid import decode, episodes

ROOT = Path(__file__).resolve().parent
W = ROOT / "work"
MASK_PNG = ROOT / "data/droid_gripper_mask.png"
MASK = cv2.imread(str(MASK_PNG), 0)
MODELS = ["gen_ltx", "gen_cogvideox_5b", "gen_wan22_5b", "gen_ctrlworld"]
SLOW_K = (1.0, 1.25, 1.5, 2.0, 2.5, 3.0)
N15, STEP = 76, 3            # 5 s at 15 fps, scored every third frame (26 frames at 5 fps)
GAPS = (1, 2, 4)             # 0.2 / 0.4 / 0.8 s at 5 fps
MAX_REL_ERR = 0.25
WINDOW_STRIDE = 5


def to5(frames15):
    return frames15[:N15][::STEP]


def slowed(frames15, k):
    idx = np.minimum(np.round(np.arange(N15) / k).astype(int), len(frames15) - 1)
    return [frames15[i] for i in idx]


def motion_sift(f5, gap=2):
    m = [epipolar.pair_stats(f5[i], f5[i + gap], MASK) for i in range(0, len(f5) - gap, 2)]
    m = [r["flow_px"] for r in m if r is not None]
    return float(np.mean(m)) if m else 0.0


def motion_dense(f5, gap=2):
    keep = MASK > 0
    vals = []
    for i in range(0, len(f5) - gap, 2):
        a = cv2.cvtColor(f5[i], cv2.COLOR_RGB2GRAY)
        b = cv2.cvtColor(f5[i + gap], cv2.COLOR_RGB2GRAY)
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.1, 0)
        vals.append(np.median(np.linalg.norm(flow, axis=2)[keep]))
    return float(np.mean(vals))


def score(f5):
    """All metrics for a 26-frame 5 fps clip."""
    tmp = W / "_match_tmp"
    io.write_clip(f5, tmp)
    try:
        r = colmap_eval.run(tmp, MASK_PNG)
        r.pop("centers")
        r["epipolar"] = epipolar.clip_stats(f5, MASK, gaps=GAPS)
        p = vggt_eval.predict(sorted((tmp / "frames").glob("*.png")))
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    r["vggt"] = {}
    for g in GAPS:
        q = [pair_scores(p, i, i + g) for i in range(len(f5) - g)]
        r["vggt"][g] = {k: float(np.nanmedian([x[k] for x in q])) for k in ("err_default", "err_filtered")}
        r["vggt"][g].update({k: float(np.nanmean([x[k] for x in q])) for k in ("cov_default", "cov_filtered")})
    r["motion_sift"] = motion_sift(f5)
    r["motion_dense"] = motion_dense(f5)
    return r


def main():
    out = W / "matched.json"
    res = json.loads(out.read_text()) if out.exists() else {"candidates": {}, "model": {}, "matched": {}, "frozen": {}}
    eps = {f"droid_ep{e['episode']:03d}": e for e in episodes(ROOT / "data/droid_100")}
    clips = sorted(d.name for d in (W / "real").iterdir() if (d / "frames").exists())
    for c in clips:
        meta = io.read_meta(W / "real" / c)
        if c not in res["candidates"]:
            ep = eps[c]
            ep15 = decode(ep["video"], ep["t0"], ep["t1"])
            real15 = io.read_clip(W / "real" / c)
            cands = {f"slow{k:g}": io.reencode(to5(slowed(real15, k))) for k in SLOW_K}
            for s in range(0, len(ep15) - N15, WINDOW_STRIDE):
                if s != meta["start_frame"]:
                    cands[f"win{s}"] = io.reencode(to5(ep15[s:s + N15]))
            res["candidates"][c] = {name: {"motion_sift": motion_sift(f), "motion_dense": motion_dense(f)}
                                    for name, f in cands.items()}
            out.write_text(json.dumps(res))
            print(c, "candidates", len(cands), flush=True)
        # frozen-frame baseline: the shared first frame repeated
        if c not in res["frozen"]:
            res["frozen"][c] = score(io.reencode([io.read_clip(W / "real" / c)[0]] * len(to5(range(N15)))))
            out.write_text(json.dumps(res))
        for m in MODELS:
            if not (W / m / c / "meta.json").exists():
                continue
            key = f"{m}/{c}"
            if key not in res["model"]:
                res["model"][key] = score(to5(io.read_clip(W / m / c)))
                out.write_text(json.dumps(res))
            for est in ("motion_sift", "motion_dense"):
                mkey = f"{key}/{est}"
                if mkey in res["matched"]:
                    continue
                target = res["model"][key][est]
                cand = res["candidates"][c]
                name = min(cand, key=lambda n: abs(np.log(max(cand[n][est], 1e-3) / max(target, 1e-3))))
                rel = cand[name][est] / max(target, 1e-3) - 1
                entry = {"candidate": name, "target": target, "cand_motion": cand[name][est], "rel_err": rel,
                         "accepted": abs(rel) <= MAX_REL_ERR}
                if entry["accepted"]:
                    real15 = io.read_clip(W / "real" / c)
                    if name.startswith("slow"):
                        f5 = io.reencode(to5(slowed(real15, float(name[4:]))))
                    else:
                        ep = eps[c]
                        s = int(name[3:])
                        f5 = io.reencode(to5(decode(ep["video"], ep["t0"], ep["t1"])[s:s + N15]))
                    entry["scores"] = score(f5)
                res["matched"][mkey] = entry
                out.write_text(json.dumps(res))
        print(c, "done", flush=True)


if __name__ == "__main__":
    main()
