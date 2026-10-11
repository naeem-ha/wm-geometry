"""VGGT scores on one shared clip set, with a visibility-filtered variant and coverage.

For every condition and every clip common to all conditions (5 s horizon, frames every 0.2 s):
- depth disagreement, default: source depth above its median confidence, reprojected into the
  target frame 0.4 s later (as in wmgeo/vggt_eval.py)
- depth disagreement, filtered: additionally the target pixel must be above the target's median
  confidence, and points landing more than 10% *behind* the target surface are treated as occluded
  and dropped (rejects points hidden in the target view rather than counting them as errors)
- coverage: fraction of source points that land in frame (default) / survive filtering (filtered)
- camera-path error vs the VGGT-estimated path of the real clip (Sim(3)-aligned RMSE / path length)
Writes work/vggt_sens.json.
"""
import json
import shutil
from pathlib import Path

import numpy as np

from wmgeo import io, vggt_eval

ROOT = Path(__file__).resolve().parent
W = ROOT / "work"
CONDS = ["real", "ctrl_real", "ctrl_blur0.7", "ctrl_slow2.5", "ctrl_warp1", "ctrl_warp2", "ctrl_warp4",
         "ctrl_warp8", "ctrl_mix", "gen_ltx", "gen_cogvideox_5b", "gen_wan22_5b", "gen_ctrlworld"]
GAP = 2           # in 5 fps frames = 0.4 s
OCCLUSION = 0.10  # relative depth margin behind the target surface treated as occluded


def common_clips():
    sets = [{d.name for d in (W / c).iterdir() if (d / "meta.json").exists() or (d / "frames").exists()} for c in CONDS]
    return sorted(set.intersection(*sets))


def pair_scores(p, i, j):
    d_i, d_j = p["depth"][i], p["depth"][j]
    c_i, c_j = p["depth_conf"][i], p["depth_conf"][j]
    H, W_ = d_i.shape
    v, u = np.nonzero(c_i > np.median(c_i))
    z = d_i[v, u]
    Ki, Kj = p["intrinsic"][i], p["intrinsic"][j]
    Ri, ti = p["extrinsic"][i][:, :3], p["extrinsic"][i][:, 3:]
    Rj, tj = p["extrinsic"][j][:, :3], p["extrinsic"][j][:, 3:]
    pw = Ri.T @ (np.linalg.inv(Ki) @ np.stack([u * z, v * z, z]) - ti)
    pj = Rj @ pw + tj
    zj = pj[2]
    uv = (Kj @ pj)[:2] / np.maximum(zj, 1e-6)
    uj, vj = np.round(uv).astype(int)
    inframe = (zj > 1e-6) & (uj >= 0) & (uj < W_) & (vj >= 0) & (vj < H)
    n = len(z)
    out = {"cov_default": inframe.mean() if n else np.nan}
    if inframe.sum() < 100:
        return {**out, "err_default": np.nan, "err_filtered": np.nan, "cov_filtered": 0.0}
    dt = d_j[vj[inframe], uj[inframe]]
    rel = (zj[inframe] - dt) / np.maximum(dt, 1e-6)
    out["err_default"] = float(np.median(np.abs(rel)))
    keep = (c_j[vj[inframe], uj[inframe]] > np.median(c_j)) & (rel < OCCLUSION)
    out["cov_filtered"] = float(keep.sum() / n)
    out["err_filtered"] = float(np.median(np.abs(rel[keep]))) if keep.sum() >= 100 else np.nan
    return out


def score_clip(frames):
    tmp = W / "_sens_tmp"
    io.write_clip(frames, tmp)
    try:
        paths = sorted((tmp / "frames").glob("*.png"))
        p = vggt_eval.predict(paths)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    pairs = [pair_scores(p, i, i + GAP) for i in range(len(paths) - GAP)]
    agg = {k: float(np.nanmedian([q[k] for q in pairs])) for k in ("err_default", "err_filtered")}
    agg.update({k: float(np.nanmean([q[k] for q in pairs])) for k in ("cov_default", "cov_filtered")})
    return agg, vggt_eval.centers(p["extrinsic"])


def main():
    clips = common_clips()
    print(f"{len(clips)} common clips", flush=True)
    out_path = W / "vggt_sens.json"
    res = json.loads(out_path.read_text()) if out_path.exists() else {}
    real_centers = {}
    for cond in CONDS:
        res.setdefault(cond, {})
        for c in clips:
            if c in res[cond] and cond != "real":
                continue
            frames = io.read_clip(W / cond / c)[:76][::3]  # 5 s at 5 fps
            agg, centers = score_clip(frames)
            if cond == "real":
                real_centers[c] = centers
            else:
                if c not in real_centers:
                    _, real_centers[c] = score_clip(io.read_clip(W / "real" / c)[:76][::3])
                agg["path_err"] = vggt_eval.ate_sim3(centers, real_centers[c])
            res[cond][c] = agg
        out_path.write_text(json.dumps(res, indent=1))
        print(cond, "done", flush=True)
    # reference paths for path-error baselines on the same clips
    res["_real_centers"] = {c: real_centers[c].tolist() for c in clips}
    out_path.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
