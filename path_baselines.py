"""Reference levels for the camera-path error (VGGT, Sim(3)-aligned, divided by real path length).

- controls (re-encoded, blurred, warped, slowed real): VGGT on the control vs the real clip's path
- frozen camera: every predicted center identical (best Sim(3) fit collapses to the mean)
- other clip path: the real path of a different clip (an unrelated but plausible trajectory)
Generated conditions are read from scores.json (5 s horizon). Writes work/ate_baselines.json.
"""
import json
import shutil
from pathlib import Path

import numpy as np

from wmgeo import io, vggt_eval

W = Path(__file__).resolve().parent / "work"
CONTROLS = ["ctrl_real", "ctrl_blur0.7", "ctrl_warp8", "ctrl_slow2.5"]
GENERATED = ["gen_ltx", "gen_cogvideox_5b", "gen_wan22_5b", "gen_ctrlworld"]


def main():
    clips = sorted(p.name for p in (W / "real").iterdir() if (p / "vggt_centers_h5.npy").exists())
    ref = {c: np.load(W / "real" / c / "vggt_centers_h5.npy") for c in clips}
    res = {}
    for cond in CONTROLS:
        res[cond] = []
        for c in clips:
            tmp = W / f"_ate_{c}"
            io.write_clip(io.read_clip(W / cond / c)[:76], tmp)
            try:
                _, p = vggt_eval.clip_stats(tmp)
                res[cond].append(vggt_eval.ate_sim3(vggt_eval.centers(p["extrinsic"]), ref[c]))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
    res["frozen camera"] = [np.sqrt(((r - r.mean(0)) ** 2).sum(1).mean()) / np.linalg.norm(np.diff(r, axis=0), axis=1).sum()
                            for r in ref.values()]
    rng = np.random.default_rng(0)
    res["other clip path"] = [vggt_eval.ate_sim3(ref[clips[(i + 1 + rng.integers(len(clips) - 1)) % len(clips)]], ref[k])
                              for i, k in enumerate(clips)]
    for cond in GENERATED:
        res[cond] = [json.loads(f.read_text())["5"]["vggt"]["ate_vs_real"] for f in sorted((W / cond).glob("*/scores.json"))]
    for k, v in res.items():
        v = np.asarray(v, float)
        print(f"{k:16s} n={len(v):2d}  path err {np.nanmean(v):.3f} +- {np.nanstd(v) / np.sqrt(len(v)):.3f}  median {np.nanmedian(v):.3f}")
    (W / "ate_baselines.json").write_text(json.dumps({k: list(map(float, v)) for k, v in res.items()}))


if __name__ == "__main__":
    main()
