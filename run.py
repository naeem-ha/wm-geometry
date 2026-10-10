"""Pipeline driver. Stages are resumable: each writes into work/<condition>/<clip>/ and is skipped if done.

  python run.py sample                      # real clips from DROID-100 wrist camera
  python run.py controls                    # real (re-encoded), warp dose-response, mix
  python run.py generate --model ltx        # one model at a time (each needs the whole GPU)
  python run.py score                       # COLMAP + epipolar + VGGT on every condition
  python run.py report                      # results/summary.csv and horizon plots
"""
import argparse
import json
import time
import zlib
from pathlib import Path

import cv2
import numpy as np

from wmgeo import controls, io

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
REAL = WORK / "real"
HORIZONS_S = (1, 2, 3, 5)  # scored as prefixes of each continuation
WARP_AMPS = (1, 2, 4, 8)   # px at EVAL resolution
SLOW_FACTORS = (2.5, 12)   # motion-matched to LTX (~7.4 px / 4 frames) and CogVideoX (~1.5 px)


def clips():
    return sorted(d for d in REAL.iterdir() if (d / "frames").exists())


def cmd_sample(a):
    from wmgeo.data_droid import build_gripper_mask, sample_clips
    n = sample_clips(ROOT / "data/droid_100", REAL, horizon_s=max(HORIZONS_S), n_clips=a.n)
    build_gripper_mask(clips())
    print(f"{n} clips in {REAL}")


def cmd_controls(a):
    pool = [io.read_clip(c) for c in clips()]
    for c, frames in zip(clips(), pool):
        meta = io.read_meta(c)
        seed = zlib.crc32(c.name.encode())  # stable across runs, unlike hash()
        conds = {"ctrl_real": controls.real(frames), "ctrl_mix": controls.mix(pool, len(frames), seed=seed)}
        for amp in WARP_AMPS:
            conds[f"ctrl_warp{amp}"] = controls.warp(frames, amp, seed=seed)
        for k in SLOW_FACTORS:
            conds[f"ctrl_slow{k:g}"] = controls.slow(frames, k)
        for name, fr in conds.items():
            out = WORK / name / c.name
            if not (out / "meta.json").exists():
                io.write_clip(fr, out, {**meta, "condition": name})
    print("controls done")


def cmd_generate_ctrlworld(a):
    """Action-conditioned rollouts; output is 5 Hz, stored at 15 fps by frame repetition."""
    from wmgeo.ctrlworld_gen import CtrlWorldRollout, SKIP, episode_inputs
    gen = CtrlWorldRollout()
    n5 = int(max(HORIZONS_S) * io.EVAL_FPS / SKIP) + 1  # 26 frames at 5 Hz for 5 s
    todo = clips()[: a.limit] if a.limit else clips()
    for c in todo:
        out = WORK / "gen_ctrlworld" / c.name
        if (out / "meta.json").exists():
            continue
        meta = io.read_meta(c)
        t = time.time()
        views, states, task = episode_inputs(ROOT / "data/droid_100", meta["episode"], meta["start_frame"],
                                             n_out=4 * ((n5 - 1 + 3) // 4) + 1)
        frames = gen.rollout(views, states, task)[:n5]
        frames = io.to_eval_res(io.resample_fps(frames, 5))[: int(max(HORIZONS_S) * io.EVAL_FPS) + 1]
        frames[0] = io.read_clip(c)[0]
        io.write_clip(frames, out, {**meta, "condition": "gen_ctrlworld", "native_fps": 5,
                                    "gen_seconds": time.time() - t, "action_conditioned": True})
        print(f"{c.name}: {time.time() - t:.0f}s, {len(frames)} frames", flush=True)


def cmd_generate(a):
    if a.model == "ctrlworld":
        return cmd_generate_ctrlworld(a)
    from wmgeo.generate import Generator
    gen = Generator(a.model, offload=not a.no_offload)
    todo = clips()[: a.limit] if a.limit else clips()
    for c in todo:
        out = WORK / f"gen_{a.model}" / c.name
        if (out / "meta.json").exists():
            continue
        meta = io.read_meta(c)
        first = io.read_clip(c)[0]
        t = time.time()
        frames, fps = gen(first, meta.get("task", ""), seed=a.seed)
        frames = io.to_eval_res(io.resample_fps(frames, fps))
        frames[0] = first  # conditioning frame is shared by every condition
        io.write_clip(frames, out, {**meta, "condition": f"gen_{a.model}", "native_fps": fps,
                                    "gen_seconds": time.time() - t, "seed": a.seed})
        print(f"{c.name}: {time.time() - t:.0f}s, {len(frames)} frames", flush=True)


def _real_centers(clip_name, horizon_s, tmp_frames):
    """VGGT camera centers of the real continuation, cached per clip and horizon."""
    from wmgeo import vggt_eval
    cache = REAL / clip_name / f"vggt_centers_h{horizon_s}.npy"
    if not cache.exists():
        _, p = vggt_eval.clip_stats(tmp_frames)
        np.save(cache, vggt_eval.centers(p["extrinsic"]))
    return np.load(cache)


def _score_one(clip_dir, mask_png, horizon_s, prev, use_vggt, step=1):
    """Fills in whatever `prev` lacks, so a later run with VGGT only adds the VGGT part.
    step=3 scores the 5 fps track (every 3rd frame), where no condition has repeated frames."""
    import shutil
    from wmgeo import colmap_eval, epipolar, vggt_eval
    n = int(horizon_s * io.EVAL_FPS) + 1
    frames = io.read_clip(clip_dir)[:n][::step]
    gaps = epipolar.GAPS if step == 1 else (1, 2, 4)  # 5 fps gaps = 0.2 / 0.4 / 0.8 s
    tmp = clip_dir.parent / f"_h{horizon_s}_s{step}_{clip_dir.name}"  # prefix copy for SfM/VGGT
    io.write_clip(frames, tmp)
    r = dict(prev)
    try:
        if "reg_rate" not in r:
            r.update(colmap_eval.run(tmp, mask_png))
            r.pop("centers")
            r["epipolar"] = epipolar.clip_stats(frames, cv2.imread(str(mask_png), 0), gaps=gaps)
        if use_vggt and "vggt" not in r:
            r["vggt"], p = vggt_eval.clip_stats(tmp)
            if clip_dir.parent.name.startswith("gen_"):
                real_tmp = REAL.parent / f"_h{horizon_s}_real_{clip_dir.name}"
                io.write_clip(io.read_clip(REAL / clip_dir.name)[:n], real_tmp)
                try:
                    ref = _real_centers(clip_dir.name, horizon_s, real_tmp)
                finally:
                    shutil.rmtree(real_tmp, ignore_errors=True)
                # only meaningful for action-conditioned models, which are given the real actions
                r["vggt"]["ate_vs_real"] = vggt_eval.ate_sim3(vggt_eval.centers(p["extrinsic"]), ref)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return r


def cmd_score(a):
    mask_png = ROOT / "data/droid_gripper_mask.png"
    step, fname = (3, "scores5.json") if a.fps5 else (1, "scores.json")
    use_vggt = not (a.no_vggt or a.fps5)  # VGGT already subsamples to 5 fps internally
    conds = [d for d in sorted(WORK.iterdir()) if d.is_dir() and (d.name in a.only if a.only else True)
             and (d.name == "real" or d.name.startswith(("ctrl_", "gen_")))]
    for cond in conds:
        for clip_dir in sorted(cond.iterdir())[: a.limit or None]:
            if not (clip_dir / "frames").exists():
                continue
            out = clip_dir / fname
            scores = json.loads(out.read_text()) if out.exists() else {}
            for h in HORIZONS_S:
                prev = scores.get(str(h), {})
                if "reg_rate" in prev and (not use_vggt or "vggt" in prev):
                    continue
                scores[str(h)] = _score_one(clip_dir, mask_png, h, prev, use_vggt, step)
                out.write_text(json.dumps(scores, indent=1, default=float))
            print(f"{cond.name}/{clip_dir.name} scored", flush=True)


def _flatten(d, prefix=""):
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        else:
            out[key] = v
    return out


def cmd_report(a):
    import pandas as pd
    rows = []
    for f in WORK.glob("*/*/scores.json"):
        for h, s in json.loads(f.read_text()).items():
            rows.append({"condition": f.parent.parent.name, "clip": f.parent.name, "horizon_s": int(h), **_flatten(s)})
    df = pd.DataFrame(rows)
    (ROOT / "results").mkdir(exist_ok=True)
    df.to_csv(ROOT / "results/scores_long.csv", index=False)
    cols = ["reg_rate", "reg_rate_any", "n_models", "reproj_err", "track_len", "n_points3D",
            "epipolar.4.inlier_ratio", "epipolar.4.n_inliers", "epipolar.4.match_fail_rate", "epipolar.4.sampson_med", "epipolar.4.flow_px",
            "vggt.6.depth_reproj_rel_med"]
    cols = [c for c in cols if c in df]
    summ = df.groupby(["condition", "horizon_s"])[cols].agg(["mean", "sem"]).round(3)
    summ.to_csv(ROOT / "results/summary.csv")
    print(summ.xs("mean", axis=1, level=1).to_string())
    _plots(df, cols)


def _plots(df, cols):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(cols), figsize=(3.2 * len(cols), 3), squeeze=False)
    for ax, c in zip(axes[0], cols):
        for cond, g in df.groupby("condition"):
            if cond.startswith("ctrl_warp") and cond not in ("ctrl_warp2", "ctrl_warp8"):
                continue
            m = g.groupby("horizon_s")[c].agg(["mean", "sem"])
            ax.errorbar(m.index, m["mean"], yerr=m["sem"], label=cond, marker="o", ms=3, capsize=2)
        ax.set_title(c, fontsize=8)
        ax.set_xlabel("horizon (s)")
    axes[0][0].legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(ROOT / "results/horizon_curves.png", dpi=150)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample"); s.add_argument("--n", type=int, default=60)
    sub.add_parser("controls")
    g = sub.add_parser("generate")
    g.add_argument("--model", required=True)
    g.add_argument("--limit", type=int, default=0)
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--no-offload", action="store_true")
    sc = sub.add_parser("score"); sc.add_argument("--only", nargs="*"); sc.add_argument("--no-vggt", action="store_true")
    sc.add_argument("--limit", type=int, default=0)
    sc.add_argument("--fps5", action="store_true", help="score every 3rd frame into scores5.json")
    sub.add_parser("report")
    a = p.parse_args()
    globals()[f"cmd_{a.cmd}"](a)
