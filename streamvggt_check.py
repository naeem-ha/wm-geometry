"""Second geometry scorer: StreamVGGT (Zheng et al., ICLR 2026), next to VGGT-1B.

VGGT sees all frames of a clip at once; StreamVGGT is its causal, streaming variant (frame t only
attends to frames <= t, through a key-value cache). Both score the same 26-frame, 5 fps tracks with
the paper's depth-reprojection disagreement (vggt_sensitivity.pair_scores, 0.4 s gap), on the clips
that every generator produced. Conditions: the real continuation, the frozen first frame, and the
four generators. VGGT runs first and is unloaded before StreamVGGT, so both fit in 16 GB.
Writes work/streamvggt_check.json; `--report` prints the comparison.
"""
import argparse
import gc
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.append(str(ROOT / "third_party/StreamVGGT/src"))  # appended, so the installed `vggt` keeps priority
from vggt_sensitivity import pair_scores  # noqa: E402
from wmgeo import io, vggt_eval  # noqa: E402

W = ROOT / "work"
OUT = W / "streamvggt_check.json"
GENS = ["gen_ltx", "gen_cogvideox_5b", "gen_wan22_5b", "gen_ctrlworld"]
CONDS = ["real", "frozen"] + GENS
N15, STEP, GAP = 76, 3, 2   # 5 s at 15 fps, every third frame (26 frames at 5 fps), gap 2 = 0.4 s

_stream = None


def _stream_model():
    global _stream
    if _stream is None:
        from streamvggt.models.streamvggt import StreamVGGT
        _stream = StreamVGGT.from_pretrained(str(ROOT / "models/lch01--StreamVGGT")).to("cuda").eval()
    return _stream


@torch.inference_mode()
def stream_predict(frame_paths):
    from streamvggt.utils.load_fn import load_and_preprocess_images
    from streamvggt.utils.pose_enc import pose_encoding_to_extri_intri
    images = load_and_preprocess_images([str(p) for p in frame_paths]).to("cuda")
    frames = [{"img": images[i].unsqueeze(0)} for i in range(images.shape[0])]
    with torch.autocast("cuda", dtype=torch.bfloat16):
        out = _stream_model().inference(frames)
    depth = torch.stack([r["depth"].squeeze(0) for r in out.ress])       # (S,H,W,1)
    conf = torch.stack([r["depth_conf"].squeeze(0) for r in out.ress])   # (S,H,W)
    pose = torch.stack([r["camera_pose"].squeeze(0) for r in out.ress])  # (S,9)
    extr, intr = pose_encoding_to_extri_intri(pose.unsqueeze(0).float(), images.shape[-2:])
    return {
        "extrinsic": extr[0].float().cpu().numpy(),
        "intrinsic": intr[0].float().cpu().numpy(),
        "depth": depth[..., 0].float().cpu().numpy(),
        "depth_conf": conf.float().cpu().numpy(),
    }


def clips():
    sets = [{d.name for d in (W / g).iterdir() if (d / "meta.json").exists()} for g in GENS]
    return sorted(set.intersection(*sets))


def track(cond, clip):
    """The 26-frame 5 fps track that matched_controls.py scores for this condition."""
    if cond == "frozen":
        return io.reencode([io.read_clip(W / "real" / clip)[0]] * len(range(0, N15, STEP)))
    return io.read_clip(W / cond / clip)[:N15][::STEP]


def scores(p):
    q = [pair_scores(p, i, i + GAP) for i in range(len(p["depth"]) - GAP)]
    return {"err_default": float(np.nanmedian([x["err_default"] for x in q])),
            "err_filtered": float(np.nanmedian([x["err_filtered"] for x in q])),
            "cov_default": float(np.nanmean([x["cov_default"] for x in q]))}


def run(limit=None, only=None):
    res = json.loads(OUT.read_text()) if OUT.exists() else {}
    todo = [(c, k) for k in clips()[:limit] for c in CONDS]
    for name, predict in (("vggt", vggt_eval.predict), ("stream", stream_predict)):
        if only and name != only:
            continue
        for cond, clip in todo:
            key = f"{cond}/{clip}"
            if name in res.get(key, {}):
                continue
            tmp = W / "_svggt_tmp"
            io.write_clip(track(cond, clip), tmp)
            try:
                p = predict(sorted((tmp / "frames").glob("*.png")))
            finally:
                shutil.rmtree(tmp, ignore_errors=True)
            res.setdefault(key, {})[name] = scores(p)
            OUT.write_text(json.dumps(res, indent=1))
            print(f"{name} {key} {res[key][name]['err_default']:.4f}", flush=True)
        if name == "vggt":  # free VGGT before loading StreamVGGT
            vggt_eval._model = None
            gc.collect()
            torch.cuda.empty_cache()


def report():
    from scipy.stats import spearmanr, wilcoxon
    res = json.loads(OUT.read_text())
    both = {k: v for k, v in res.items() if "vggt" in v and "stream" in v}
    ks = sorted(k for k in {key.split("/", 1)[1] for key in both}
                if all(f"{c}/{k}" in both for c in CONDS))  # only clips scored under every condition
    both = {key: v for key, v in both.items() if key.split("/", 1)[1] in ks}
    print(f"{len(ks)} clips x {len(CONDS)} conditions; depth disagreement at 0.4 s (median over clips)")
    for metric in ("err_default", "err_filtered"):
        print(f"\n{metric}:")
        for c in CONDS:
            v = [both[f"{c}/{k}"]["vggt"][metric] for k in ks if f"{c}/{k}" in both]
            s = [both[f"{c}/{k}"]["stream"][metric] for k in ks if f"{c}/{k}" in both]
            print(f"  {c:18s} VGGT {np.nanmedian(v):.4f}   StreamVGGT {np.nanmedian(s):.4f}")
        for name in ("vggt", "stream"):
            fr = np.array([both[f"frozen/{k}"][name][metric] for k in ks])
            re = np.array([both[f"real/{k}"][name][metric] for k in ks])
            ok = ~(np.isnan(fr) | np.isnan(re))
            p = wilcoxon(fr[ok], re[ok]).pvalue if ok.sum() > 5 else np.nan
            print(f"  {name}: frozen lower than real on {np.mean(fr[ok] < re[ok]):.0%} of clips (Wilcoxon p={p:.3g})")
        allv = [both[k]["vggt"][metric] for k in both]
        alls = [both[k]["stream"][metric] for k in both]
        ok = ~(np.isnan(allv) | np.isnan(alls))
        rho = spearmanr(np.array(allv)[ok], np.array(alls)[ok])
        print(f"  per clip-condition rank agreement: Spearman rho={rho.statistic:.2f} (n={ok.sum()})")
        rv = spearmanr([np.nanmedian([both[f"{c}/{k}"]["vggt"][metric] for k in ks]) for c in CONDS],
                       [np.nanmedian([both[f"{c}/{k}"]["stream"][metric] for k in ks]) for c in CONDS])
        print(f"  condition ranking agreement: Spearman rho={rv.statistic:.2f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int)
    ap.add_argument("--only", choices=["vggt", "stream"])
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report() if a.report else run(a.limit, a.only)
