"""Every number in the paper, from one clip set (the clips common to all conditions).

Settings match the main experiment throughout: 5 s horizon; COLMAP and epipolar metrics on the
5 fps track with an epipolar gap of 2 frames (0.4 s); VGGT depth disagreement at 0.4 s and camera
path error from work/vggt_sens.json. Differences are model minus control on the same clips, with
paired bootstrap 95% CIs (10,000 resamples). Writes paper/tab_main.tex, paper/tab_calib.tex,
paper/tab_diff.tex and work/numbers.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from wmgeo import vggt_eval

ROOT = Path(__file__).resolve().parent
W = ROOT / "work"
H = "5"
LABEL = {
    "ctrl_real": "Real (re-encoded)", "ctrl_slow2.5": "Real, slowed 2.5$\\times$",
    "ctrl_blur0.7": "Real, blurred", "ctrl_warp1": "Real, warp 1\\,px", "ctrl_warp2": "Real, warp 2\\,px",
    "ctrl_warp4": "Real, warp 4\\,px", "ctrl_warp8": "Real, warp 8\\,px", "ctrl_mix": "Unrelated frames",
    "gen_ltx": "LTX-Video", "gen_cogvideox_5b": "CogVideoX-5B", "gen_wan22_5b": "Wan2.2-5B",
    "gen_ctrlworld": "Ctrl-World",
}
MAIN_ROWS = ["ctrl_real", "ctrl_slow2.5", "ctrl_blur0.7", "ctrl_warp8", "ctrl_mix",
             "gen_ltx", "gen_cogvideox_5b", "gen_wan22_5b", "gen_ctrlworld"]
CALIB_ROWS = ["ctrl_real", "ctrl_warp1", "ctrl_warp2", "ctrl_warp4", "ctrl_warp8", "ctrl_mix"]
# model vs the control that matches it on the relevant nuisance factor
PAIRS = [("gen_ltx", "ctrl_slow2.5", "LTX $-$ slowed real (matched motion)"),
         ("gen_cogvideox_5b", "ctrl_real", "CogVideoX $-$ real"),
         ("gen_wan22_5b", "ctrl_real", "Wan2.2 $-$ real"),
         ("gen_ctrlworld", "ctrl_real", "Ctrl-World $-$ real"),
         ("gen_ctrlworld", "ctrl_blur0.7", "Ctrl-World $-$ blurred real (matched sharpness)")]


def load():
    sens = json.loads((W / "vggt_sens.json").read_text())
    clips = sorted(sens["_real_centers"])
    rows = []
    for cond in LABEL:
        for c in clips:
            f = W / cond / c / "scores5.json"
            s = json.loads(f.read_text())[H] if f.exists() else {}
            e = s.get("epipolar", {}).get("2", {})
            v = sens.get(cond, {}).get(c, {})
            rows.append({"cond": cond, "clip": c, "motion": e.get("flow_px"), "reg": s.get("reg_rate"),
                         "reproj": s.get("reproj_err"), "fail": e.get("match_fail_rate"),
                         "inliers": e.get("n_inliers"), "sampson": e.get("sampson_med"),
                         "vggt": v.get("err_default"), "vggt_f": v.get("err_filtered"),
                         "cov": v.get("cov_default"), "cov_f": v.get("cov_filtered"), "path": v.get("path_err")})
    return pd.DataFrame(rows).astype({k: float for k in ("motion", "reg", "reproj", "fail", "inliers",
                                                         "sampson", "vggt", "vggt_f", "cov", "cov_f", "path")}), sens


def path_baselines(sens):
    ref = {c: np.array(v) for c, v in sens["_real_centers"].items()}
    keys = sorted(ref)
    frozen = [np.sqrt(((r - r.mean(0)) ** 2).sum(1).mean()) / np.linalg.norm(np.diff(r, axis=0), axis=1).sum()
              for r in ref.values()]
    other = [vggt_eval.ate_sim3(ref[o], ref[k]) for k in keys for o in keys if o != k]  # every other clip
    return np.array(frozen), np.array(other)


def fmt(m, s=None):
    nd = 3 if abs(m) < 0.1 else (1 if abs(m) >= 10 else 2)
    return f"{m:.{nd}f}" if s is None else f"{m:.{nd}f} $\\pm$ {s:.{nd}f}"


def cell(g, col):
    v = g[col].dropna()
    return "--" if v.empty else fmt(v.mean(), v.sem())


def boot_ci(a, b, n=10000, seed=0):
    d = (a - b).dropna().to_numpy()
    if len(d) < 3:
        return np.nan, np.nan, np.nan, len(d)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return d.mean(), *np.percentile(means, [2.5, 97.5]), len(d)


def main():
    d, sens = load()
    clips = sorted(d["clip"].unique())
    frozen, other = path_baselines(sens)
    nums = {"n_clips": len(clips), "path_frozen": float(frozen.mean()), "path_other_clip": float(other.mean()),
            "path_other_clip_p10": float(np.percentile(other, 10))}

    # main table
    cols = [("motion", "Motion (px)"), ("reg", "Registered"), ("fail", "Match fail"), ("inliers", "Inliers"),
            ("sampson", "Sampson (px)"), ("vggt", "VGGT depth err."), ("cov", "Coverage"), ("path", "Path err.")]
    L = ["\\begin{tabular}{l" + "c" * len(cols) + "}", "\\toprule", " & " + " & ".join(c[1] for c in cols) + " \\\\", "\\midrule"]
    for i, cond in enumerate(MAIN_ROWS):
        if cond == "gen_ltx":
            L.append("\\midrule")
        g = d[d.cond == cond]
        L.append(LABEL[cond] + " & " + " & ".join(cell(g, c) for c, _ in cols) + " \\\\")
    L.append("\\midrule")
    L.append("\\multicolumn{" + str(len(cols)) + "}{l}{\\emph{Path-error references:} frozen camera "
             + fmt(frozen.mean()) + "; real path of a different clip " + fmt(other.mean()) + "} & \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper/tab_main.tex").write_text("\n".join(L), encoding="utf-8")

    # calibration table on the same clips and settings
    ccols = [("reproj", "Reprojection error (px) $\\downarrow$"), ("sampson", "Median Sampson error (px) $\\downarrow$"),
             ("inliers", "Epipolar inliers per pair $\\uparrow$"), ("fail", "Pairs failing to match $\\downarrow$"),
             ("vggt", "VGGT depth disagreement $\\downarrow$"), ("vggt_f", "VGGT, visibility-filtered $\\downarrow$"),
             ("reg", "Registration rate")]
    L = ["\\begin{tabular}{l" + "c" * len(CALIB_ROWS) + "}", "\\toprule",
         " & " + " & ".join(["Real", "Warp-1", "Warp-2", "Warp-4", "Warp-8", "Unrelated"]) + " \\\\", "\\midrule"]
    for col, lab in ccols:
        vals = []
        for cond in CALIB_ROWS:
            v = d[d.cond == cond][col].dropna()
            vals.append("--" if len(v) < 5 else fmt(v.mean()))
        L.append(lab + " & " + " & ".join(vals) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper/tab_calib.tex").write_text("\n".join(L), encoding="utf-8")

    # paired differences with bootstrap CIs
    dcols = [("reg", "Registered"), ("fail", "Match fail"), ("vggt", "VGGT err."), ("vggt_f", "VGGT err.\\ (filtered)"), ("path", "Path err.")]
    L = ["\\begin{tabular}{l" + "c" * len(dcols) + "}", "\\toprule", " & " + " & ".join(c[1] for c in dcols) + " \\\\", "\\midrule"]
    nums["diffs"] = {}
    for a, b, lab in PAIRS:
        A = d[d.cond == a].set_index("clip")
        B = d[d.cond == b].set_index("clip")
        cells = []
        for col, _ in dcols:
            m, lo, hi, n = boot_ci(A[col], B[col])
            nums["diffs"][f"{a}-{b}-{col}"] = [m, lo, hi, n]
            cells.append("--" if np.isnan(m) else f"{fmt(m)} [{fmt(lo)}, {fmt(hi)}]")
        L.append(lab + " & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper/tab_diff.tex").write_text("\n".join(L), encoding="utf-8")

    # path error against each clip's own unrelated-path baseline (mean over all other clips' real paths)
    ref = {c: np.array(v) for c, v in sens["_real_centers"].items()}
    base = pd.Series({k: np.mean([vggt_eval.ate_sim3(ref[o], ref[k]) for o in clips if o != k]) for k in clips})
    nums["path_vs_unrelated"] = {}
    for cond in ["ctrl_real", "ctrl_slow2.5", "gen_ltx", "gen_cogvideox_5b", "gen_wan22_5b", "gen_ctrlworld"]:
        p = d[d.cond == cond].set_index("clip")["path"]
        m, lo, hi, n = boot_ci(p, base)
        nums["path_vs_unrelated"][cond] = [m, lo, hi, n, float((p - base).lt(0).mean())]
        print(f"path - unrelated baseline {cond:18s} {m:+.3f} [{lo:+.3f}, {hi:+.3f}] better on {(p - base).lt(0).mean():.0%}")

    summ = d.groupby("cond")[["motion", "reg", "reproj", "fail", "inliers", "sampson", "vggt", "vggt_f", "cov", "cov_f", "path"]]
    nums["mean"] = summ.mean().round(4).to_dict(orient="index")
    nums["count"] = summ.count().to_dict(orient="index")
    (W / "numbers.json").write_text(json.dumps(nums, indent=1, default=float))
    pd.set_option("display.width", 200)
    print(f"{len(clips)} clips")
    print(summ.mean().reindex(MAIN_ROWS + ["ctrl_warp1", "ctrl_warp2", "ctrl_warp4"]).round(3).to_string())
    print("\ncounts (valid clips per metric)")
    print(summ.count().reindex(MAIN_ROWS).to_string())
    print(f"\npath refs: frozen {frozen.mean():.3f}, other clip {other.mean():.3f}")
    for k, v in nums["diffs"].items():
        print(k, [round(x, 3) if isinstance(x, float) else x for x in v])


if __name__ == "__main__":
    main()
