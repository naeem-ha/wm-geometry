"""Raw vs. motion-matched comparisons from work/matched.json.

For each model and matching estimator (SIFT displacement, dense Farneback flow):
  raw     = model - re-encoded real continuation (same clip)
  matched = model - same-episode real footage matched to the model clip's image motion
Paired bootstrap 95% CIs over accepted clips. A "reversal" is a metric where the raw comparison
favors the model and the matched comparison favors real footage (both CIs excluding zero).
Primary set: the 25 clips common to all models. Robustness: LTX and Ctrl-World on all 60 clips,
other frame-pair gaps, and the second motion estimator. Writes paper/tab_matched.tex and
work/matched_numbers.json.
"""
import json
from pathlib import Path

import numpy as np

from matched_controls import N15, STEP, score, to5
from wmgeo import io

ROOT = Path(__file__).resolve().parent
W = ROOT / "work"
MODELS = [("gen_ltx", "LTX"), ("gen_cogvideox_5b", "CogVideoX"), ("gen_wan22_5b", "Wan2.2"), ("gen_ctrlworld", "Ctrl-World")]
# (key, label, lower_is_better)
METRICS = [("fail", "Fail", True), ("inliers", "Inliers", False), ("vggt", "VGGT", True),
           ("vggt_f", "VGGT (filt.)", True), ("reg", "Reg.", False)]


def flat(s, gap=2):
    e = s["epipolar"][str(gap)] if str(gap) in s["epipolar"] else s["epipolar"][gap]
    v = s["vggt"][str(gap)] if str(gap) in s["vggt"] else s["vggt"][gap]
    return {"fail": e["match_fail_rate"], "inliers": e["n_inliers"], "sampson": e["sampson_med"],
            "vggt": v["err_default"], "vggt_f": v["err_filtered"], "cov": v["cov_default"], "reg": s["reg_rate"]}


def boot(d, n=10000, seed=0):
    d = np.asarray([x for x in d if x is not None and np.isfinite(x)], float)
    if len(d) < 3:
        return np.nan, np.nan, np.nan, len(d)
    rng = np.random.default_rng(seed)
    m = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    return float(d.mean()), float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5)), len(d)


def fmt(m):
    return f"{m:+.3f}" if abs(m) < 0.1 else (f"{m:+.1f}" if abs(m) >= 10 else f"{m:+.2f}")


def cell(m, lo, hi):
    if np.isnan(m):
        return "--"
    star = "$^*$" if lo > 0 or hi < 0 else ""
    return f"{fmt(m)}{star}"


def compare(r, clips, model, est, gap):
    raw, mat, errs, n_all = {k: [] for k, *_ in METRICS}, {k: [] for k, *_ in METRICS}, [], 0
    raw_acc = {k: [] for k, *_ in METRICS}
    for c in clips:
        key = f"{model}/{c}"
        if key not in r["model"]:
            continue
        n_all += 1
        mm = r["matched"].get(f"{key}/{est}")
        g, real = flat(r["model"][key], gap), flat(r["real"][c], gap)
        for k, *_ in METRICS:
            raw[k].append(g[k] - real[k])
        if mm and mm["accepted"]:
            errs.append(abs(mm["rel_err"]))
            ctl = flat(mm["scores"], gap)
            for k, *_ in METRICS:
                mat[k].append(g[k] - ctl[k])
                raw_acc[k].append(g[k] - real[k])
    # raw is reported on the same accepted clips as matched, so a sign change is a like-for-like reversal
    return {k: boot(raw_acc[k]) for k in raw_acc}, {k: boot(mat[k]) for k in mat}, len(errs), n_all, (float(np.median(errs)) if errs else np.nan)


def main():
    r = json.loads((W / "matched.json").read_text())
    r.setdefault("real", {})
    for c in sorted(r["candidates"]):
        if c not in r["real"]:
            r["real"][c] = score(io.reencode(to5(io.read_clip(W / "real" / c))))
            (W / "matched.json").write_text(json.dumps(r))
    common = sorted(d.name for d in (W / "gen_wan22_5b").iterdir() if (d / "meta.json").exists())
    allc = sorted(r["candidates"])
    out = {}
    lines = ["\\begin{tabular}{ll" + "c" * len(METRICS) + "c}", "\\toprule",
             "Model & Reference & " + " & ".join(m[1] for m in METRICS) + " & $n$ \\\\", "\\midrule"]
    for model, label in MODELS:
        raw, mat, n_ok, n_all, err = compare(r, common, model, "motion_sift", 2)
        out[f"{model}/common/sift/2"] = {"raw": raw, "matched": mat, "n_ok": n_ok, "n_all": n_all, "med_rel_err": err}
        lines.append(f"{label} & real & " + " & ".join(cell(*raw[k][:3]) for k, *_ in METRICS) + f" & {n_ok} \\\\")
        lines.append(f" & matched & " + " & ".join(cell(*mat[k][:3]) for k, *_ in METRICS) + f" & {n_ok} \\\\")
    fz = {k: [] for k, *_ in METRICS}
    for c in common:
        f, real = flat(r["frozen"][c]), flat(r["real"][c])
        for k, *_ in METRICS:
            fz[k].append(f[k] - real[k])
    fzb = {k: boot(fz[k]) for k in fz}
    out["frozen/common"] = fzb
    lines.append("\\midrule")
    lines.append("Frozen frame & real & " + " & ".join(cell(*fzb[k][:3]) for k, *_ in METRICS) + f" & {len(common)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper/tab_matched.tex").write_text("\n".join(lines), encoding="utf-8")

    # robustness: other gaps, dense-flow matching, all 60 clips for LTX and Ctrl-World
    for model, _ in MODELS:
        for est in ("motion_sift", "motion_dense"):
            for gap in (1, 2, 4):
                for name, clips in (("common", common), ("all", allc)):
                    if name == "all" and model in ("gen_cogvideox_5b", "gen_wan22_5b"):
                        continue
                    raw, mat, n_ok, n_all, err = compare(r, clips, model, est, gap)
                    out[f"{model}/{name}/{est.split('_')[1]}/{gap}"] = {"raw": raw, "matched": mat, "n_ok": n_ok,
                                                                      "n_all": n_all, "med_rel_err": err}
    (W / "matched_numbers.json").write_text(json.dumps(out, indent=1))

    # robustness table: VGGT raw -> matched across gaps and both motion estimators (common clips)
    L = ["\\begin{tabular}{llccc}", "\\toprule", "Model & Matching & 0.2\\,s & 0.4\\,s & 0.8\\,s \\\\", "\\midrule"]
    for model, label in MODELS:
        for est, elab in (("sift", "SIFT"), ("dense", "dense flow")):
            cells = []
            for g in (1, 2, 4):
                v = out[f"{model}/common/{est}/{g}"]
                cells.append(f"{cell(*v['raw']['vggt'][:3])} $\\to$ {cell(*v['matched']['vggt'][:3])}")
            L.append(f"{label if est == 'sift' else ''} & {elab} ($n$={out[f'{model}/common/{est}/2']['n_ok']}) & " + " & ".join(cells) + " \\\\")
    L += ["\\bottomrule", "\\end{tabular}"]
    (ROOT / "paper/tab_robust.tex").write_text("\n".join(L), encoding="utf-8")

    def show(k):
        v = out[k]
        s = f"{k:40s} n={v['n_ok']}/{v['n_all']} err={v['med_rel_err']:.2f} | "
        s += " ".join(f"{m}: raw {fmt(v['raw'][m][0])} [{v['raw'][m][1]:+.3f},{v['raw'][m][2]:+.3f}] "
                      f"matched {fmt(v['matched'][m][0])} [{v['matched'][m][1]:+.3f},{v['matched'][m][2]:+.3f}]"
                      for m in ("fail", "vggt", "reg") if not np.isnan(v["matched"][m][0]))
        print(s)
    for k in sorted(out):
        if not k.startswith("frozen"):
            show(k)
    print("frozen", {k: [round(x, 3) for x in v[:3]] for k, v in fzb.items()})


if __name__ == "__main__":
    main()
