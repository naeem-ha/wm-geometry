"""Main results table for the paper: 5 fps track, 5 s horizon, clips scored in every listed condition.

  python make_tables.py            # prints LaTeX and writes paper/results_table.tex
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
WORK = ROOT / "work"
ROWS = [  # (condition, label, group)
    ("ctrl_real", "Real (re-encoded)", "ref"), ("ctrl_slow2.5", "Real, slowed 2.5$\\times$", "ref"),
    ("ctrl_blur0.7", "Real, blurred ($\\sigma{=}0.7$)", "ref"), ("ctrl_warp8", "Real, warp 8\\,px", "ref"),
    ("ctrl_mix", "Unrelated frames", "ref"),
    ("gen_ltx", "LTX-Video", "gen"), ("gen_cogvideox_5b", "CogVideoX-5B", "gen"),
    ("gen_wan22_5b", "Wan2.2-5B", "gen"), ("gen_ctrlworld", "Ctrl-World (action-cond.)", "gen"),
]
# (column, label, track, getter); epipolar gap 2 on the 5 fps track = 0.4 s
COLS = [
    ("flow", "Motion (px)", "5", lambda s: s["epipolar"]["2"]["flow_px"]),
    ("reg", "Registered", "5", lambda s: s["reg_rate"]),
    ("fail", "Match fail", "5", lambda s: s["epipolar"]["2"]["match_fail_rate"]),
    ("inl", "Inliers", "5", lambda s: s["epipolar"]["2"]["n_inliers"]),
    ("samp", "Sampson (px)", "5", lambda s: s["epipolar"]["2"]["sampson_med"]),
    ("vggt", "VGGT depth err.", "15", lambda s: s["vggt"]["6"]["depth_reproj_rel_med"]),
    ("ate", "Path err.", "15", lambda s: s["vggt"].get("ate_vs_real", np.nan)),
]


def load(h="5"):
    rows = []
    for cond, *_ in ROWS:
        for d in sorted((WORK / cond).glob("*")):
            rec = {"cond": cond, "clip": d.name}
            for track, fname in (("5", "scores5.json"), ("15", "scores.json")):
                f = d / fname
                s = json.loads(f.read_text()).get(h) if f.exists() else None
                for col, _, tr, get in COLS:
                    if tr == track:
                        try:
                            rec[col] = get(s)
                        except (KeyError, TypeError):
                            rec[col] = np.nan
            rows.append(rec)
    return pd.DataFrame(rows)


def main():
    d = load()
    have = d.dropna(subset=["reg"]).groupby("clip").cond.nunique()
    common = have[have == len(ROWS)].index
    d = d[d["clip"].isin(common)]
    lines = [f"% {len(common)} clips scored in every condition; mean $\\pm$ s.e.m.",
             "\\begin{tabular}{l" + "c" * len(COLS) + "}", "\\toprule",
             " & " + " & ".join(c[1] for c in COLS) + " \\\\", "\\midrule"]
    for i, (cond, label, group) in enumerate(ROWS):
        if i and group != ROWS[i - 1][2]:
            lines.append("\\midrule")
        g = d[d.cond == cond]
        cells = []
        for col, *_ in COLS:
            v = g[col].dropna()
            cells.append("--" if v.empty else f"{v.mean():.2f} $\\pm$ {v.sem():.2f}")
        lines.append(f"{label} & " + " & ".join(cells) + " \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    tex = "\n".join(lines)
    (ROOT / "paper/results_table.tex").write_text(tex, encoding="utf-8")
    print(tex)


if __name__ == "__main__":
    main()
