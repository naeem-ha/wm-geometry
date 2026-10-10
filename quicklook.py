"""Paired comparison on whatever has been scored so far (clips present in every listed condition)."""
import json
import sys
from pathlib import Path

import pandas as pd

WORK = Path(__file__).resolve().parent / "work"
conds = sys.argv[1:] or ["ctrl_real", "ctrl_warp2", "ctrl_warp8", "ctrl_mix", "gen_ltx"]
rows = []
for f in WORK.glob("*/*/scores.json"):
    for h, s in json.loads(f.read_text()).items():
        e = s["epipolar"]["4"]
        v = s.get("vggt", {}).get("6", {})
        rows.append(dict(cond=f.parent.parent.name, clip=f.parent.name, h=int(h), reg=s["reg_rate"],
                         reproj=s["reproj_err"], track=s["track_len"], inliers=e["n_inliers"],
                         fail=e["match_fail_rate"], sampson=e["sampson_med"], flow=e["flow_px"],
                         vggt=v.get("depth_reproj_rel_med")))
d = pd.DataFrame(rows)
d = d[d.cond.isin(conds)]
common = set.intersection(*(set(d[d.cond == c]["clip"]) for c in conds if (d.cond == c).any()))
d = d[d["clip"].isin(common)]
print(f"paired clips: {len(common)}")
cols = ["reg", "reproj", "track", "inliers", "fail", "sampson", "flow", "vggt"]
print("\n5 s horizon")
print(d[d.h == 5].groupby("cond")[cols].mean().reindex(conds).round(3).to_string())
for c in conds:
    if c.startswith("gen_"):
        print(f"\n{c} by horizon")
        print(d[d.cond == c].groupby("h")[cols].mean().round(3).to_string())
