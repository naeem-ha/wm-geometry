"""Qualitative figures: rows = conditions, columns = time, labels outside the images.

  python make_figure.py droid_ep005 paper/fig_qual_ep005.pdf
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from wmgeo import io

W = Path(__file__).resolve().parent / "work"
ROWS = [("real", "Real"), ("gen_ctrlworld", "Ctrl-World"), ("gen_wan22_5b", "Wan2.2-5B"),
        ("gen_cogvideox_5b", "CogVideoX-5B"), ("gen_ltx", "LTX-Video")]
TIMES = (0, 1, 2, 3, 4, 5)  # seconds


def main(clip, out):
    fig, axes = plt.subplots(len(ROWS), len(TIMES), figsize=(7.0, 3.3), gridspec_kw=dict(wspace=0.03, hspace=0.06))
    for r, (cond, label) in enumerate(ROWS):
        frames = io.read_clip(W / cond / clip)
        for c, t in enumerate(TIMES):
            ax = axes[r, c]
            ax.imshow(frames[min(t * io.EVAL_FPS, len(frames) - 1)])
            ax.set_xticks([]), ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
            if r == 0:
                ax.set_title(f"{t} s", fontsize=8, pad=2)
            if c == 0:
                ax.set_ylabel(label, fontsize=8, rotation=0, ha="right", va="center", labelpad=4)
    fig.savefig(out, bbox_inches="tight", dpi=200)
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
