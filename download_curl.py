"""Resumable parallel download over plain HTTPS into models/<repo>/ (curl -C -).

hf_xet ran at ~1 MB/s on this connection and restarted partial files from zero; curl gets ~4 MB/s
per stream and resumes. Loaders take the local folder (models/<org>--<name>) instead of a repo id.

  python download_curl.py              # everything in QUEUE, in order
  python download_curl.py vggt ltx     # selected keys
"""
import fnmatch
import json
import subprocess
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent / "models"
DIFFUSERS = ["model_index.json", "scheduler/*", "tokenizer/*", "vae/*", "transformer/*"]
QUEUE = {
    "vggt": ("facebook/VGGT-1B", ["config.json", "model.safetensors"]),
    # StreamVGGT (ICLR 2026, Wenzhao Zheng et al.): causal streaming variant of VGGT, a second scorer
    "streamvggt": ("lch01/StreamVGGT", ["config.json", "model.safetensors"]),
    "t5": ("zai-org/CogVideoX-5b-I2V", ["text_encoder/*"]),  # shared by CogVideoX and LTX
    "ltx": ("Lightricks/LTX-Video", DIFFUSERS),
    "cogvideox_5b": ("zai-org/CogVideoX-5b-I2V", DIFFUSERS),
    "cosmos2_2b": ("nvidia/Cosmos-Predict2-2B-Video2World", DIFFUSERS + ["text_encoder/*"]),
    # Ctrl-World (ICLR 2026): action-conditioned world model trained on DROID, built on SVD + CLIP
    "ctrlworld": ("yjguo/Ctrl-World", ["config.json", "checkpoint-10000.pt"]),
    "svd": ("stabilityai/stable-video-diffusion-img2vid",
            ["model_index.json", "scheduler/*", "feature_extractor/*", "unet/config.json",
             "unet/diffusion_pytorch_model.safetensors", "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
             "image_encoder/config.json", "image_encoder/model.safetensors"]),
    "clip": ("openai/clip-vit-base-patch32", ["*.json", "*.txt", "pytorch_model.bin"]),
    "wan22_5b": ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", DIFFUSERS + ["text_encoder/*", "image_processor/*", "image_encoder/*"]),
}


def local_dir(repo):
    return ROOT / repo.replace("/", "--")


def _token():
    p = Path.home() / ".cache/huggingface/token"
    return p.read_text().strip() if p.exists() else None


def _files(repo, patterns):
    req = urllib.request.Request(f"https://huggingface.co/api/models/{repo}/tree/main?recursive=true")
    if _token():
        req.add_header("Authorization", f"Bearer {_token()}")
    tree = json.load(urllib.request.urlopen(req))
    return [(f["path"], f.get("size", 0)) for f in tree if f["type"] == "file"
            and any(fnmatch.fnmatch(f["path"], p) for p in patterns)]


PART = 256 * 2**20  # files larger than this are fetched as parallel byte ranges


def _curl(url, dst):
    cmd = ["curl", "-sSL", "--fail", "--retry", "20", "--retry-delay", "5", "--retry-all-errors",
           "-C", "-", "-o", str(dst), url]
    if _token():
        cmd[1:1] = ["-H", f"Authorization: Bearer {_token()}"]
    return subprocess.run(cmd, capture_output=True, text=True).stderr.strip()[:200]


def _range(url, dst, lo, hi, tries=20):
    """Fetch bytes lo..hi into dst, appending to whatever a previous attempt already wrote."""
    import time
    err = ""
    for _ in range(tries):
        have = dst.stat().st_size if dst.exists() else 0
        if have >= hi - lo + 1:
            return ""
        req = urllib.request.Request(url, headers={"Range": f"bytes={lo + have}-{hi}"})
        if _token():
            req.add_header("Authorization", f"Bearer {_token()}")
        try:
            with urllib.request.urlopen(req, timeout=60) as r, open(dst, "ab") as f:
                while chunk := r.read(2**20):
                    f.write(chunk)
        except Exception as e:
            err = f"{type(e).__name__}: {e}"
            time.sleep(5)
    return "" if dst.exists() and dst.stat().st_size >= hi - lo + 1 else err[:200]


def _get(repo, path, size, ranges=8):
    dst = local_dir(repo) / path
    if dst.exists() and dst.stat().st_size == size:
        return f"have {path}"
    dst.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{repo}/resolve/main/{path}"
    if size <= PART:
        err = _curl(url, dst)
    else:
        bounds = [(i * size // ranges, (i + 1) * size // ranges - 1) for i in range(ranges)]
        parts = [dst.with_name(f"{dst.name}.part{i}") for i in range(ranges)]
        with ThreadPoolExecutor(ranges) as ex:
            errs = list(ex.map(lambda b: _range(url, b[1], *b[0]), zip(bounds, parts)))
        err = "; ".join(e for e in errs if e)
        if all(p.exists() and p.stat().st_size == hi - lo + 1 for p, (lo, hi) in zip(parts, bounds)):
            with open(dst, "wb") as out:
                for p in parts:
                    with open(p, "rb") as f:
                        while chunk := f.read(64 * 2**20):
                            out.write(chunk)
            for p in parts:
                p.unlink()
    ok = dst.exists() and dst.stat().st_size == size
    return f"{'OK' if ok else 'FAIL'} {path} {size / 1e9:.2f} GB {err}"


if __name__ == "__main__":
    keys = sys.argv[1:] or list(QUEUE)
    for k in keys:
        repo, patterns = QUEUE[k]
        try:
            files = _files(repo, patterns)
        except Exception as e:  # gated repo without accepted license, network error
            print(f"FAIL {k}: cannot list {repo}: {e}", flush=True)
            continue
        print(f"== {k}: {len(files)} files, {sum(s for _, s in files) / 1e9:.1f} GB", flush=True)
        with ThreadPoolExecutor(2) as ex:  # 2 files x 8 ranges
            for line in ex.map(lambda f: _get(repo, *f), files):
                print(line, flush=True)
        print(f"DONE {k}", flush=True)
