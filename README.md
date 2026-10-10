# Geometric consistency of video world models (CoRL 2026 workshop paper A)

Target: *Do Robots Need World Models?* workshop, deadline **Oct 12, 2026**.

The test: give a video model one real robot wrist-camera frame and let it predict the next few
seconds. Then ask whether a single rigid 3D scene can explain the generated frames. COLMAP, VGGT
and two-view epipolar geometry answer that, and every score is compared against the real
continuation and against controls.

## Setup (this machine: RTX 5080 Laptop, 16 GB VRAM, 32 GB RAM)

```bash
python -m venv .venv
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/pip install diffusers transformers accelerate sentencepiece protobuf ftfy "imageio[ffmpeg]" av opencv-python pandas pyarrow huggingface_hub pycolmap scipy matplotlib einops tqdm
.venv/Scripts/pip install git+https://github.com/facebookresearch/vggt.git git+https://github.com/cvg/LightGlue.git
.venv/Scripts/pip install --no-deps "numpy>=2.0,<2.4" "opencv-python>=4.12"   # vggt pins old numpy
.venv/Scripts/python download_curl.py vggt t5 ltx cogvideox_5b wan22_5b ctrlworld svd clip   # resumable, parallel byte ranges

# Ctrl-World code (not vendored) and the pinned deps its pipelines need
curl -L https://github.com/Robert-gyj/Ctrl-World/archive/refs/heads/main.zip -o ctrlworld.zip   # unzip to third_party/Ctrl-World
.venv/Scripts/pip install --no-deps --target third_party/pydeps_ctrlworld diffusers==0.34.0 transformers==4.53.3 tokenizers==0.21.4 huggingface_hub==0.34.4
```

Data: the DROID-100 sample (`lerobot/droid_100`, 100 episodes, 320x180 @ 15 fps) is in `data/droid_100`.

## Pipeline

```bash
.venv/Scripts/python run.py sample               # 60 wrist-camera clips, 5 s, highest camera motion per episode
.venv/Scripts/python run.py controls             # ctrl_real, ctrl_warp{1,2,4,8}, ctrl_mix
.venv/Scripts/python run.py generate --model ltx # also: cogvideox_5b, wan22_5b, ctrlworld
.venv/Scripts/python run.py score                # COLMAP + epipolar + VGGT at horizons 1/2/3/5 s
.venv/Scripts/python run.py report               # results/summary.csv, results/horizon_curves.png
```

Every stage is resumable and writes into `work/<condition>/<clip>/`.

## Design decisions

- **Wrist camera only.** DROID's exterior cameras sit on tripods, so they give SfM no baseline.
- **Fixed gripper mask** (`data/droid_gripper_mask.png`). The gripper never moves in the wrist
  view, and SfM would otherwise lock onto it. The mask is learned once from all clips and applied
  to every condition.
- **Common evaluation format.** Every clip is scored at 320x180, 15 fps and H.264, so generator
  output size, frame rate and compression cannot explain the differences.
- **Horizons are prefixes** of one 5 s generation (1, 2, 3, 5 s), not separate generations.
- **No trajectory error against the real camera path.** Image- and text-conditioned models don't
  control the camera, so a different path is legitimate. All metrics measure self-consistency.
- **Controls:** `ctrl_real` (true future, re-encoded) is the upper bound. `ctrl_warp*` are smooth
  per-frame non-rigid warps, a dose-response check that the metrics detect geometric distortion.
  `ctrl_mix` (frames from unrelated clips) is the geometry-free floor.
- COLMAP is tuned for low resolution: SIFT `first_octave=-1`, 40 initial inliers, 15 per registration.

## Metrics

| Metric | Source | Meaning |
|---|---|---|
| `reg_rate`, `reg_rate_any`, `n_models` | COLMAP | fraction of frames in the largest model / in any model; how many pieces the clip broke into |
| `reproj_err`, `track_len`, `n_points3D` | COLMAP | quality of the model that was built |
| `epipolar.{1,4,8}.inlier_ratio`, `sampson_med` | SIFT + MAGSAC F-matrix | two-view rigidity at frame gaps 1/4/8; works even when SfM fails |
| `epipolar.*.flow_px` | same | motion magnitude; near-static clips are uninformative |
| `vggt.{1,4,8}.depth_reproj_rel_med` | VGGT-1B | median relative disagreement when frame i's depth is reprojected into frame j |

### Control check (Oct 5, 15 clips, COLMAP + epipolar)

At the 5 s horizon, going from real to warp1, warp2, warp4 and warp8:
- `reproj_err` rises 0.71 (ctrl_real) → 0.71 → 0.90 → 1.04 → 1.37 px
- `epipolar.4.sampson_med` rises 0.29 → 0.32 → 0.35 → 0.39 → 0.45
- `track_len` falls 9.1 → 8.6 → 9.2 → 6.8 → 5.7

These metrics respond to non-rigid distortion; lead with them.

`reg_rate` and `n_points3D` do **not** respond. Warped clips still register, and broken tracks
produce *more* (shorter) 3D points. They only detect total failure: ctrl_mix registers < 1%.
`sampson_med` must be read together with `n_inliers` and `match_fail_rate`. ctrl_mix has a low
Sampson error only because it barely matches at all (14 inliers vs 55; 83% of pairs fail to match).

## Models (diffusers 0.40, 16 GB with CPU offload)

| key | repo | native output |
|---|---|---|
| `ltx` | Lightricks/LTX-Video (uses CogVideoX's bf16 T5) | 704x480, 121 frames @ 24 fps |
| `cogvideox_5b` | zai-org/CogVideoX-5b-I2V | 720x480, 49 @ 8 fps |
| `cosmos2_2b` | nvidia/Cosmos-Predict2-2B-Video2World (license must be accepted on HF) | 1280x704, 93 @ 16 fps |
| `wan22_5b` | Wan-AI/Wan2.2-TI2V-5B-Diffusers | 1280x704, 121 @ 24 fps |
