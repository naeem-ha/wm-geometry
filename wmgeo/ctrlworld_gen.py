"""Action-conditioned rollouts with Ctrl-World (Guo et al., ICLR 2026; github.com/Robert-gyj/Ctrl-World).

Ctrl-World is trained on DROID. It sees all three cameras stacked vertically (exterior 1,
exterior 2, wrist; 320x192 each) at 5 Hz, and predicts 1 s chunks (5 frames, the first being the
current one) conditioned on 7-D end-effector states (xyz, roll-pitch-yaw, gripper). Following its
replay script, we feed the *recorded* states of the real episode, so the predicted wrist camera
should follow the real camera path.

DROID-100's `observation.state` already holds exactly these 7 values. DROID-100 episodes are most
likely in Ctrl-World's training set, so this is an in-distribution (best-case) test.
"""
import importlib.machinery
import json
import sys
import types
from pathlib import Path

import cv2
import einops
import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
REPO = ROOT / "third_party/Ctrl-World"
MODELS = ROOT / "models"
VIEWS = ("observation.images.exterior_image_1_left", "observation.images.exterior_image_2_left",
         "observation.images.wrist_image_left")  # Ctrl-World's stacking order
W, H = 320, 192
SKIP = 3            # 15 Hz -> 5 Hz, as in Ctrl-World's preprocessing
PRED = 5            # frames per interaction (first = current)
HISTORY_IDX = [0, 0, -8, -6, -4, -2]  # from rollout_replay_traj.py


def _stub_training_deps():
    # ctrl_world.py imports logging/video libraries it only uses for training
    for name in ("wandb", "swanlab", "decord"):
        if name not in sys.modules:
            mod = types.ModuleType(name)
            mod.__spec__ = importlib.machinery.ModuleSpec(name, None)  # accelerate calls find_spec
            mod.VideoReader = mod.cpu = None
            sys.modules[name] = mod
    sys.path.insert(0, str(REPO))
    # Ctrl-World's pipelines need diffusers 0.34 (0.40 removed text_to_video_synthesis), so this
    # process alone imports a pinned copy; the rest of the project keeps 0.40.
    sys.path.insert(0, str(ROOT / "third_party/pydeps_ctrlworld"))


class CtrlWorldRollout:
    def __init__(self, steps=50):
        _stub_training_deps()
        from config import wm_args
        from models.ctrl_world import CrtlWorld
        args = wm_args(task_type="replay")
        args.svd_model_path = str(MODELS / "stabilityai--stable-video-diffusion-img2vid")
        args.clip_model_path = str(MODELS / "openai--clip-vit-base-patch32")
        args.num_inference_steps = steps
        self.args = args
        self.model = CrtlWorld(args)
        state = torch.load(MODELS / "yjguo--Ctrl-World/checkpoint-10000.pt", map_location="cpu")
        self.model.load_state_dict(state)
        del state
        self.model.to("cuda").to(args.dtype).eval()
        stat = json.loads((REPO / "dataset_meta_info/droid/stat.json").read_text())
        self.p01 = np.array(stat["state_01"])[None]
        self.p99 = np.array(stat["state_99"])[None]

    def _norm(self, a):
        return np.clip(2 * (a - self.p01) / (self.p99 - self.p01 + 1e-8) - 1, -1, 1)

    @torch.no_grad()
    def _encode(self, frame):
        vae = self.model.pipeline.vae
        x = torch.from_numpy(frame).to("cuda", self.args.dtype).permute(2, 0, 1)[None] / 255.0 * 2 - 1
        return vae.encode(x).latent_dist.sample().mul_(vae.config.scaling_factor)[0]  # (4, 24, 40)

    @torch.no_grad()
    def _decode_wrist(self, latents):
        """latents (F, 4, 72, 40) stacked views -> wrist frames (F, H, W, 3) uint8."""
        vae = self.model.pipeline.vae
        wrist = latents[:, :, 48:72] / vae.config.scaling_factor
        out = vae.decode(wrist, num_frames=wrist.shape[0]).sample
        return ((out / 2 + 0.5).clamp(0, 1) * 255).float().cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)

    @torch.no_grad()
    def rollout(self, views0, states, text):
        """views0: 3 first frames (H, W, 3) in VIEWS order; states: (N, 7) recorded at 5 Hz.
        Returns wrist frames at 5 Hz, starting with the decoded first frame."""
        from models.pipeline_ctrl_world import CtrlWorldDiffusionPipeline
        a = self.args
        first = torch.cat([self._encode(v) for v in views0], dim=1)[None]  # (1, 4, 72, 40)
        his_cond = [first] * (a.num_history * 4)
        his_eef = [states[0:1]] * (a.num_history * 4)
        n_steps = (len(states) - 1) // (PRED - 1)
        frames = []
        for i in range(n_steps):
            s0 = i * (PRED - 1)
            pose = states[s0:s0 + PRED]
            action = np.concatenate([np.concatenate([his_eef[j] for j in HISTORY_IDX]), pose])
            action = torch.tensor(self._norm(action))[None].to("cuda", a.dtype)
            hidden = self.model.action_encoder(action, text, self.model.tokenizer, self.model.text_encoder)
            his = torch.cat([his_cond[j] for j in HISTORY_IDX], dim=0)[None]
            _, lat = CtrlWorldDiffusionPipeline.__call__(
                self.model.pipeline, image=his_cond[-1], text=hidden, width=a.width, height=a.height * 3,
                num_frames=a.num_frames, history=his, num_inference_steps=a.num_inference_steps,
                decode_chunk_size=a.decode_chunk_size, max_guidance_scale=a.guidance_scale, fps=a.fps,
                motion_bucket_id=a.motion_bucket_id, mask=None, output_type="latent", return_dict=False,
                frame_level_cond=True)
            lat = lat[0]  # (5, 4, 72, 40)
            wrist = self._decode_wrist(lat)
            frames.extend(wrist if i == 0 else wrist[1:])
            his_eef.append(pose[-1:])
            his_cond.append(lat[-1:])
        return frames


def episode_inputs(root, episode, start_frame, n_out):
    """3 first views (at start_frame, 15 Hz index) and n_out recorded 5 Hz states for one DROID-100 episode."""
    from .data_droid import decode
    root = Path(root)
    eps = pd.read_parquet(root / "meta/episodes/chunk-000/file-000.parquet").set_index("episode_index")
    e = eps.loc[episode]
    views = []
    for cam in VIEWS:
        video = root / f"videos/{cam}/chunk-000/file-000.mp4"
        t0 = float(e[f"videos/{cam}/from_timestamp"]) + start_frame / 15.0
        views.append(cv2.resize(decode(str(video), t0, t0 + 0.2)[0], (W, H), interpolation=cv2.INTER_LINEAR))
    data = pd.read_parquet(root / "data/chunk-000/file-000.parquet", columns=["episode_index", "frame_index", "observation.state"])
    st = np.stack(data[data.episode_index == episode].sort_values("frame_index")["observation.state"].to_numpy())
    idx = np.minimum(start_frame + SKIP * np.arange(n_out), len(st) - 1)
    return views, st[idx].astype(np.float64), str(e["tasks"][0])
