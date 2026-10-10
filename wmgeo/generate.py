"""Image(+text)-to-video generation with open models through diffusers.

Each model spec gives its native resolution, frame count and fps. Outputs are returned at native
fps; the runner resamples everything to EVAL_W x EVAL_H at EVAL_FPS before scoring.
All specs are sized to fit a 16 GB GPU with CPU offload and VAE tiling.
"""
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image

NEGATIVE = "blurry, distorted, low quality, watermark, text, cartoon"


def prompt_for(task):
    # Some DROID-100 episodes have no instruction; use one neutral sentence for those.
    action = task.rstrip(".").lower() if task.strip() else "pick up and move an object on the table"
    return f"Robot wrist camera view of a robot arm. The robot is trying to {action}. Realistic, steady footage."


@dataclass
class Spec:
    repo: str
    pipeline: str
    width: int
    height: int
    num_frames: int
    fps: int
    steps: int
    guidance: float
    extra: dict = field(default_factory=dict)
    fp8_weights: bool = False


MODELS = {
    # LTX borrows CogVideoX's bf16 T5-v1.1-XXL (same encoder, half the download of its fp32 copy).
    "ltx": Spec("Lightricks/LTX-Video", "LTXImageToVideoPipeline", 704, 480, 121, 24, 40, 3.0,
                {"text_encoder_from": "zai-org/CogVideoX-5b-I2V"}),
    "wan22_5b": Spec("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "WanImageToVideoPipeline", 1280, 704, 121, 24, 40, 5.0,
                     fp8_weights=True),
    "cosmos2_2b": Spec("nvidia/Cosmos-Predict2-2B-Video2World", "Cosmos2VideoToWorldPipeline", 1280, 704, 93, 16, 35, 7.0),
    "cogvideox_5b": Spec("zai-org/CogVideoX-5b-I2V", "CogVideoXImageToVideoPipeline", 720, 480, 49, 8, 50, 6.0,
                         fp8_weights=True),
}


MODELS_DIR = Path(__file__).resolve().parents[1] / "models"


def resolve(repo):
    """Prefer the curl-downloaded copy in models/<org>--<name>; fall back to the HF cache."""
    local = MODELS_DIR / repo.replace("/", "--")
    return str(local) if (local / "model_index.json").exists() or (local / "text_encoder").exists() else repo


class Generator:
    def __init__(self, name, offload=True):
        import diffusers
        self.name, self.spec = name, MODELS[name]
        cls = getattr(diffusers, self.spec.pipeline)
        extra = dict(self.spec.extra)
        if "text_encoder_from" in extra:
            from transformers import T5EncoderModel
            extra["text_encoder"] = T5EncoderModel.from_pretrained(
                resolve(extra.pop("text_encoder_from")), subfolder="text_encoder", torch_dtype=torch.bfloat16)
        self.pipe = cls.from_pretrained(resolve(self.spec.repo), torch_dtype=torch.bfloat16, **extra)
        if self.spec.fp8_weights:
            # Store transformer weights in fp8, compute in bf16: halves weight memory so the 5B
            # models fit in 16 GB without spilling into shared system memory.
            self.pipe.transformer.enable_layerwise_casting(
                storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
        if offload:
            self.pipe.enable_model_cpu_offload()
        else:
            self.pipe.to("cuda")
        vae = getattr(self.pipe, "vae", None)
        if vae is not None and hasattr(vae, "enable_tiling"):
            vae.enable_tiling()

    @torch.inference_mode()
    def __call__(self, first_frame, task, seed=0):
        s = self.spec
        image = Image.fromarray(first_frame).resize((s.width, s.height), Image.BICUBIC)
        kwargs = dict(image=image, prompt=prompt_for(task), negative_prompt=NEGATIVE,
                      num_frames=s.num_frames, num_inference_steps=s.steps, guidance_scale=s.guidance,
                      generator=torch.Generator("cpu").manual_seed(seed), output_type="np")
        if self.name != "cogvideox_5b":  # CogVideoX infers size from its config
            kwargs.update(width=s.width, height=s.height)
        out = self.pipe(**kwargs).frames[0]
        frames = [(np.clip(f, 0, 1) * 255).astype(np.uint8) for f in out]
        return frames, s.fps
