"""Feed-forward reconstruction with VGGT, plus a multi-view depth-consistency score.

VGGT predicts per-frame depth and cameras in one pass. If a clip shows one rigid scene, frame i's
depth unprojected with its camera and reprojected into frame j should agree with frame j's depth.
We report the median relative disagreement over frame pairs at fixed gaps.
"""
from pathlib import Path

import numpy as np
import torch

_model = None
# A fixed 5 fps subsample (every 3rd frame at 15 fps) keeps frame gaps comparable across horizons
# and stays at <= 26 frames (about 7 GB peak) for the 5 s horizon.
FRAME_STEP = 3


def _load():
    global _model
    if _model is None:
        from vggt.models.vggt import VGGT
        local = Path(__file__).resolve().parents[1] / "models/facebook--VGGT-1B"
        src = str(local) if (local / "model.safetensors").exists() else "facebook/VGGT-1B"
        _model = VGGT.from_pretrained(src).to("cuda").eval()
    return _model


@torch.inference_mode()
def predict(frame_paths):
    from vggt.utils.load_fn import load_and_preprocess_images
    from vggt.utils.pose_enc import pose_encoding_to_extri_intri
    images = load_and_preprocess_images([str(p) for p in frame_paths]).to("cuda")
    with torch.autocast("cuda", dtype=torch.bfloat16):
        pred = _load()(images)
    extr, intr = pose_encoding_to_extri_intri(pred["pose_enc"], images.shape[-2:])
    return {
        "extrinsic": extr[0].float().cpu().numpy(),         # (S,3,4) cam_from_world
        "intrinsic": intr[0].float().cpu().numpy(),         # (S,3,3)
        "depth": pred["depth"][0, ..., 0].float().cpu().numpy(),  # (S,H,W)
        "depth_conf": pred["depth_conf"][0].float().cpu().numpy(),
    }


def _reproject_err(p, i, j, conf_q=0.5):
    d_i, d_j = p["depth"][i], p["depth"][j]
    H, W = d_i.shape
    conf = p["depth_conf"][i]
    keep = conf > np.quantile(conf, conf_q)
    v, u = np.nonzero(keep)
    z = d_i[v, u]
    Ki, Kj = p["intrinsic"][i], p["intrinsic"][j]
    pts_cam = np.linalg.inv(Ki) @ np.stack([u * z, v * z, z])
    Ri, ti = p["extrinsic"][i][:, :3], p["extrinsic"][i][:, 3:]
    Rj, tj = p["extrinsic"][j][:, :3], p["extrinsic"][j][:, 3:]
    pts_w = Ri.T @ (pts_cam - ti)
    pj = Rj @ pts_w + tj
    zj = pj[2]
    uv = (Kj @ pj)[:2] / np.maximum(zj, 1e-6)
    uj, vj = np.round(uv).astype(int)
    ok = (zj > 1e-6) & (uj >= 0) & (uj < W) & (vj >= 0) & (vj < H)
    if ok.sum() < 100:
        return np.nan, float(ok.mean())
    rel = np.abs(zj[ok] - d_j[vj[ok], uj[ok]]) / np.maximum(d_j[vj[ok], uj[ok]], 1e-6)
    return float(np.median(rel)), float(ok.mean())


def clip_stats(clip_dir, gaps=(3, 6, 12)):
    """gaps are in original 15 fps frames (0.2, 0.4, 0.8 s)."""
    paths = sorted((Path(clip_dir) / "frames").glob("*.png"))[::FRAME_STEP]
    p = predict(paths)
    S = len(paths)
    out = {"n_frames": S, "frame_step": FRAME_STEP}
    for g in gaps:
        gs = g // FRAME_STEP
        pairs = [_reproject_err(p, i, i + gs) for i in range(S - gs)]
        errs, vis = zip(*pairs) if pairs else ((np.nan,), (np.nan,))
        out[g] = {"depth_reproj_rel_med": float(np.nanmedian(errs)), "covis_frac": float(np.nanmean(vis))}
    centers = np.stack([-e[:, :3].T @ e[:, 3] for e in p["extrinsic"]])
    out["path_length"] = float(np.linalg.norm(np.diff(centers, axis=0), axis=1).sum())
    return out, p


def centers(extrinsic):
    return np.stack([-e[:, :3].T @ e[:, 3] for e in extrinsic])


def ate_sim3(est, ref):
    """Trajectory error of est vs ref camera centers (N,3) after a Umeyama Sim(3) fit,
    as RMSE divided by ref path length (scale-free; both come from VGGT's arbitrary frame)."""
    mu_e, mu_r = est.mean(0), ref.mean(0)
    E, Rf = est - mu_e, ref - mu_r
    U, D, Vt = np.linalg.svd(Rf.T @ E / len(est))
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1
    R = U @ S @ Vt
    var_e = (E ** 2).sum() / len(est)
    scale = np.trace(np.diag(D) @ S) / max(var_e, 1e-12)
    aligned = scale * (R @ E.T).T + mu_r
    rmse = np.sqrt(((aligned - ref) ** 2).sum(1).mean())
    path = np.linalg.norm(np.diff(ref, axis=0), axis=1).sum()
    return float(rmse / max(path, 1e-9))
