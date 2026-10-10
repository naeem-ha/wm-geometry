"""Incremental SfM with COLMAP (pycolmap 4.x) on one clip, with a shared camera and optional mask."""
import shutil
import tempfile
from pathlib import Path

import cv2
import numpy as np
import pycolmap

pycolmap.logging.minloglevel = 2  # errors only


def _masks(frames_dir, mask_png, out_dir):
    """COLMAP wants one mask per image, named <image>.png; 0 = ignore."""
    out_dir.mkdir(parents=True, exist_ok=True)
    m = cv2.imread(str(mask_png), 0)
    for f in sorted(frames_dir.glob("*.png")):
        cv2.imwrite(str(out_dir / f"{f.name}.png"), m)
    return out_dir


def run(clip_dir, mask_png=None, overlap=10, seed=0):
    clip_dir = Path(clip_dir)
    frames_dir = clip_dir / "frames"
    n_frames = len(list(frames_dir.glob("*.png")))
    work = Path(tempfile.mkdtemp(prefix="colmap_"))
    try:
        reader = pycolmap.ImageReaderOptions()
        reader.camera_model = "SIMPLE_RADIAL"
        if mask_png is not None and Path(mask_png).exists():
            reader.mask_path = str(_masks(frames_dir, mask_png, work / "masks"))
        extract = pycolmap.FeatureExtractionOptions()
        # Frames are only 320x180: upsample one octave and lower the peak threshold for more features.
        extract.sift.max_num_features = 4096
        extract.sift.first_octave = -1
        extract.sift.peak_threshold = 0.004
        db = work / "db.db"
        pycolmap.extract_features(db, frames_dir, camera_mode=pycolmap.CameraMode.SINGLE,
                                  reader_options=reader, extraction_options=extract)
        pairing = pycolmap.SequentialPairingOptions()
        pairing.overlap = overlap
        pairing.loop_detection = False
        verify = pycolmap.TwoViewGeometryOptions()
        verify.min_num_inliers = 12
        pycolmap.match_sequential(db, pairing_options=pairing, verification_options=verify)
        opts = pycolmap.IncrementalPipelineOptions()
        opts.random_seed = seed
        opts.multiple_models = True
        opts.min_num_matches = 12
        # COLMAP defaults (100 init inliers, 30 per registration) assume megapixel images.
        opts.mapper.init_min_num_inliers = 40
        opts.mapper.abs_pose_min_num_inliers = 15
        opts.mapper.init_min_tri_angle = 4.0
        recs = pycolmap.incremental_mapping(db, frames_dir, work / "sparse", options=opts)
        return summarize(recs, n_frames)
    finally:
        shutil.rmtree(work, ignore_errors=True)


def summarize(recs, n_frames):
    if not recs:
        return {"n_frames": n_frames, "n_models": 0, "reg_rate": 0.0, "reg_rate_any": 0.0, "reproj_err": np.nan,
                "track_len": np.nan, "n_points3D": 0, "centers": {}}
    rec = max(recs.values(), key=lambda r: r.num_reg_images())
    centers = {img.name: img.projection_center().tolist() for img in rec.images.values() if img.has_pose}
    return {
        "n_frames": n_frames,
        "n_models": len(recs),               # >1 means the clip broke into disconnected pieces
        "reg_rate": rec.num_reg_images() / n_frames,                       # largest model
        # models can share images, so count unique names rather than summing
        "reg_rate_any": len({im.name for r in recs.values() for im in r.images.values() if im.has_pose}) / n_frames,
        "reproj_err": float(rec.compute_mean_reprojection_error()),
        "track_len": float(rec.compute_mean_track_length()),
        "n_points3D": int(rec.num_points3D()),
        "centers": centers,
    }
