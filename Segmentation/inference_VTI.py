#!/usr/bin/env python3
"""
inference_VTI.py
================
Single-file VTI Doppler segmentation inference.
Supported model keys: MV, AV, LVOT, RVOT, PV

Weight files (place checkpoints here or use --seg_weights / --yolo_weights):
  weights/VTI/MV/segm_weights.ckpt
  weights/VTI/MV/yolo_weights.pt
  (same pattern for AV / LVOT / RVOT / PV)

Usage
-----
  # MV inference on a single DICOM (default)
  python inference_VTI.py

  # Explicit paths
  python inference_VTI.py \\
      --m_name MV \\
      --input_file data/Input_dicoms/MV_1.2.840....dcm \\
      --output_dir data/output/
"""

import argparse
import sys
from pathlib import Path

import torch

_SEGM_DIR = Path(__file__).resolve().parent   
_ROOT     = _SEGM_DIR.parent.parent           

if str(_SEGM_DIR) not in sys.path:
    sys.path.insert(0, str(_SEGM_DIR))

from utils import (   # noqa: E402
    load_dicom_frame_bgr,
    load_seg_model,
    load_yolo_model,
    process_doppler_still,
    save_overlay_vti,
)


_W = _SEGM_DIR / "weights" / "VTI"

DEFAULT_SEG_WEIGHTS = {
    "MV":          str(_W / "MV"            / "t04_xl6_260311_MV_Segm_scratch.ckpt"),
    "AV":          str(_W / "AV"            / "t04_4ve_260311_AV_Segm_scratch.ckpt"),
    "LVOT":        str(_W / "LVOT"          / "t04_ndx_260311_LVOT_Segm_scratch.ckpt"),
    "RVOT":        str(_W / "RVOT"          / "t04_z39_260311_RVOT_Segm_scratch.ckpt"),
    "PV":          str(_W / "PV"            / "t04_ypd_260311_PV_Segm_scratch.ckpt"),
    "All_below":   str(_W / "All_below_base" / "t04_sjw_260311_All_below_base_Segm_scratch.ckpt"),
}

DEFAULT_YOLO_WEIGHTS = {
    "MV":          str(_W / "MV"            / "t07_x5v_260311_MV_YOLO_yolo11n.pt"),
    "AV":          str(_W / "AV"            / "t07_6r9_260311_AV_YOLO_yolo11n.pt"),
    "LVOT":        str(_W / "LVOT"          / "t07_kt9_260312_LVOT_YOLO_yolo11n.pt"),
    "RVOT":        str(_W / "RVOT"          / "t07_dh5_260312_RVOT_YOLO_yolo11n.pt"),
    "PV":          str(_W / "PV"            / "t07_p0o_260312_PV_YOLO_yolo11n.pt"),
    "All_below":   str(_W / "All_below_base" / "t07_2ra_260312_All_below_base_YOLO_yolo11n.pt"),
}

# Default sample input
DEFAULT_INPUT = str(
    _SEGM_DIR / "data" / "Input_dicoms"
    / "XXXXXX.dcm" #Please add some dicoms here.
)

# Default output directory
DEFAULT_OUTPUT_DIR = str(_SEGM_DIR / "data" / "output")


# ─────────────────────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Single-file VTI Doppler segmentation inference"
    )
    p.add_argument("--m_name", type=str, default="MV",
                   choices=["MV", "AV", "LVOT", "RVOT", "PV", "All_below"],
                   help="VTI model key (default: MV)")
    p.add_argument("--input_file", type=str, default=DEFAULT_INPUT,
                   help="Path to input DICOM (.dcm)")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                   help="Output directory for overlay PNG")
    p.add_argument("--seg_weights", type=str, default=None,
                   help="Override segmentation checkpoint path")
    p.add_argument("--yolo_weights", type=str, default=None,
                   help="Override YOLO weights path")
    p.add_argument("--crop_y", type=int, default=342,
                   help="Y pixel where Doppler region starts (default: 342)")
    p.add_argument("--baseline_ratio", type=float, default=0.5,
                   help="Estimated baseline as fraction of Doppler height (default: 0.5). "
                        "Set -1 to disable baseline QC.")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Segmentation sigmoid threshold (default: 0.5)")
    p.add_argument("--yolo_conf", type=float, default=0.35,
                   help="YOLO detection confidence threshold (default: 0.35)")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU index (default: 0; CPU used if CUDA unavailable)")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    seg_weights  = args.seg_weights  or DEFAULT_SEG_WEIGHTS[args.m_name]
    yolo_weights = args.yolo_weights or DEFAULT_YOLO_WEIGHTS[args.m_name]

    seg_weights_path  = Path(seg_weights)
    yolo_weights_path = Path(yolo_weights)
    input_path        = Path(args.input_file)
    output_dir        = Path(args.output_dir)

    if not input_path.exists():
        raise FileNotFoundError(f"Input DICOM not found: {input_path}")
    if not seg_weights_path.exists():
        raise FileNotFoundError(
            f"Segmentation weights not found: {seg_weights_path}\n"
            f"Place the checkpoint at that path or pass --seg_weights."
        )
    if not yolo_weights_path.exists():
        raise FileNotFoundError(
            f"YOLO weights not found: {yolo_weights_path}\n"
            f"Place the .pt file at that path or pass --yolo_weights."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[inference_VTI] device : {device}")
    print(f"[inference_VTI] m_name : {args.m_name}")
    print(f"[inference_VTI] input  : {input_path}")

    print("[inference_VTI] Loading segmentation model ...")
    seg_model  = load_seg_model(str(seg_weights_path), device)

    print("[inference_VTI] Loading YOLO mountain detector ...")
    yolo_model = load_yolo_model(str(yolo_weights_path))

    print("[inference_VTI] Reading DICOM ...")
    full_bgr = load_dicom_frame_bgr(input_path)
    ih, iw   = full_bgr.shape[:2]
    print(f"[inference_VTI] Frame size: {iw}×{ih}")

    # ── Crop Doppler region ───────────────────────────────────────────────────
    crop_y      = min(args.crop_y, ih - 1)
    doppler_bgr = full_bgr[crop_y:, :]
    print(f"[inference_VTI] Doppler region: y>={crop_y}  size={doppler_bgr.shape[1]}×{doppler_bgr.shape[0]}")

    # ── Run pipeline ─────────────────────────────────────────────────────────
    result = process_doppler_still(
        doppler_bgr       = doppler_bgr,
        seg_model         = seg_model,
        yolo_model        = yolo_model,
        device            = device,
        model_key         = args.m_name.lower(),
        threshold         = args.threshold,
        yolo_conf         = args.yolo_conf,
        baseline_ratio    = args.baseline_ratio,
        gpu               = args.gpu,
    )

    # ── Save overlay ─────────────────────────────────────────────────────────
    file_uid     = input_path.stem
    overlay_dir  = output_dir / args.m_name
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = overlay_dir / f"{file_uid}_{args.m_name}_overlay.png"
    save_overlay_vti(
        full_img_bgr  = full_bgr,
        doppler_bgr   = doppler_bgr,
        result        = result,
        crop_y        = crop_y,
        model_key     = args.m_name.lower(),
        output_path   = overlay_path,
        file_uid      = file_uid,
    )  # full_img_bgr is accepted but unused (signature kept for API compatibility)
    print(f"[inference_VTI] Overlay saved → {overlay_path}")


if __name__ == "__main__":
    main()
