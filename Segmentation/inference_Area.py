#!/usr/bin/env python3
"""
inference_Area.py
=================
Single-file B-mode area segmentation inference (LA_AREA / RA_AREA).

Usage
-----
  # LA_AREA inference on a single DICOM (default)
  python inference_Area.py

  # Explicit paths
  python inference_Area.py \\
      --m_name LA_AREA \\
      --input_file data/Input_dicoms/LA_....dcm \\
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
    process_area_still,
    save_overlay_area,
)

_W = _SEGM_DIR / "weights" / "Area"

DEFAULT_SEG_WEIGHTS = {
    "LA_AREA": str(_W / "LA_AREA" / "t_a_05_74l_260319_LA_AREA_scratch.ckpt"),
    "RA_AREA": str(_W / "RA_AREA" / "t_a_05_jnj_260319_RA_AREA_scratch.ckpt"),
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
        description="Single-file B-mode area segmentation inference (LA_AREA / RA_AREA)"
    )
    p.add_argument("--m_name", type=str, default="LA_AREA",
                   choices=["LA_AREA", "RA_AREA"],
                   help="Area model key (default: LA_AREA)")
    p.add_argument("--input_file", type=str, default=DEFAULT_INPUT,
                   help="Path to input DICOM (.dcm)")
    p.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR,
                   help="Output directory for overlay PNG")
    p.add_argument("--seg_weights", type=str, default=None,
                   help="Override segmentation checkpoint path")
    p.add_argument("--threshold", type=float, default=0.5,
                   help="Segmentation sigmoid threshold (default: 0.5)")
    p.add_argument("--gpu", type=int, default=0,
                   help="GPU index (default: 0; CPU used if CUDA unavailable)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    args = parse_args()

    # ── Resolve weight path ──────────────────────────────────────────────────
    seg_weights = args.seg_weights or DEFAULT_SEG_WEIGHTS[args.m_name]

    seg_weights_path = Path(seg_weights)
    input_path       = Path(args.input_file)
    output_dir       = Path(args.output_dir)

    # ── Validation ───────────────────────────────────────────────────────────
    if not input_path.exists():
        raise FileNotFoundError(f"Input DICOM not found: {input_path}")
    if not seg_weights_path.exists():
        raise FileNotFoundError(
            f"Segmentation weights not found: {seg_weights_path}\n"
            f"Place the checkpoint at that path or pass --seg_weights."
        )

    output_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ───────────────────────────────────────────────────────────────
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"[inference_Area] device : {device}")
    print(f"[inference_Area] m_name : {args.m_name}")
    print(f"[inference_Area] input  : {input_path}")

    # ── Load model ───────────────────────────────────────────────────────────
    print("[inference_Area] Loading segmentation model ...")
    seg_model = load_seg_model(str(seg_weights_path), device)

    # ── Load DICOM frame ─────────────────────────────────────────────────────
    print("[inference_Area] Reading DICOM ...")
    frame_bgr = load_dicom_frame_bgr(input_path)
    ih, iw    = frame_bgr.shape[:2]
    print(f"[inference_Area] Frame size: {iw}×{ih}")

    # ── Run segmentation ─────────────────────────────────────────────────────
    result = process_area_still(
        frame_bgr = frame_bgr,
        seg_model = seg_model,
        device    = device,
        model_key = args.m_name.lower(),
        threshold = args.threshold,
    )
    print(f"[inference_Area] mask_area_px   : {result['mask_area_px']}")
    print(f"[inference_Area] mean_confidence: {result['mean_confidence']:.4f}")

    # ── Save overlay ─────────────────────────────────────────────────────────
    file_uid     = input_path.stem
    overlay_dir  = output_dir / args.m_name
    overlay_dir.mkdir(parents=True, exist_ok=True)
    overlay_path = overlay_dir / f"{file_uid}_{args.m_name}_overlay.png"

    save_overlay_area(
        frame_bgr   = frame_bgr,
        result      = result,
        model_key   = args.m_name.lower(),
        output_path = overlay_path,
        file_uid    = file_uid,
    )
    print(f"[inference_Area] Overlay saved → {overlay_path}")


if __name__ == "__main__":
    main()
