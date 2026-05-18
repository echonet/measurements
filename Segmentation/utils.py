import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import pydicom
import torch

_SEGM_DIR = Path(__file__).resolve().parent         
_ROOT     = _SEGM_DIR.parent.parent             

for _p in [str(_ROOT), str(_ROOT / "cvair")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

# Prevent local wandb/ stub from shadowing the real package
import types as _types
try:
    import wandb as _wandb_check
    if not hasattr(_wandb_check, "run"):
        raise AttributeError
except (ImportError, AttributeError):
    _wb = _types.ModuleType("wandb")
    _wb.run = None
    sys.modules.setdefault("wandb", _wb)

from cvair.training.model_wrappers import SegmentationModelWrapper  
from torchvision.models.segmentation import deeplabv3_resnet50      
from ultralytics import YOLO as _YOLOModel                           

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
CROP_SIZE = (256, 256)

# Area model resize resolution (must match training in e2e_utils.py)
AREA_RESIZE_W: int = 640
AREA_RESIZE_H: int = 480

VTI_OVERLAY_COLOR = {
    "av":   [1.0, 0.4, 0.1],
    "mv":   [0.2, 0.8, 0.9],
    "lvot": [0.9, 0.2, 0.8],
    "rvot": [0.9, 0.9, 0.1],
    "pv":   [0.3, 0.9, 0.3],
}

AREA_OVERLAY_COLOR = {
    "la_area": [0.2, 0.6, 1.0],   # blue
    "ra_area": [1.0, 0.5, 0.2],   # orange
}

def _ybr_to_rgb(frame: np.ndarray) -> np.ndarray:
    """Convert YBR_FULL / YBR_FULL_422 frame to RGB uint8."""
    try:
        return cv2.cvtColor(frame, cv2.COLOR_YCrCb2RGB)
    except cv2.error:
        return frame


def load_dicom_frame_bgr(dcm_path: Path) -> np.ndarray:
    ds  = pydicom.dcmread(str(dcm_path))
    pa  = ds.pixel_array

    pi = ds.get("PhotometricInterpretation", "")
    if isinstance(pi, bytes):
        pi = pi.decode("ascii", errors="ignore").strip()

    # Select first frame for multi-frame DICOMs
    if pa.ndim == 4:        # (F, H, W, C)
        frame = pa[0, :, :, :3].copy()
    elif pa.ndim == 3 and pa.shape[-1] in (3, 4):   # (H, W, C) single frame
        frame = pa[:, :, :3].copy()
    elif pa.ndim == 3:      # (F, H, W) grayscale multi-frame
        frame = pa[0].copy()
    else:                   # (H, W) grayscale still
        frame = pa.copy()

    # Normalise to uint8
    if frame.dtype != np.uint8:
        mx = frame.max()
        frame = (frame.astype(np.float32) / mx * 255).astype(np.uint8) if mx > 0 else frame.astype(np.uint8)

    # Colour space conversion
    if "YBR" in pi:
        frame = _ybr_to_rgb(frame)

    # Grayscale → BGR
    if frame.ndim == 2:
        return cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    return cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)



def load_seg_model(weights_path: str, device: torch.device):
    """
    Build DeepLabV3-ResNet50 and load checkpoint weights.

    The model output hook flattens DeepLab's dict output to the 'out' tensor
    so the wrapper receives a plain (B, 1, H, W) logit tensor.
    """
    def _hook(module, inp, output):
        return output["out"]

    backbone = deeplabv3_resnet50(num_classes=1)
    backbone.register_forward_hook(_hook)

    model = SegmentationModelWrapper(
        backbone, num_classes=1, lr=1e-3, tracked_metric="loss"
    )

    ckpt = torch.load(weights_path, map_location="cpu")
    sd   = ckpt.get("state_dict", ckpt)

    has_m  = any(k.startswith("m.") for k in model.state_dict())
    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module."):]
        elif ".module." in nk:
            parts = nk.split(".module.", 1)
            nk = parts[0] + "." + parts[1]
        if has_m and not nk.startswith("m."):
            nk = "m." + nk
        new_sd[nk] = v

    model.load_state_dict(new_sd, strict=False)
    model.eval().to(device)
    return model


def load_yolo_model(weights_path: str):
    """Load an Ultralytics YOLO model from a .pt file."""
    return _YOLOModel(str(weights_path))


@dataclass
class Mountain:
    """One YOLO-detected mountain (waveform peak) in Doppler coordinates."""
    idx:     int
    x_left:  int
    x_right: int
    width:   int
    conf:    float = 0.0


@dataclass
class CropPrediction:
    """Segmentation + QC result for one 256×256 mountain crop."""
    mountain:          Mountain
    crop_mask_256:     np.ndarray   # (256, 256) uint8 binary mask
    prob_map_256:      np.ndarray   # (256, 256) float32 sigmoid probs
    passed_qc:         bool
    qc_reject_reason:  str   = ""
    mask_area_px:      int   = 0
    mean_confidence:   float = 0.0
    n_contours:        int   = 0
    touches_top:       bool  = False
    mask_area_ratio:   float = 0.0



def detect_mountains_yolo(
    yolo_model,
    doppler_bgr: np.ndarray,
    conf:        float = 0.25,
    gpu:         int   = 0,
) -> List[Mountain]:
    """Detect mountain bounding boxes with YOLO, sorted left-to-right."""
    results   = yolo_model.predict(doppler_bgr, conf=conf, device=gpu, verbose=False)
    mountains = []
    if results and len(results[0].boxes):
        for box in results[0].boxes:
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
            c = float(box.conf[0].cpu())
            mountains.append(Mountain(
                idx=0, x_left=int(x1), x_right=int(x2),
                width=int(x2 - x1), conf=c,
            ))
    mountains.sort(key=lambda m: m.x_left)
    for i, m in enumerate(mountains):
        m.idx = i
    return mountains


def crop_mountain(
    doppler_bgr: np.ndarray,
    mountain:    Mountain,
    target_size: Tuple[int, int] = CROP_SIZE,
) -> np.ndarray:
    """Crop full-height mountain column and resize to target_size (W, H)."""
    crop = doppler_bgr[:, mountain.x_left:mountain.x_right]
    if crop.size == 0:
        return np.zeros((*target_size[::-1], 3), dtype=doppler_bgr.dtype)
    return cv2.resize(crop, target_size, interpolation=cv2.INTER_LINEAR)


@torch.no_grad()
def predict_crop(
    seg_model,
    crop_bgr: np.ndarray,
    device:   torch.device,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Run segmentation model on a 256×256 BGR crop.
    Returns (binary_mask uint8, prob_map float32).
    """
    rgb    = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)
    logits = seg_model.m(tensor)
    prob   = torch.sigmoid(logits).squeeze().cpu().numpy()
    mask   = (prob >= threshold).astype(np.uint8)
    return mask, prob


def qc_crop_prediction(
    mask:               np.ndarray,
    prob:               np.ndarray,
    baseline_y_in_crop: Optional[int],
    m_name:             str   = "MV",
    min_area_px:        int   = 200,
    max_area_ratio:     float = 0.60,
    min_confidence:     float = 0.65,
    max_contours:       int   = 4,
    top_edge_limit:     int   = 10,
    baseline_margin:    int   = 20,
) -> CropPrediction:
    """Quality-check one 256×256 crop prediction."""
    h, w       = mask.shape
    total_px   = h * w
    area       = int(mask.sum())
    area_ratio = area / total_px
    mean_conf  = float(prob[mask > 0].mean()) if area > 0 else 0.0

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    n_contours  = len(contours)

    touches_top = (int(mask[:top_edge_limit, :].sum()) > 5) if m_name == "MV" else False

    if baseline_y_in_crop is not None and 0 < baseline_y_in_crop < h:
        bl_zone = mask[
            max(0, baseline_y_in_crop - baseline_margin):baseline_y_in_crop + 5, :
        ]
        baseline_connected = int(bl_zone.sum()) > 10
    else:
        baseline_connected = True

    reject_reason = ""
    if area < min_area_px:
        reject_reason = f"area_too_small({area}<{min_area_px})"
    elif area_ratio > max_area_ratio:
        reject_reason = f"area_too_large({area_ratio:.2f}>{max_area_ratio})"
    elif mean_conf < min_confidence:
        reject_reason = f"low_confidence({mean_conf:.3f}<{min_confidence})"
    elif n_contours > max_contours:
        reject_reason = f"too_many_contours({n_contours}>{max_contours})"
    elif touches_top:
        reject_reason = "touches_top_edge"
    elif not baseline_connected:
        reject_reason = "not_baseline_connected"

    return CropPrediction(
        mountain          = None,    # caller sets this
        crop_mask_256     = mask,
        prob_map_256      = prob,
        passed_qc         = (reject_reason == ""),
        qc_reject_reason  = reject_reason,
        mask_area_px      = area,
        mean_confidence   = round(mean_conf, 4),
        n_contours        = n_contours,
        touches_top       = touches_top,
        mask_area_ratio   = round(area_ratio, 5),
    )


def reverse_transform_mask(
    crop_mask_256: np.ndarray,
    mountain:      Mountain,
    doppler_h:     int,
    doppler_w:     int,
) -> np.ndarray:
    """Resize a 256×256 crop mask back to full Doppler-region coordinates."""
    mtn_w = mountain.x_right - mountain.x_left
    if mtn_w <= 0:
        return np.zeros((doppler_h, doppler_w), dtype=np.uint8)
    restored  = cv2.resize(crop_mask_256, (mtn_w, doppler_h), interpolation=cv2.INTER_NEAREST)
    full_mask = np.zeros((doppler_h, doppler_w), dtype=np.uint8)
    x_start   = max(0, mountain.x_left)
    x_end     = min(doppler_w, mountain.x_right)
    full_mask[:, x_start:x_end] = restored[:, :x_end - x_start]
    return full_mask


def process_doppler_still(
    doppler_bgr:       np.ndarray,
    seg_model,
    yolo_model,
    device:            torch.device,
    model_key:         str,            # lowercase: "mv", "av", "lvot", "rvot", "pv"
    threshold:         float = 0.5,
    yolo_conf:         float = 0.35,
    baseline_ratio:    float = 0.5,
    gpu:               int   = 0,
    qc_min_area:       int   = 200,
    qc_max_area_ratio: float = 0.60,
    qc_min_confidence: float = 0.65,
    qc_max_contours:   int   = 4,
) -> dict:
    """
    Full YOLO → crop → seg → QC → combine pipeline for one Doppler still.

    Returns dict with keys:
      combined_mask    : (dh, dw) uint8  — OR of all accepted mountain masks
      best_mask        : (dh, dw) uint8  — highest-confidence single mountain
      n_mountains      : int
      n_passed         : int
      n_rejected       : int
      best_pred        : CropPrediction or None
      crop_predictions : List[CropPrediction]
    """
    dh, dw       = doppler_bgr.shape[:2]
    m_name_upper = model_key.upper()

    baseline_local     = int(dh * baseline_ratio) if baseline_ratio >= 0 else None
    baseline_y_in_crop = (
        int(baseline_local * 256.0 / dh)
        if (baseline_local is not None and dh > 0) else None
    )

    mountains        = detect_mountains_yolo(yolo_model, doppler_bgr, conf=0.25, gpu=gpu)
    crop_predictions: List[CropPrediction] = []
    combined_mask    = np.zeros((dh, dw), dtype=np.uint8)

    for mtn in mountains:
        if mtn.conf < yolo_conf:
            pred = CropPrediction(
                mountain=mtn,
                crop_mask_256=np.zeros((256, 256), dtype=np.uint8),
                prob_map_256=np.zeros((256, 256), dtype=np.float32),
                passed_qc=False,
                qc_reject_reason=f"low_yolo_conf({mtn.conf:.2f}<{yolo_conf})",
            )
            crop_predictions.append(pred)
            continue

        crop_bgr      = crop_mountain(doppler_bgr, mtn)
        mask_256, prob_256 = predict_crop(seg_model, crop_bgr, device, threshold)
        pred = qc_crop_prediction(
            mask_256, prob_256, baseline_y_in_crop,
            m_name=m_name_upper,
            min_area_px=qc_min_area, max_area_ratio=qc_max_area_ratio,
            min_confidence=qc_min_confidence, max_contours=qc_max_contours,
        )
        pred.mountain = mtn
        crop_predictions.append(pred)

    # Area-consistency QC: reject mountains whose area deviates >20% from best
    passed   = [p for p in crop_predictions if p.passed_qc]
    ref_area = (
        max(passed, key=lambda p: p.mean_confidence).mask_area_px if passed else None
    )

    for pred in crop_predictions:
        if not pred.passed_qc:
            continue
        if ref_area is not None:
            lo, hi = ref_area * 0.80, ref_area * 1.20
            if not (lo <= pred.mask_area_px <= hi):
                pred.passed_qc       = False
                pred.qc_reject_reason = (
                    f"area_inconsistent({pred.mask_area_px} vs ref {ref_area})"
                )
                continue
        mtn_mask      = reverse_transform_mask(pred.crop_mask_256, pred.mountain, dh, dw)
        combined_mask = np.maximum(combined_mask, mtn_mask)

    passed_final = [p for p in crop_predictions if p.passed_qc]
    best_pred    = (
        max(passed_final, key=lambda p: p.mean_confidence) if passed_final else None
    )

    best_mask = np.zeros((dh, dw), dtype=np.uint8)
    if best_pred is not None:
        best_mask = reverse_transform_mask(
            best_pred.crop_mask_256, best_pred.mountain, dh, dw
        )

    return {
        "combined_mask":    combined_mask,
        "best_mask":        best_mask,
        "n_mountains":      len(mountains),
        "n_passed":         len(passed_final),
        "n_rejected":       len(crop_predictions) - len(passed_final),
        "best_pred":        best_pred,
        "crop_predictions": crop_predictions,
    }


def save_overlay_vti(
    full_img_bgr:     np.ndarray,
    doppler_bgr:      np.ndarray,
    result:           dict,
    crop_y:           int,
    model_key:        str,
    output_path:      Path,
    file_uid:         str = "",
) -> None:

    combined_mask = result["combined_mask"]
    overlay_color = VTI_OVERLAY_COLOR.get(model_key.lower(), [1.0, 0.0, 0.0])
    doppler_rgb   = cv2.cvtColor(doppler_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(8, 4))
    ax.imshow(doppler_rgb)
    if combined_mask.sum() > 0:
        ov = np.zeros((*combined_mask.shape, 4), dtype=np.float32)
        ov[combined_mask > 0] = [*overlay_color, 0.45]
        ax.imshow(ov)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def process_area_still(
    frame_bgr: np.ndarray,
    seg_model,
    device:    torch.device,
    model_key: str,           # lowercase: "la_area", "ra_area"
    threshold: float = 0.5,
) -> dict:
    """
    Resize frame to 640×480 (Area training resolution), run DeepLabV3 segmentation,
    resize mask back to original resolution.

    Returns dict with keys:
      mask_full        : (H, W) uint8  — binary mask at original resolution
      prob_map_seg     : (480, 640) float32  — sigmoid probability map at seg resolution
      mask_seg         : (480, 640) uint8   — binary mask at seg resolution
      mask_area_px     : int   — foreground pixel count (original resolution)
      mean_confidence  : float — mean sigmoid prob over foreground pixels
    """
    ih, iw = frame_bgr.shape[:2]

    # Resize to training resolution and convert BGR→RGB, normalize to [0,1]
    frame_resized = cv2.resize(frame_bgr, (AREA_RESIZE_W, AREA_RESIZE_H),
                               interpolation=cv2.INTER_LINEAR)
    rgb = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).to(device)

    logits   = seg_model.m(tensor)
    prob_seg = torch.sigmoid(logits).squeeze().cpu().numpy().astype(np.float32)
    mask_seg = (prob_seg >= threshold).astype(np.uint8)

    # Resize mask back to original frame size
    mask_full = cv2.resize(mask_seg, (iw, ih), interpolation=cv2.INTER_NEAREST)

    area      = int(mask_full.sum())
    mean_conf = float(prob_seg[mask_seg > 0].mean()) if mask_seg.sum() > 0 else 0.0

    return {
        "mask_full":       mask_full,
        "prob_map_seg":    prob_seg,
        "mask_seg":        mask_seg,
        "mask_area_px":    area,
        "mean_confidence": round(mean_conf, 4),
    }


def save_overlay_area(
    frame_bgr:   np.ndarray,
    result:      dict,
    model_key:   str,
    output_path: Path,
    file_uid:    str = "",
) -> None:
    """
    Save the B-mode frame with the segmentation mask overlaid.
    """
    mask_full     = result["mask_full"]
    overlay_color = AREA_OVERLAY_COLOR.get(model_key.lower(), [1.0, 0.0, 0.0])
    frame_rgb     = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    ax.imshow(frame_rgb)
    if mask_full.sum() > 0:
        ov = np.zeros((*mask_full.shape, 4), dtype=np.float32)
        ov[mask_full > 0] = [*overlay_color, 0.45]
        ax.imshow(ov)
    title = f"{model_key.upper()}  |  {file_uid}" if file_uid else model_key.upper()
    ax.set_title(title, fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
