from pathlib import Path
import os
import torch
import pandas as pd
import numpy as np
from argparse import ArgumentParser
from torchvision.models.segmentation import deeplabv3_resnet50
import cv2
import pydicom
from pydicom.pixel_data_handlers.util import convert_color_space
from tqdm import tqdm
from utils import segmentation_to_coordinates, process_video_with_diameter, get_coordinates_from_dicom, ybr_to_rgb

"""
Purpose:
    Unified 2D B-mode (frame-to-frame) segmentation inference script.
    Supports three modes selected automatically from the arguments provided:

      - single   : one video file  ->  annotated AVI/MP4 + distance overlay + CSV
      - folders  : folder of files ->  per-file AVI + distance overlay + metadata CSV
      - manifest : folder + frame manifest CSV -> per-frame JPG + metadata CSV

    Disclaimer: Please use the appropriate echocardiographic view for each measurement
    (e.g., LVID in standard PLAX view, not A4C or zoomed-out PLAX).
    The model expects videos with a 3:4 aspect ratio (height 480, width 640).

Input  (mode = single):
    --model_weights   : ivs | lvid | lvpw | aorta | aortic_root | la | rv_base | pa | ivc
    --file_path       : path to .avi or .dcm video
    --output_path     : output file (.avi or .mp4)
    [--phase_estimate]: (LVID only) annotate systole/diastole phase

Input  (mode = folders):
    --model_weights       : same choices as above
    --folders             : path to a folder of .avi or .dcm files (uniform extension)
    --output_path_folders : output folder

Input  (mode = manifest):
    --model_weights       : same choices as above
    --folders             : path to a folder of .avi or .dcm files
    --manifest_with_frame : CSV with columns [video_path, frame_number]
    --output_path_folders : output folder

Output (single):
    - Annotated AVI/MP4 video
    - *_distance.avi/mp4 with diameter overlay  (DICOM input only)
    - CSV with per-frame coordinates

Output (folders):
    - Per-file annotated AVI + *_distance.avi
    - metadata_<model_weights>.csv

Output (manifest):
    - Per-file annotated JPG for the specified frame
    - metadata_<model_weights>.csv

======================================================================================================
Sample Usage:
    # Single file (IVS)
    python inference_2D_image.py \\
        --model_weights ivs \\
        --file_path     ./data/SAMPLE_DICOM/IVS_FOLDERS/IVS_SAMPLE_0.dcm \\
        --output_path   ./data/OUTPUT/AVI/IVS_SAMPLE_GENERATED.avi

    # Single file (LVID) with phase estimation
    python inference_2D_image.py \\
        --model_weights lvid \\
        --file_path     ./data/SAMPLE_DICOM/IVS_FOLDERS/LVID_SAMPLE_0.dcm \\
        --output_path   ./data/OUTPUT/AVI/LVID_SAMPLE_GENERATED.avi \\
        --phase_estimate

    # Other single-file examples
    # python inference_2D_image.py --model_weights lvpw  --file_path ./data/SAMPLE_DICOM/LVPW_SAMPLE_0.dcm  --output_path ./data/OUTPUT/AVI/LVPW_SAMPLE_GENERATED.avi
    # python inference_2D_image.py --model_weights aorta --file_path ./data/SAMPLE_DICOM/AORTA_SAMPLE_0.dcm --output_path ./data/OUTPUT/AVI/AORTA_SAMPLE_GENERATED.avi
    # python inference_2D_image.py --model_weights aortic_root --file_path ./data/SAMPLE_DICOM/AORTIC_ROOT_SAMPLE_0.dcm --output_path ./data/OUTPUT/AVI/AORTIC_ROOT_SAMPLE_GENERATED.avi
    # python inference_2D_image.py --model_weights la    --file_path ./data/SAMPLE_DICOM/LA_SAMPLE_0.dcm    --output_path ./data/OUTPUT/AVI/LA_SAMPLE_GENERATED.avi
    # python inference_2D_image.py --model_weights rv_base --file_path ./data/SAMPLE_DICOM/RV_BASE_SAMPLE_0.dcm --output_path ./data/OUTPUT/AVI/RV_BASE_SAMPLE_GENERATED.avi
    # python inference_2D_image.py --model_weights pa    --file_path ./data/SAMPLE_DICOM/PA_SAMPLE_0.dcm    --output_path ./data/OUTPUT/AVI/PA_SAMPLE_GENERATED.avi
    # python inference_2D_image.py --model_weights ivc   --file_path ./data/SAMPLE_DICOM/IVC_SAMPLE_0.dcm   --output_path ./data/OUTPUT/AVI/IVC_SAMPLE_GENERATED.avi

    # Folder batch (IVS)
    python inference_2D_image.py \\
        --model_weights       ivs \\
        --folders             ./data/SAMPLE_DICOM/IVS_FOLDERS \\
        --output_path_folders ./data/OUTPUT/IVS

    # Manifest (specific frames)
    python inference_2D_image.py \\
        --model_weights       ivs \\
        --folders             ./data/SAMPLE_DICOM/IVS_FOLDERS \\
        --manifest_with_frame ./my_manifest.csv \\
        --output_path_folders ./data/OUTPUT/IVS
"""

# ─────────────────────────── ARGUMENT PARSING ────────────────────────────────
parser = ArgumentParser(description="2D B-mode segmentation inference (single / folders / manifest)")
parser.add_argument("--model_weights", type=str, required=True, choices=[
    "ivs", "lvid", "lvpw", "aorta", "aortic_root", "la", "rv_base", "pa", "ivc",
])
# single mode
parser.add_argument("--file_path",       type=str, default=None, help="[single] Path to input .avi or .dcm")
parser.add_argument("--output_path",     type=str, default=None, help="[single] Output .avi or .mp4 path")
parser.add_argument("--phase_estimate",  action="store_true",    help="[single, LVID only] Annotate systole/diastole phase")
# folders / manifest mode
parser.add_argument("--folders",             type=str, default=None, help="[folders/manifest] Folder of .avi or .dcm files")
parser.add_argument("--output_path_folders", type=str, default=None, help="[folders/manifest] Output folder")
parser.add_argument("--manifest_with_frame", type=str, default=None, help="[manifest] CSV with columns: video_path, frame_number")
args = parser.parse_args()

# ─────────────────────────── MODE DETECTION ──────────────────────────────────
if args.file_path is not None:
    MODE = "single"
elif args.folders is not None and args.manifest_with_frame is not None:
    MODE = "manifest"
elif args.folders is not None:
    MODE = "folders"
else:
    parser.error("Specify --file_path (single mode) or --folders (folders / manifest mode).")

print("=" * 60)
print(f"  2D B-mode Inference  |  mode={MODE}  |  weights={args.model_weights}")
print("=" * 60)

# ─────────────────────────── CONFIGURATION ───────────────────────────────────
SEGMENTATION_THRESHOLD = 0.0
DO_SIGMOID             = True

# DICOM tags
REGION_PHYSICAL_DELTA_X_SUBTAG    = (0x0018, 0x602C)
REGION_PHYSICAL_DELTA_Y_SUBTAG    = (0x0018, 0x602E)
PHOTOMETRIC_INTERPRETATION_TAG    = (0x0028, 0x0004)
ULTRASOUND_COLOR_DATA_PRESENT_TAG = (0x0028, 0x0014)

# ─────────────────────────── MODEL LOADING ───────────────────────────────────
device       = "cuda:0"
weights_path = f"./weights/2D_models/{args.model_weights}_weights.ckpt"

print(f"[Model] Loading weights: {weights_path}")
weights  = torch.load(weights_path, map_location=device)
backbone = deeplabv3_resnet50(num_classes=2)
weights  = {k.replace("m.", ""): v for k, v in weights.items()}
print(f"[Model] {backbone.load_state_dict(weights)}")
backbone = backbone.to(device)
backbone.eval()
print(f"[Model] Ready on {device}")
print("-" * 60)

# ─────────────────────────── SHARED HELPERS ──────────────────────────────────

def forward_pass(inputs):
    """Run one forward pass; returns (B, 2, 2) coordinate tensor."""
    logits = backbone(inputs)["out"]
    if DO_SIGMOID:
        logits = torch.sigmoid(logits)
    if SEGMENTATION_THRESHOLD is not None:
        logits[logits < SEGMENTATION_THRESHOLD] = 0.0
    return segmentation_to_coordinates(logits, normalize=False, order="XY")


def load_video_frames(video_file):
    """
    Load all frames from a .avi or .dcm file.

    Returns:
        frames : list of HxWxC uint8 numpy arrays (resized to 480x640 for DICOM)
        meta   : dict with conversion_factor_X/Y, ratio_height/width,
                 PhotometricInterpretation, ultrasound_color_data_present
    """
    frames = []
    meta = {
        "PhotometricInterpretation":     None,
        "ultrasound_color_data_present": None,
        "conversion_factor_X": None,
        "conversion_factor_Y": None,
        "ratio_height": 1.0,
        "ratio_width":  1.0,
    }

    if video_file.endswith(".avi"):
        cap = cv2.VideoCapture(video_file)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

    elif video_file.endswith(".dcm"):
        ds = pydicom.dcmread(video_file)

        meta["ultrasound_color_data_present"] = (
            ds[ULTRASOUND_COLOR_DATA_PRESENT_TAG].value
            if ULTRASOUND_COLOR_DATA_PRESENT_TAG in ds else np.nan
        )
        meta["PhotometricInterpretation"] = (
            ds[PHOTOMETRIC_INTERPRETATION_TAG].value
            if PHOTOMETRIC_INTERPRETATION_TAG in ds else None
        )

        pixel_array = ds.pixel_array  # (Frames, H, W, C)
        height, width = pixel_array.shape[1], pixel_array.shape[2]
        meta["ratio_height"] = height / 480
        meta["ratio_width"]  = width  / 640

        if meta["ratio_height"] != meta["ratio_width"]:
            raise ValueError(
                f"Aspect ratio mismatch ({height}x{width}). "
                "Model expects 3:4 aspect ratio (480x640)."
            )

        regions = get_coordinates_from_dicom(ds)
        doppler_region = regions[0] if regions else None
        if doppler_region is not None:
            if REGION_PHYSICAL_DELTA_X_SUBTAG in doppler_region:
                meta["conversion_factor_X"] = abs(doppler_region[REGION_PHYSICAL_DELTA_X_SUBTAG].value)
            if REGION_PHYSICAL_DELTA_Y_SUBTAG in doppler_region:
                meta["conversion_factor_Y"] = abs(doppler_region[REGION_PHYSICAL_DELTA_Y_SUBTAG].value)

        for frame in pixel_array:
            if ds.PhotometricInterpretation == "YBR_FULL_422":
                frame = ybr_to_rgb(frame)
            frames.append(cv2.resize(frame, (640, 480)))

    else:
        raise ValueError(f"Unsupported file type: {video_file}. Must be .avi or .dcm")

    return frames, meta


def check_extension_uniformity(folder_path, allowed=(".dcm", ".avi")):
    """Check that all files in the folder share a single allowed extension."""
    exts = set()
    for fname in os.listdir(folder_path):
        ext = os.path.splitext(fname)[1].lower()
        if ext in allowed:
            exts.add(ext)
        elif ext:
            print(f"  [Warn] Unexpected file: {fname}")
            return False
    if len(exts) == 1:
        print(f"  [Check] Extension uniformity OK  ({exts.pop()})")
        return True
    print(f"  [Warn] Mixed or unrecognised extensions found: {exts}")
    return False


def make_annotated_frame(frame_tensor, prediction, dot_radius=5, color=(235, 206, 135)):
    """
    Draw two landmark points and a connecting line onto a frame tensor.

    Args:
        frame_tensor : (C, H, W) float tensor in [0, 1]
        prediction   : (2, 2) array  [[x1, y1], [x2, y2]]

    Returns:
        annotated (H, W, C) uint8 image, x1, y1, x2, y2 ints
    """
    frame = frame_tensor.permute(1, 2, 0).cpu().numpy()
    frame = (np.clip(frame, 0, 1) * 255).astype(np.uint8)
    x1, y1 = int(prediction[0][0]), int(prediction[0][1])
    x2, y2 = int(prediction[1][0]), int(prediction[1][1])
    cv2.circle(frame, (x1, y1), dot_radius, color, -1)
    cv2.circle(frame, (x2, y2), dot_radius, color, -1)
    cv2.line(frame, (x1, y1), (x2, y2), color, 2)
    return frame, x1, y1, x2, y2


def compute_diameter(x1, y1, x2, y2, ratio, conv_x, conv_y):
    """Return physical diameter (cm), or None if conversion factors are missing."""
    if conv_x is None or conv_y is None:
        return None
    delta_x = abs(x2 - x1) * ratio
    delta_y = abs(y2 - y1) * ratio
    return float(np.sqrt((delta_x * conv_x) ** 2 + (delta_y * conv_y) ** 2))


# ======================================================================
#  MODE: single
# ======================================================================
if MODE == "single":
    VIDEO_FILE  = args.file_path
    OUTPUT_FILE = args.output_path

    if   VIDEO_FILE.endswith(".avi"):  input_type  = "avi"
    elif VIDEO_FILE.endswith(".dcm"):  input_type  = "dcm"
    else: raise ValueError("--file_path must be .avi or .dcm")

    if   OUTPUT_FILE.endswith(".avi"): output_type = "avi"
    elif OUTPUT_FILE.endswith(".mp4"): output_type = "mp4"
    else: raise ValueError("--output_path must be .avi or .mp4")

    print(f"[single] Input : {VIDEO_FILE}")
    print(f"[single] Output: {OUTPUT_FILE}")

    # ── Load frames ────────────────────────────────────────────────────────
    frames, meta = load_video_frames(VIDEO_FILE)
    print(f"[single] Loaded {len(frames)} frames from {input_type.upper()}")
    if meta["conversion_factor_X"]:
        print(f"[single] Physical delta  X={meta['conversion_factor_X']:.6f}  "
              f"Y={meta['conversion_factor_Y']:.6f}")

    # ── Inference ──────────────────────────────────────────────────────────
    input_tensor = torch.tensor(np.array(frames)).float() / 255.0
    input_tensor = input_tensor.to(device).permute(0, 3, 1, 2)  # (F, C, H, W)
    print(f"[single] Running inference on {input_tensor.shape[0]} frames ...")

    predictions = []
    for i in range(input_tensor.shape[0]):
        with torch.no_grad():
            predictions.append(forward_pass(input_tensor[i].unsqueeze(0)))
    predictions = torch.cat(predictions, dim=0).cpu().numpy()  # (F, 2, 2)
    print(f"[single] Inference complete")

    # ── Write annotated video ──────────────────────────────────────────────
    os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_FILE)), exist_ok=True)
    fourcc    = cv2.VideoWriter_fourcc(*("XVID" if output_type == "avi" else "mp4v"))
    h, w      = input_tensor.shape[-2], input_tensor.shape[-1]
    out_video = cv2.VideoWriter(OUTPUT_FILE, fourcc, 30, (w, h))
    if not out_video.isOpened():
        raise ValueError("VideoWriter failed to open.")

    frame_nums, x1s, y1s, x2s, y2s = [], [], [], [], []
    for i, (frame_t, pred) in enumerate(zip(input_tensor, predictions)):
        annotated, x1, y1, x2, y2 = make_annotated_frame(
            frame_t, pred, dot_radius=3, color=(135, 206, 235)
        )
        out_video.write(annotated)
        frame_nums.append(i); x1s.append(x1); y1s.append(y1)
        x2s.append(x2);       y2s.append(y2)
    out_video.release()
    print(f"[single] Saved annotated video    -> {OUTPUT_FILE}")

    # ── CSV ────────────────────────────────────────────────────────────────
    df       = pd.DataFrame({"frame_number": frame_nums,
                              "pred_x1": x1s, "pred_y1": y1s,
                              "pred_x2": x2s, "pred_y2": y2s})
    csv_path = OUTPUT_FILE.rsplit(".", 1)[0] + ".csv"
    df.to_csv(csv_path, index=False)
    print(f"[single] Saved coordinates CSV    -> {csv_path}")

    # ── Distance overlay (DICOM only) ──────────────────────────────────────
    if input_type == "dcm":
        if meta["conversion_factor_X"] is None or meta["conversion_factor_Y"] is None:
            print("[single] No physical delta tags found in DICOM — skipping distance overlay")
        else:
            if args.phase_estimate and args.model_weights != "lvid":
                print("[single] Note: --phase_estimate is only supported for model_weights='lvid'")
            systole_diastole = args.phase_estimate and args.model_weights == "lvid"
            dist_path = OUTPUT_FILE.rsplit(".", 1)[0] + f"_distance.{output_type}"
            process_video_with_diameter(
                video_path=OUTPUT_FILE,
                output_path=dist_path,
                conversion_factor_X=meta["conversion_factor_X"],
                conversion_factor_Y=meta["conversion_factor_Y"],
                df=df,
                ratio=meta["ratio_height"],
                systole_diastole_analysis=systole_diastole,
            )
            print(f"[single] Saved distance overlay   -> {dist_path}")
    else:
        print("[single] AVI input: physical distance not calculated (no DICOM physical tags)")

    print("-" * 60)
    print("[single] Done.")


# ======================================================================
#  MODE: folders
# ======================================================================
elif MODE == "folders":
    INPUT_FOLDER  = args.folders
    OUTPUT_FOLDER = args.output_path_folders

    print(f"[folders] Input folder : {INPUT_FOLDER}")
    print(f"[folders] Output folder: {OUTPUT_FOLDER}")
    print(f"[folders] Checking extension uniformity ...")
    check_extension_uniformity(INPUT_FOLDER)

    if OUTPUT_FOLDER:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    VIDEO_FILES = sorted([
        os.path.join(INPUT_FOLDER, f)
        for f in os.listdir(INPUT_FOLDER)
        if f.endswith(".dcm") or f.endswith(".avi")
    ])
    print(f"[folders] Found {len(VIDEO_FILES)} video files")
    print("-" * 60)

    results_all = []
    ok_count, err_count = 0, 0

    for VIDEO_FILE in tqdm(VIDEO_FILES, desc="Processing"):
        results_one = []
        try:
            frames, meta = load_video_frames(VIDEO_FILE)
            conv_x = meta["conversion_factor_X"]
            conv_y = meta["conversion_factor_Y"]
            ratio  = meta["ratio_height"]

            input_tensor = torch.tensor(np.array(frames)).float() / 255.0
            input_tensor = input_tensor.to(device).permute(0, 3, 1, 2)

            predictions = []
            for i in range(input_tensor.shape[0]):
                with torch.no_grad():
                    predictions.append(forward_pass(input_tensor[i].unsqueeze(0)))
            predictions = torch.cat(predictions, dim=0).cpu().numpy()  # (F, 2, 2)

            out_path = os.path.join(
                OUTPUT_FOLDER, os.path.basename(VIDEO_FILE) + "_generated.avi"
            )
            h, w    = input_tensor.shape[-2], input_tensor.shape[-1]
            out_avi = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*"XVID"), 30, (w, h))

            for i, (frame_t, pred) in enumerate(zip(input_tensor, predictions)):
                annotated, x1, y1, x2, y2 = make_annotated_frame(frame_t, pred)
                out_avi.write(annotated)
                results_one.append({
                    "filename":                      VIDEO_FILE,
                    "frame_number":                  i,
                    "measurement_name":              args.model_weights,
                    "PhotometricInterpretation":     meta["PhotometricInterpretation"],
                    "ultrasound_color_data_present": meta["ultrasound_color_data_present"],
                    "pred_x1": x1, "pred_y1": y1,
                    "pred_x2": x2, "pred_y2": y2,
                    "coordinates":        f"{x1}:{x2}:{y1}:{y2}",
                    "predicted_diameter": compute_diameter(x1, y1, x2, y2, ratio, conv_x, conv_y),
                })
            out_avi.release()

            if VIDEO_FILE.endswith(".dcm") and results_one and conv_x is not None and conv_y is not None:
                df_one = pd.DataFrame(results_one)
                process_video_with_diameter(
                    video_path=out_path,
                    output_path=out_path.replace(".avi", "_distance.avi"),
                    conversion_factor_X=conv_x,
                    conversion_factor_Y=conv_y,
                    df=df_one,
                    ratio=ratio,
                )

            tqdm.write(f"  [OK]    {os.path.basename(VIDEO_FILE)}  ({len(frames)} frames)  -> {out_path}")
            ok_count += 1

        except Exception as e:
            tqdm.write(f"  [Error] {os.path.basename(VIDEO_FILE)}: {e}")
            err_count += 1

        results_all.extend(results_one)

    metadata = pd.DataFrame(results_all)
    if OUTPUT_FOLDER:
        csv_path = os.path.join(OUTPUT_FOLDER, f"metadata_{args.model_weights}.csv")
        metadata.to_csv(csv_path, index=False)
        print(f"\n[folders] Saved metadata CSV -> {csv_path}")

    print(f"\n[folders] Summary: {ok_count} OK / {err_count} errors  "
          f"(total {len(VIDEO_FILES)} files)")
    print(metadata.head())
    print("-" * 60)
    print("[folders] Done.")


# ======================================================================
#  MODE: manifest
# ======================================================================
elif MODE == "manifest":
    INPUT_FOLDER  = args.folders
    OUTPUT_FOLDER = args.output_path_folders

    manifest    = pd.read_csv(args.manifest_with_frame)
    VIDEO_FILES = manifest["video_path"].unique()

    print(f"[manifest] Manifest     : {args.manifest_with_frame}")
    print(f"[manifest] Entries      : {len(manifest)}  |  Unique videos: {len(VIDEO_FILES)}")
    print(f"[manifest] Output folder: {OUTPUT_FOLDER}")

    if OUTPUT_FOLDER:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    print("-" * 60)

    results_all = []
    ok_count, err_count = 0, 0

    for VIDEO_FILE in tqdm(VIDEO_FILES, desc="Processing"):
        try:
            frames, meta = load_video_frames(VIDEO_FILE)
            conv_x = meta["conversion_factor_X"]
            conv_y = meta["conversion_factor_Y"]

            input_tensor = torch.tensor(np.array(frames)).float() / 255.0
            input_tensor = input_tensor.to(device).permute(0, 3, 1, 2)

            frame_idx   = int(manifest[manifest["video_path"] == VIDEO_FILE]["frame_number"].values[0])
            batch_input = input_tensor[frame_idx].unsqueeze(0)
            with torch.no_grad():
                prediction = forward_pass(batch_input).squeeze(0).cpu().numpy()  # (2, 2)

            annotated, x1, y1, x2, y2 = make_annotated_frame(input_tensor[frame_idx], prediction)

            if OUTPUT_FOLDER:
                out_jpg = os.path.join(
                    OUTPUT_FOLDER,
                    os.path.basename(VIDEO_FILE).rsplit(".", 1)[0] + ".jpg",
                )
                cv2.imwrite(out_jpg, annotated)
                tqdm.write(f"  [OK]    {os.path.basename(VIDEO_FILE)}  frame={frame_idx}  -> {out_jpg}")
            ok_count += 1

            results_all.append({
                "filename":                      VIDEO_FILE,
                "frame_number":                  frame_idx,
                "measurement_name":              args.model_weights,
                "PhotometricInterpretation":     meta["PhotometricInterpretation"],
                "ultrasound_color_data_present": meta["ultrasound_color_data_present"],
                "predicted_x1": x1, "predicted_y1": y1,
                "predicted_x2": x2, "predicted_y2": y2,
                "predicted_diameter": compute_diameter(x1, y1, x2, y2, meta["ratio_height"], conv_x, conv_y),
            })

        except Exception as e:
            tqdm.write(f"  [Error] {os.path.basename(VIDEO_FILE)}: {e}")
            err_count += 1

    metadata = pd.DataFrame(results_all)
    if OUTPUT_FOLDER:
        csv_path = os.path.join(OUTPUT_FOLDER, f"metadata_{args.model_weights}.csv")
        metadata.to_csv(csv_path, index=False)
        print(f"\n[manifest] Saved metadata CSV -> {csv_path}")

    print(f"\n[manifest] Summary: {ok_count} OK / {err_count} errors  "
          f"(total {len(VIDEO_FILES)} files)")
    print(metadata.head())
    print("-" * 60)
    print("[manifest] Done.")
