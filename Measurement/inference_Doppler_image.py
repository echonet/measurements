import os
import torch
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from torchvision.models.segmentation import deeplabv3_resnet50
import pydicom
from pydicom.pixel_data_handlers.util import convert_color_space
from tqdm import tqdm
from utils import get_coordinates_from_dicom, ybr_to_rgb

"""
Purpose:
    Unified Doppler peak-velocity inference script.
    Supports two modes selected automatically from the arguments:

      - single  : one DICOM file -> annotated JPG + overlay + heatmap
      - folders : folder of DICOM files -> per-file JPG + metadata CSV

    The model detects the peak-velocity point on the Doppler waveform and
    computes the velocity in cm/s using DICOM physical-delta tags.

Input  (mode = single):
    --model_weights : avvmax | trvmax | mrvmax | lvotvmax | latevel | medevel
    --file_path     : path to Doppler DICOM (.dcm)
    --output_path   : output JPG path

Input  (mode = folders):
    --model_weights       : same choices as above
    --folders             : folder of Doppler DICOM files (.dcm)
    --output_path_folders : output folder

Output (single):
    - Annotated JPG with predicted point
    - *_overlay.jpg  (heatmap blended)
    - *_heatmap.jpg
    - Printed peak velocity (cm/s)

Output (folders):
    - Per-file annotated JPG
    - metadata_<model_weights>.csv

Sample Usage:
    # Single file
    python inference_Doppler_image.py \\
        --model_weights avvmax \\
        --file_path     ./data/SAMPLE_DICOM/AVVMAX_SAMPLE_0.dcm \\
        --output_path   ./data/OUTPUT/JPG/AVVMAX_SAMPLE_GENERATED.jpg

    # Other single examples:
    # python inference_Doppler_image.py --model_weights trvmax  --file_path ./data/SAMPLE_DICOM/TRVMAX_SAMPLE_0.dcm  --output_path ./data/OUTPUT/JPG/TRVMAX_SAMPLE_GENERATED.jpg
    # python inference_Doppler_image.py --model_weights mrvmax  --file_path ./data/SAMPLE_DICOM/MRVMAX_SAMPLE_0.dcm  --output_path ./data/OUTPUT/JPG/MRVMAX_SAMPLE_GENERATED.jpg
    # python inference_Doppler_image.py --model_weights lvotvmax --file_path ./data/SAMPLE_DICOM/LVOT_VMAX_SAMPLE_0.dcm --output_path ./data/OUTPUT/JPG/LVOTVMAX_SAMPLE_GENERATED.jpg
    # python inference_Doppler_image.py --model_weights latevel  --file_path ./data/SAMPLE_DICOM/LATEVEL_SAMPLE_0.dcm  --output_path ./data/OUTPUT/JPG/LATEVEL_SAMPLE_GENERATED.jpg
    # python inference_Doppler_image.py --model_weights medevel  --file_path ./data/SAMPLE_DICOM/MEDEVEL_SAMPLE_0.dcm  --output_path ./data/OUTPUT/JPG/MEDEVEL_SAMPLE_GENERATED.jpg

    # Folder batch
    python inference_Doppler_image.py \\
        --model_weights       avvmax \\
        --folders             ./data/SAMPLE_DICOM/AVV_FOLDERS \\
        --output_path_folders ./data/OUTPUT/AVV
"""

# ─────────────────────────── ARGUMENT PARSING ────────────────────────────────
parser = ArgumentParser(description="Doppler peak-velocity inference (single / folders)")
parser.add_argument("--model_weights", type=str, required=True, choices=[
    "avvmax", "trvmax", "mrvmax", "lvotvmax",
    "latevel",   # Lateral e'
    "medevel",   # Septal e'
])
parser.add_argument("--file_path",           type=str, default=None, help="[single] Path to input .dcm")
parser.add_argument("--output_path",         type=str, default=None, help="[single] Output .jpg path")
parser.add_argument("--folders",             type=str, default=None, help="[folders] Folder of .dcm files")
parser.add_argument("--output_path_folders", type=str, default=None, help="[folders] Output folder")
args = parser.parse_args()

# ─────────────────────────── MODE DETECTION ──────────────────────────────────
if args.file_path is not None:
    MODE = "single"
elif args.folders is not None:
    MODE = "folders"
else:
    parser.error("Specify --file_path (single mode) or --folders (folders mode).")

print("=" * 60)
print(f"  Doppler Inference  |  mode={MODE}  |  weights={args.model_weights}")
print("=" * 60)

# ─────────────────────────── CONFIGURATION ───────────────────────────────────
SEGMENTATION_THRESHOLD = 0.0
DO_SIGMOID             = True

# DICOM tags
REGION_X0_SUBTAG                  = (0x0018, 0x6018)
REGION_Y0_SUBTAG                  = (0x0018, 0x601A)
REGION_X1_SUBTAG                  = (0x0018, 0x601C)
REGION_Y1_SUBTAG                  = (0x0018, 0x601E)
PHOTOMETRIC_INTERPRETATION_TAG    = (0x0028, 0x0004)
REGION_PHYSICAL_DELTA_Y_SUBTAG    = (0x0018, 0x602E)
ULTRASOUND_COLOR_DATA_PRESENT_TAG = (0x0028, 0x0014)
REFERENCE_LINE_TAG                = (0x0018, 0x6022)

# ─────────────────────────── MODEL LOADING ───────────────────────────────────
device       = "cuda:0"
weights_path = f"./weights/Doppler_models/{args.model_weights}_weights.ckpt"

print(f"[Model] Loading weights: {weights_path}")
weights  = torch.load(weights_path, map_location=device)
backbone = deeplabv3_resnet50(num_classes=1)
weights  = {k.replace("m.", ""): v for k, v in weights.items()}
print(f"[Model] {backbone.load_state_dict(weights)}")
backbone = backbone.to(device)
backbone.eval()
print(f"[Model] Ready on {device}")
print("-" * 60)

# ─────────────────────────── SHARED HELPERS ──────────────────────────────────

def forward_pass(inputs):
    """Run one forward pass; returns raw (sigmoid-activated) logit tensor."""
    logits = backbone(inputs)["out"]
    if DO_SIGMOID:
        logits = torch.sigmoid(logits)
    if SEGMENTATION_THRESHOLD is not None:
        logits[logits < SEGMENTATION_THRESHOLD] = 0.0
    return logits


def load_dicom_image(dicom_file):
    """
    Load a Doppler DICOM and apply photometric conversion + ECG masking.

    Returns:
        input_image  : (H, W, 3) uint8 RGB array
        ds           : pydicom Dataset
        meta         : dict with PhotometricInterpretation, ultrasound_color_data_present
    """
    ds = pydicom.dcmread(dicom_file)
    input_image = ds.pixel_array
    meta = {}

    meta["PhotometricInterpretation"] = (
        ds[PHOTOMETRIC_INTERPRETATION_TAG].value
        if PHOTOMETRIC_INTERPRETATION_TAG in ds else None
    )
    meta["ultrasound_color_data_present"] = (
        ds[ULTRASOUND_COLOR_DATA_PRESENT_TAG].value
        if ULTRASOUND_COLOR_DATA_PRESENT_TAG in ds else np.nan
    )

    pi = meta["PhotometricInterpretation"]

    if pi == "MONOCHROME2":
        input_image = np.stack((input_image,) * 3, axis=-1)
    elif pi == "YBR_FULL_422" and len(input_image.shape) == 3:
        input_image = convert_color_space(arr=input_image, current="YBR_FULL_422", desired="RGB")
        ecg_mask = np.logical_and(input_image[:, :, 1] > 200, input_image[:, :, 0] < 100)
        input_image[ecg_mask, :] = 0
    elif pi == "RGB":
        ecg_mask = np.logical_and(input_image[:, :, 1] > 200, input_image[:, :, 0] < 100)
        input_image[ecg_mask, :] = 0
    else:
        raise ValueError(f"Unsupported PhotometricInterpretation: {pi}")

    if len(input_image.shape) == 2:
        meta["height"], meta["width"] = input_image.shape
    else:
        meta["height"], meta["width"] = input_image.shape[0], input_image.shape[1]

    return input_image, ds, meta


def extract_doppler_region(ds, doppler_y0):
    """
    Extract the cropped Doppler region tensor (from y0 downward).

    Returns:
        doppler_area_tensor : (1, 3, H', W') float tensor on CPU
        y0, y1, x0, x1, conversion_factor, horizontal_y
    """
    doppler_region = get_coordinates_from_dicom(ds)[0]

    conversion_factor = abs(doppler_region[REGION_PHYSICAL_DELTA_Y_SUBTAG].value) \
        if REGION_PHYSICAL_DELTA_Y_SUBTAG in doppler_region else None
    y0 = doppler_region[REGION_Y0_SUBTAG].value if REGION_Y0_SUBTAG in doppler_region else None
    y1 = doppler_region[REGION_Y1_SUBTAG].value if REGION_Y1_SUBTAG in doppler_region else None
    x0 = doppler_region[REGION_X0_SUBTAG].value if REGION_X0_SUBTAG in doppler_region else None
    x1 = doppler_region[REGION_X1_SUBTAG].value if REGION_X1_SUBTAG in doppler_region else None
    horizontal_y = doppler_region[REFERENCE_LINE_TAG].value if REFERENCE_LINE_TAG in doppler_region else 0

    return conversion_factor, y0, y1, x0, x1, horizontal_y


def run_inference_on_image(input_image, y0, device):
    """
    Crop the Doppler region and run forward pass.

    Returns:
        logits_normalized : (H', W') normalized heatmap
        predicted_x, predicted_y : pixel coordinates in the FULL image
        X, Y : coordinates in the cropped region
    """
    doppler_area = input_image[y0:, :, :]
    t = torch.tensor(doppler_area).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t = t.to(device)

    with torch.no_grad():
        logit = forward_pass(t)

    max_val = logit.max().item()
    min_val = logit.min().item()
    logits_normalized = ((logit - min_val) / (max_val - min_val)).squeeze().cpu().numpy()

    max_coords  = np.unravel_index(np.argmax(logits_normalized), logits_normalized.shape)
    X           = int(max_coords[1])
    Y           = int(max_coords[0])
    predicted_x = X
    predicted_y = Y + y0

    return logits_normalized, predicted_x, predicted_y, X, Y


# ======================================================================
#  MODE: single
# ======================================================================
if MODE == "single":
    if not args.file_path.endswith(".dcm"):
        raise ValueError("--file_path must be .dcm")
    if not args.output_path.endswith(".jpg"):
        raise ValueError("--output_path must be .jpg")

    print(f"[single] Input : {args.file_path}")
    print(f"[single] Output: {args.output_path}")

    input_image, ds, meta = load_dicom_image(args.file_path)
    print(f"[single] Image shape: {input_image.shape}  |  "
          f"Photometric: {meta['PhotometricInterpretation']}")

    conversion_factor, y0, y1, x0, x1, horizontal_y = extract_doppler_region(ds, None)
    print(f"[single] Doppler region  X: {x0}-{x1}  Y: {y0}-{y1}")
    print(f"[single] Doppler baseline (reference line) at Y={horizontal_y}")
    print(f"[single] Physical delta Y = {conversion_factor}")

    logits_norm, predicted_x, predicted_y, X, Y = run_inference_on_image(input_image, y0, device)

    peak_velocity = round(conversion_factor * (predicted_y - (y0 + horizontal_y)), 2)
    print(f"[single] Predicted point  x={predicted_x}, y={predicted_y}")
    print(f"[single] Peak Velocity = {peak_velocity} cm/s")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)

    # Overlay / heatmap
    doppler_area = input_image[y0:, :, :]
    heatmap      = cv2.applyColorMap(np.uint8(255 * logits_norm), cv2.COLORMAP_MAGMA)
    overlay      = cv2.addWeighted(doppler_area, 0.25, heatmap, 0.75, 0)
    cv2.imwrite(args.output_path.replace(".jpg", "_overlay.jpg"), overlay)
    cv2.imwrite(args.output_path.replace(".jpg", "_heatmap.jpg"), heatmap)
    print(f"[single] Saved overlay  -> {args.output_path.replace('.jpg', '_overlay.jpg')}")
    print(f"[single] Saved heatmap  -> {args.output_path.replace('.jpg', '_heatmap.jpg')}")

    # Annotated image
    cv2.circle(input_image, (predicted_x, predicted_y), 10, (135, 206, 235), -1)
    plt.figure(figsize=(4, 4))
    plt.imshow(input_image, cmap="gray")
    plt.savefig(args.output_path)
    plt.close()
    print(f"[single] Saved annotated -> {args.output_path}")
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

    if OUTPUT_FOLDER:
        os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    DICOM_FILES = sorted([
        os.path.join(INPUT_FOLDER, f)
        for f in os.listdir(INPUT_FOLDER)
        if f.endswith(".dcm")
    ])
    print(f"[folders] Found {len(DICOM_FILES)} DICOM files")
    print("-" * 60)

    results    = []
    ok_count   = 0
    err_count  = 0

    for DICOM_FILE in tqdm(DICOM_FILES, desc="Processing"):
        try:
            input_image, ds, meta = load_dicom_image(DICOM_FILE)
            conversion_factor, y0, y1, x0, x1, horizontal_y = extract_doppler_region(ds, None)

            if y0 < 340 or y0 > 350:
                tqdm.write(f"  [Skip]  {os.path.basename(DICOM_FILE)}: "
                           f"y0={y0} outside expected range 340-350")
                err_count += 1
                continue

            logits_norm, predicted_x, predicted_y, X, Y = run_inference_on_image(
                input_image, y0, device
            )
            peak_velocity = round(conversion_factor * (predicted_y - (y0 + horizontal_y)), 2)

            if OUTPUT_FOLDER:
                out_jpg = os.path.join(
                    OUTPUT_FOLDER,
                    os.path.basename(DICOM_FILE).replace(".dcm", ".jpg"),
                )
                img_copy = input_image.copy()
                cv2.circle(img_copy, (predicted_x, predicted_y), 10, (135, 206, 235), -1)
                plt.figure(figsize=(4, 4))
                plt.imshow(img_copy, cmap="gray")
                plt.savefig(out_jpg)
                plt.close()

            tqdm.write(f"  [OK]    {os.path.basename(DICOM_FILE)}  "
                       f"velocity={peak_velocity} cm/s  y0={y0}")
            ok_count += 1

            results.append({
                "filename":                      DICOM_FILE,
                "measurement_name":              args.model_weights,
                "PhotometricInterpretation":     meta["PhotometricInterpretation"],
                "ultrasound_color_data_present": meta["ultrasound_color_data_present"],
                "PhysicalDeltaY":                conversion_factor,
                "y0":                            y0,
                "horizontal_line":               horizontal_y,
                "predicted_x":                   predicted_x,
                "predicted_y":                   predicted_y,
                "peak_velocity":                 peak_velocity,
                "height":                        meta["height"],
                "width":                         meta["width"],
            })

        except Exception as e:
            tqdm.write(f"  [Error] {os.path.basename(DICOM_FILE)}: {e}")
            err_count += 1

    metadata = pd.DataFrame(results)
    if OUTPUT_FOLDER:
        csv_path = os.path.join(OUTPUT_FOLDER, f"metadata_{args.model_weights}.csv")
        metadata.to_csv(csv_path, index=False)
        print(f"\n[folders] Saved metadata CSV -> {csv_path}")

    print(f"\n[folders] Summary: {ok_count} OK / {err_count} errors  "
          f"(total {len(DICOM_FILES)} files)")
    print(metadata.head())
    print("-" * 60)
    print("[folders] Done.")
