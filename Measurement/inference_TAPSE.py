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
from utils import get_coordinates_from_dicom, calculate_weighted_centroids_with_meshgrid, ybr_to_rgb

"""
Purpose:
    Unified TAPSE (Tricuspid Annular Plane Systolic Excursion) inference script.
    Supports two modes selected automatically from the arguments:

      - single  : one DICOM file -> annotated JPG + printed TAPSE value
      - folders : folder of DICOM files -> per-file JPG + metadata CSV

    Uses a 2-channel DeepLabV3+ to detect two landmark points on the M-mode
    trace, then computes TAPSE (cm) using DICOM physical-delta tags.

    Input DICOMs are expected to be 768 x 1024 with a Doppler region at y0 ~ 342-348.

Input  (mode = single):
    --file_path   : path to TAPSE DICOM (.dcm)
    --output_path : output JPG path

Input  (mode = folders):
    --folders             : folder of TAPSE DICOM files (.dcm)
    --output_path_folders : output folder

Output (single):
    - Annotated JPG with predicted landmark points
    - Printed TAPSE value (cm)

Output (folders):
    - Per-file annotated JPG
    - metadata_tapse.csv with TAPSE values and DICOM metadata

Sample Usage:
    # Single file
    python inference_TAPSE.py \\
        --file_path   ./data/SAMPLE_DICOM/TAPSE_SAMPLE_0.dcm \\
        --output_path ./data/OUTPUT/JPG/TAPSE_SAMPLE_GENERATED.jpg

    # Folder batch
    python inference_TAPSE.py \\
        --folders             ./data/SAMPLE_DICOM/TAPSE_FOLDERS \\
        --output_path_folders ./data/OUTPUT/TAPSE
"""

# ─────────────────────────── ARGUMENT PARSING ────────────────────────────────
parser = ArgumentParser(description="TAPSE inference (single / folders)")
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
print(f"  TAPSE Inference  |  mode={MODE}")
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
REGION_PHYSICAL_DELTA_X_SUBTAG    = (0x0018, 0x602C)
REGION_PHYSICAL_DELTA_Y_SUBTAG    = (0x0018, 0x602E)
ULTRASOUND_COLOR_DATA_PRESENT_TAG = (0x0028, 0x0014)

# ─────────────────────────── MODEL LOADING ───────────────────────────────────
device       = "cuda:0"
weights_path = "./weights/Doppler_models/tapse_2c_weights.ckpt"

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
    """Run forward pass; returns (x1, y1, x2, y2) sorted left-to-right."""
    logits = backbone(inputs)["out"]
    if DO_SIGMOID:
        logits = torch.sigmoid(logits)
    if SEGMENTATION_THRESHOLD is not None:
        logits[logits < SEGMENTATION_THRESHOLD] = 0.0

    logits_numpy = logits.squeeze().detach().cpu().numpy()

    logits_first = logits_numpy[1, :, :]
    rng = logits_first.max() - logits_first.min()
    logits_first = (logits_first - logits_first.min()) / rng if rng > 0 else logits_first
    _, _, _, max_loc_first_channel = cv2.minMaxLoc(logits_first)

    logits_second = logits_numpy[0, :, :]
    rng = logits_second.max() - logits_second.min()
    logits_second = (logits_second - logits_second.min()) / rng if rng > 0 else logits_second
    _, _, _, max_loc_second_channel = cv2.minMaxLoc(logits_second)

    combine_logit = logits_first + logits_second
    _, _, _, max_loc_combine = cv2.minMaxLoc(combine_logit)

    diff_first  = np.sqrt((max_loc_combine[0]-max_loc_first_channel[0])**2  + (max_loc_combine[1]-max_loc_first_channel[1])**2)
    diff_second = np.sqrt((max_loc_combine[0]-max_loc_second_channel[0])**2 + (max_loc_combine[1]-max_loc_second_channel[1])**2)

    centroids_first,  _ = calculate_weighted_centroids_with_meshgrid(logits_first)
    centroids_second, _ = calculate_weighted_centroids_with_meshgrid(logits_second)
    centroids,        _ = calculate_weighted_centroids_with_meshgrid(combine_logit)

    d_btw_max = {c: np.sqrt((max_loc_combine[0]-c[0])**2 + (max_loc_combine[1]-c[1])**2) for c in centroids}
    try:
        min_coord = min(d_btw_max, key=d_btw_max.get)
    except ValueError:
        raise ValueError("min_distance_coord not found (low prediction score). Use good-quality TAPSE data.")

    if diff_second - diff_first > 15:
        pair_source = centroids_second
    elif diff_first - diff_second > 15:
        pair_source = centroids_first
    else:
        pair_source = centroids

    d_btw = {c: np.sqrt((min_coord[0]-c[0])**2 + (min_coord[1]-c[1])**2) for c in pair_source}
    non_zero = {k: v for k, v in d_btw.items() if v > 15}
    try:
        paired_coord = min(non_zero, key=non_zero.get)
    except ValueError:
        raise ValueError("Pair point not found (low prediction score). Use good-quality TAPSE data.")

    point_x1, point_y1 = min_coord
    point_x2, point_y2 = paired_coord

    if point_x1 > point_x2:
        point_x1, point_y1, point_x2, point_y2 = point_x2, point_y2, point_x1, point_y1

    return point_x1, point_y1, point_x2, point_y2


def load_dicom_image(dicom_file):
    """Load DICOM, apply photometric conversion + ECG masking."""
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
        input_image = ybr_to_rgb(input_image)
        ecg_mask = np.logical_and(input_image[:, :, 1] > 200, input_image[:, :, 0] < 100)
        input_image[ecg_mask, :] = 0
    elif pi == "RGB":
        ecg_mask = np.logical_and(input_image[:, :, 1] > 200, input_image[:, :, 0] < 100)
        input_image[ecg_mask, :] = 0
    else:
        raise ValueError(f"Unsupported PhotometricInterpretation: {pi}")

    return input_image, ds, meta


def extract_doppler_tags(ds):
    """Return y0, y1, x0, x1, PhysicalDeltaX, PhysicalDeltaY from DICOM tags."""
    doppler_region = get_coordinates_from_dicom(ds)[0]
    PhysicalDeltaX = abs(doppler_region[REGION_PHYSICAL_DELTA_X_SUBTAG].value) \
        if REGION_PHYSICAL_DELTA_X_SUBTAG in doppler_region else None
    PhysicalDeltaY = abs(doppler_region[REGION_PHYSICAL_DELTA_Y_SUBTAG].value) \
        if REGION_PHYSICAL_DELTA_Y_SUBTAG in doppler_region else None
    y0 = doppler_region[REGION_Y0_SUBTAG].value if REGION_Y0_SUBTAG in doppler_region else None
    y1 = doppler_region[REGION_Y1_SUBTAG].value if REGION_Y1_SUBTAG in doppler_region else None
    x0 = doppler_region[REGION_X0_SUBTAG].value if REGION_X0_SUBTAG in doppler_region else None
    x1 = doppler_region[REGION_X1_SUBTAG].value if REGION_X1_SUBTAG in doppler_region else None
    return y0, y1, x0, x1, PhysicalDeltaX, PhysicalDeltaY


def compute_tapse(x1, y1, x2, y2, delta_x, delta_y):
    return round(float(np.sqrt(
        (abs(x1 - x2) * delta_x) ** 2 +
        (abs(y1 - y2) * delta_y) ** 2
    )), 2)


def save_jpg(input_image, x1, y1, x2, y2, y0, out_path):
    """Save annotated JPG with two red scatter points and connecting line."""
    img_copy = input_image.copy()
    cv2.line(img_copy, (x1, y1 + y0), (x2, y2 + y0), (255, 0, 0), 2)
    plt.figure(figsize=(8, 8))
    plt.imshow(img_copy, cmap="gray")
    plt.scatter(x1, y1 + y0, color="red", s=20)
    plt.scatter(x2, y2 + y0, color="red", s=20)
    plt.savefig(out_path)
    plt.close()


# ======================================================================
#  MODE: single
# ======================================================================
if MODE == "single":
    if not args.file_path.endswith(".dcm"):
        raise ValueError("--file_path must be .dcm")
    if not args.output_path.endswith(".jpg"):
        raise ValueError("--output_path must end in .jpg")

    print(f"[single] Input : {args.file_path}")
    print(f"[single] Output: {args.output_path}")

    input_image, ds, meta = load_dicom_image(args.file_path)
    print(f"[single] Image shape: {input_image.shape}  |  Photometric: {meta['PhotometricInterpretation']}")

    y0, y1, x0, x1, PhysicalDeltaX, PhysicalDeltaY = extract_doppler_tags(ds)
    print(f"[single] Doppler region  X: {x0}-{x1}  Y: {y0}-{y1}")
    print(f"[single] PhysicalDelta  X={PhysicalDeltaX}  Y={PhysicalDeltaY}")

    input_dicom_doppler_area = input_image[y0:, :, :]
    t = torch.tensor(input_dicom_doppler_area).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t = t.to(device)

    with torch.no_grad():
        point_x1, point_y1, point_x2, point_y2 = forward_pass(t)

    print(f"[single] Points: ({point_x1}, {point_y1}) and ({point_x2}, {point_y2})")

    if (point_x1 < point_x2) and (point_y1 < point_y2):
        print("[single] Warning: left point is higher than right — check TAPSE orientation")

    tapse = compute_tapse(point_x1, point_y1, point_x2, point_y2, PhysicalDeltaX, PhysicalDeltaY)
    print(f"[single] TAPSE = {tapse} cm")

    os.makedirs(os.path.dirname(os.path.abspath(args.output_path)), exist_ok=True)
    save_jpg(input_image, point_x1, point_y1, point_x2, point_y2, y0, args.output_path)
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

    results   = []
    ok_count  = 0
    err_count = 0

    for DICOM_FILE in tqdm(DICOM_FILES, desc="Processing"):
        try:
            input_image, ds, meta = load_dicom_image(DICOM_FILE)
            y0, y1, x0, x1, PhysicalDeltaX, PhysicalDeltaY = extract_doppler_tags(ds)

            if y0 < 340 or y0 > 350:
                tqdm.write(f"  [Skip]  {os.path.basename(DICOM_FILE)}: y0={y0} outside range")
                err_count += 1
                continue

            input_dicom_doppler_area = input_image[342:, :, :]
            t = torch.tensor(input_dicom_doppler_area).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            t = t.to(device)

            with torch.no_grad():
                point_x1, point_y1, point_x2, point_y2 = forward_pass(t)

            if (point_x1 < point_x2) and (point_y1 < point_y2):
                tqdm.write(f"  [Skip]  {os.path.basename(DICOM_FILE)}: invalid TAPSE point order")
                err_count += 1
                continue

            tapse = compute_tapse(point_x1, point_y1, point_x2, point_y2, PhysicalDeltaX, PhysicalDeltaY)

            if OUTPUT_FOLDER:
                out_jpg = os.path.join(
                    OUTPUT_FOLDER,
                    os.path.basename(DICOM_FILE).replace(".dcm", ".jpg"),
                )
                save_jpg(input_image, point_x1, point_y1, point_x2, point_y2, y0, out_jpg)

            tqdm.write(f"  [OK]    {os.path.basename(DICOM_FILE)}  TAPSE={tapse} cm")
            ok_count += 1

            results.append({
                "filename":                      DICOM_FILE,
                "measurement_name":              "TAPSE",
                "PhotometricInterpretation":     meta["PhotometricInterpretation"],
                "ultrasound_color_data_present": meta["ultrasound_color_data_present"],
                "PhysicalDeltaX":                PhysicalDeltaX,
                "PhysicalDeltaY":                PhysicalDeltaY,
                "y0":                            y0,
                "predicted_x1":                  point_x1,
                "predicted_y1":                  point_y1,
                "predicted_x2":                  point_x2,
                "predicted_y2":                  point_y2,
                "TAPSE":                         tapse,
            })

        except Exception as e:
            tqdm.write(f"  [Error] {os.path.basename(DICOM_FILE)}: {e}")
            err_count += 1

    metadata = pd.DataFrame(results)
    if OUTPUT_FOLDER:
        csv_path = os.path.join(OUTPUT_FOLDER, "metadata_tapse.csv")
        metadata.to_csv(csv_path, index=False)
        print(f"\n[folders] Saved metadata CSV -> {csv_path}")

    print(f"\n[folders] Summary: {ok_count} OK / {err_count} errors  "
          f"(total {len(DICOM_FILES)} files)")
    print(metadata.head())
    print("-" * 60)
    print("[folders] Done.")
