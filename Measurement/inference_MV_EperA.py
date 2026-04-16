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
    Unified MV E/A Doppler inference script.
    Supports two modes selected automatically from the arguments:

      - single  : one DICOM file -> annotated JPG + printed E, A, E/A values
      - folders : folder of DICOM files -> per-file JPG + metadata CSV

    Uses a 2-channel DeepLabV3+ to detect E-peak and A-peak landmark points,
    then computes E/A ratio using DICOM physical-delta tags.

    Note: model may return an error if A-velocity is not detected well
    (low prediction score). Use good-quality MVPeak Doppler DICOMs.

Input  (mode = single):
    --file_path   : path to MVPeak Doppler DICOM (.dcm)
    --output_path : output JPG path

Input  (mode = folders):
    --folders             : folder of MVPeak Doppler DICOM files (.dcm)
    --output_path_folders : output folder

Output (single):
    - Annotated JPG with predicted E (red) and A (blue) points
    - Printed E_Vel, A_Vel, E/A ratio (cm/s)

Output (folders):
    - Per-file annotated JPG
    - metadata_mvpeak.csv with E/A velocities, ratio, and DICOM metadata

Sample Usage:
    # Single file
    python inference_MV_EperA.py \\
        --file_path   ./data/SAMPLE_DICOM/MVPEAK_SAMPLE_0.dcm \\
        --output_path ./data/OUTPUT/JPG/MVPEAK_SAMPLE_GENERATED.jpg

    # Folder batch
    python inference_MV_EperA.py \\
        --folders             ./data/SAMPLE_DICOM/MVPEAK_FOLDERS \\
        --output_path_folders ./data/OUTPUT/MVPEAK
"""

# ─────────────────────────── ARGUMENT PARSING ────────────────────────────────
parser = ArgumentParser(description="MV E/A Doppler inference (single / folders)")
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
print(f"  MV E/A Inference  |  mode={MODE}")
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
weights_path = "./weights/Doppler_models/mvpeak_2c_weights.ckpt"

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
    """
    Run forward pass and return predicted (x1,y1), (x2,y2) peak coordinates
    and a swapped flag (True if E/A points were swapped to enforce left=E).
    """
    logits = backbone(inputs)["out"]

    if DO_SIGMOID:
        logits = torch.sigmoid(logits)
    if SEGMENTATION_THRESHOLD is not None:
        logits[logits < SEGMENTATION_THRESHOLD] = 0.0

    logits_numpy = logits.squeeze().detach().cpu().numpy()

    logits_first = logits_numpy[1, :, :]   # channel 1 -> E
    max_val_first, min_val_first = logits_first.max(), logits_first.min()
    logits_first = (logits_first - min_val_first) / (max_val_first - min_val_first)
    _, _, _, max_loc_first_channel = cv2.minMaxLoc(logits_first)

    logits_second = logits_numpy[0, :, :]  # channel 0 -> A
    max_val_second, min_val_second = logits_second.max(), logits_second.min()
    logits_second = (logits_second - min_val_second) / (max_val_second - min_val_second)
    _, _, _, max_loc_second_channel = cv2.minMaxLoc(logits_second)

    combine_logit = logits_first + logits_second
    _, _, _, max_loc_combine = cv2.minMaxLoc(combine_logit)

    diff_maxloc_combine_first  = np.sqrt(
        (max_loc_combine[0] - max_loc_first_channel[0])**2 +
        (max_loc_combine[1] - max_loc_first_channel[1])**2
    )
    diff_maxloc_combine_second = np.sqrt(
        (max_loc_combine[0] - max_loc_second_channel[0])**2 +
        (max_loc_combine[1] - max_loc_second_channel[1])**2
    )

    centroids_first,  _ = calculate_weighted_centroids_with_meshgrid(logits_first)
    centroids_second, _ = calculate_weighted_centroids_with_meshgrid(logits_second)
    centroids,        _ = calculate_weighted_centroids_with_meshgrid(combine_logit)

    # Pick closest centroid to the max-logit location
    distance_centroid_btw_maxlogits = {
        c: np.sqrt((max_loc_combine[0]-c[0])**2 + (max_loc_combine[1]-c[1])**2)
        for c in centroids
    }
    try:
        min_distance_coord = min(distance_centroid_btw_maxlogits, key=distance_centroid_btw_maxlogits.get)
    except ValueError:
        raise ValueError("min_distance_coord not found (low prediction score). Use good-quality MVPeak data.")

    # Choose pairing channel
    if diff_maxloc_combine_second - diff_maxloc_combine_first > 15:
        pair_source = centroids_second
    elif diff_maxloc_combine_first - diff_maxloc_combine_second > 15:
        pair_source = centroids_first
    else:
        pair_source = centroids

    distance_btw_centroids = {
        c: np.sqrt((min_distance_coord[0]-c[0])**2 + (min_distance_coord[1]-c[1])**2)
        for c in pair_source
    }
    non_zero = {k: v for k, v in distance_btw_centroids.items() if v > 15}
    try:
        min_distance_paired_coord = min(non_zero, key=non_zero.get)
    except ValueError:
        raise ValueError("Pair point not found (low A-velocity score). Use good-quality MVPeak data.")

    point_x1, point_y1 = min_distance_coord
    point_x2, point_y2 = min_distance_paired_coord

    swapped = False
    if point_x1 > point_x2:
        point_x1, point_y1, point_x2, point_y2 = point_x2, point_y2, point_x1, point_y1
        swapped = True

    if abs(point_x1 - point_x2) > 300:
        raise ValueError("Distance between E/A points > 300 px. Use good-quality Doppler data.")

    return point_x1, point_y1, point_x2, point_y2, swapped


def load_dicom_image(dicom_file):
    """Load a Doppler DICOM, apply photometric conversion + ECG masking."""
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

    if len(input_image.shape) == 2:
        meta["height"], meta["width"] = input_image.shape
    else:
        meta["height"], meta["width"] = input_image.shape[0], input_image.shape[1]

    return input_image, ds, meta


def extract_doppler_tags(ds):
    """Return y0, x0, x1, PhysicalDeltaY, horizontal_y from DICOM tags."""
    doppler_region   = get_coordinates_from_dicom(ds)[0]
    PhysicalDeltaY   = abs(doppler_region[REGION_PHYSICAL_DELTA_Y_SUBTAG].value) \
        if REGION_PHYSICAL_DELTA_Y_SUBTAG in doppler_region else None
    y0 = doppler_region[REGION_Y0_SUBTAG].value if REGION_Y0_SUBTAG in doppler_region else None
    y1 = doppler_region[REGION_Y1_SUBTAG].value if REGION_Y1_SUBTAG in doppler_region else None
    x0 = doppler_region[REGION_X0_SUBTAG].value if REGION_X0_SUBTAG in doppler_region else None
    x1 = doppler_region[REGION_X1_SUBTAG].value if REGION_X1_SUBTAG in doppler_region else None
    horizontal_y = doppler_region[REFERENCE_LINE_TAG].value \
        if REFERENCE_LINE_TAG in doppler_region else 0
    return y0, y1, x0, x1, PhysicalDeltaY, horizontal_y


def save_jpg(input_image, x1, y1, x2, y2, y0, out_path):
    """Save annotated JPG with E (red) and A (blue) scatter points."""
    plt.figure(figsize=(8, 8))
    plt.imshow(input_image, cmap="gray")
    plt.scatter(x1, y1 + y0, color="red",  s=20)
    plt.scatter(x2, y2 + y0, color="blue", s=20)
    plt.savefig(out_path)
    plt.close()


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
    print(f"[single] Image shape: {input_image.shape}  |  Photometric: {meta['PhotometricInterpretation']}")

    y0, y1, x0, x1, PhysicalDeltaY, horizontal_y = extract_doppler_tags(ds)
    print(f"[single] Doppler region  X: {x0}-{x1}  Y: {y0}-{y1}")
    print(f"[single] Baseline Y={horizontal_y}  |  PhysicalDeltaY={PhysicalDeltaY}")

    if y0 < 340 or y0 > 350:
        raise ValueError(f"y0={y0} outside expected range 340-350. Model trained on y0 ~ 342-348.")

    input_dicom_doppler_area = input_image[342:, :, :]
    t = torch.tensor(input_dicom_doppler_area).permute(2, 0, 1).unsqueeze(0).float() / 255.0
    t = t.to(device)

    with torch.no_grad():
        point_x1, point_y1, point_x2, point_y2, swapped = forward_pass(t)

    if swapped:
        print("[single] Note: E/A points were swapped to enforce left=E, right=A")

    Inference_E_Vel = round(abs((point_y1 - horizontal_y) * PhysicalDeltaY), 4)
    Inference_A_Vel = round(abs((point_y2 - horizontal_y) * PhysicalDeltaY), 4)
    Inference_EperA = round(Inference_E_Vel / Inference_A_Vel, 3) if Inference_A_Vel != 0 else float("nan")

    print(f"[single] Predicted E point: ({point_x1}, {point_y1})")
    print(f"[single] Predicted A point: ({point_x2}, {point_y2})")
    print(f"[single] E_Vel = {Inference_E_Vel} cm/s")
    print(f"[single] A_Vel = {Inference_A_Vel} cm/s")
    print(f"[single] E/A   = {Inference_EperA}")

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

    results              = []
    ok_count             = 0
    count_swap           = 0
    count_missing        = 0
    count_dist_large     = 0
    count_other_errors   = 0

    for DICOM_FILE in tqdm(DICOM_FILES, desc="Processing"):
        try:
            input_image, ds, meta = load_dicom_image(DICOM_FILE)
            y0, y1, x0, x1, PhysicalDeltaY, horizontal_y = extract_doppler_tags(ds)

            if y0 < 340 or y0 > 350:
                tqdm.write(f"  [Skip]  {os.path.basename(DICOM_FILE)}: y0={y0} outside range")
                count_other_errors += 1
                continue

            input_dicom_doppler_area = input_image[342:, :, :]
            t = torch.tensor(input_dicom_doppler_area).permute(2, 0, 1).unsqueeze(0).float() / 255.0
            t = t.to(device)

            with torch.no_grad():
                point_x1, point_y1, point_x2, point_y2, swapped = forward_pass(t)

            if swapped:
                count_swap += 1
                tqdm.write(f"  [Swap]  {os.path.basename(DICOM_FILE)}: E/A swapped")

            Inference_E_Vel = round(abs((point_y1 - horizontal_y) * PhysicalDeltaY), 4)
            Inference_A_Vel = round(abs((point_y2 - horizontal_y) * PhysicalDeltaY), 4)
            Inference_EperA = round(Inference_E_Vel / Inference_A_Vel, 3) if Inference_A_Vel != 0 else float("nan")

            if OUTPUT_FOLDER:
                out_jpg = os.path.join(
                    OUTPUT_FOLDER,
                    os.path.basename(DICOM_FILE).replace(".dcm", ".jpg"),
                )
                save_jpg(input_image, point_x1, point_y1, point_x2, point_y2, y0, out_jpg)

            tqdm.write(f"  [OK]    {os.path.basename(DICOM_FILE)}  "
                       f"E={Inference_E_Vel}  A={Inference_A_Vel}  E/A={Inference_EperA}")
            ok_count += 1

            results.append({
                "filename":                      DICOM_FILE,
                "measurement_name":              "MV_Peak",
                "PhotometricInterpretation":     meta["PhotometricInterpretation"],
                "ultrasound_color_data_present": meta["ultrasound_color_data_present"],
                "PhysicalDeltaY":                PhysicalDeltaY,
                "y0":                            y0,
                "horizontal_line":               horizontal_y,
                "predicted_xe":                  point_x1,
                "predicted_ye":                  point_y1,
                "predicted_xa":                  point_x2,
                "predicted_ya":                  point_y2,
                "Inference_E_Vel":               Inference_E_Vel,
                "Inference_A_Vel":               Inference_A_Vel,
                "Inference_EperA":               Inference_EperA,
                "height":                        meta["height"],
                "width":                         meta["width"],
            })

        except ValueError as e:
            msg = str(e)
            if "min_distance_coord" in msg:
                count_missing += 1
            elif "Distance between" in msg or "distance between" in msg:
                count_dist_large += 1
            else:
                count_other_errors += 1
            tqdm.write(f"  [Error] {os.path.basename(DICOM_FILE)}: {msg}")
        except Exception as e:
            tqdm.write(f"  [Error] {os.path.basename(DICOM_FILE)}: {e}")
            count_other_errors += 1

    metadata = pd.DataFrame(results)
    if OUTPUT_FOLDER:
        csv_path = os.path.join(OUTPUT_FOLDER, "metadata_mvpeak.csv")
        metadata.to_csv(csv_path, index=False)
        print(f"\n[folders] Saved metadata CSV -> {csv_path}")

    print(f"\n[folders] Summary: {ok_count} OK / {count_other_errors + count_missing + count_dist_large} errors  "
          f"(total {len(DICOM_FILES)} files)")
    print(f"  E/A swapped: {count_swap}  |  missing point: {count_missing}  |  dist>300: {count_dist_large}")
    print(metadata.head())
    print("-" * 60)
    print("[folders] Done.")
