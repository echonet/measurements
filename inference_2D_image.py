from pathlib import Path
import torch
import pandas as pd
from torchvision.models.segmentation import deeplabv3_resnet50
import cv2
import numpy as np
from argparse import ArgumentParser
from utils import segmentation_to_coordinates, process_video_with_diameter, get_coordinates_from_dicom
import pydicom

"""
This file is for 2D frame-to-frame inference.
The input is a video file (AVI or DICOM) and the output is a video file (AVI) with the predicted frame-to-frame annotation.
"""

#Configuration
parser = ArgumentParser()
parser.add_argument("--model_weights", type=str, required = True, choices=[
            "ivs",
            "lvid",
            "lvpw",
            "aorta",
            "aortic_root",
            "la",
            "rv_base",
            "pa",
            "ivc",
        ])
parser.add_argument("--file_path", type=str, required = True, help= "Path to the video file (both AVI and DICOM)")
parser.add_argument("--output_path", type=str, required = True, help= "Output. Defalut should be AVI")
args = parser.parse_args()

SEGMENTATION_THRESHOLD = 0.0
DO_SIGMOID = True
N_POINTS = 2

#Dicom TAG
REGION_PHYSICAL_DELTA_X_SUBTAG = (0x0018, 0x602C)
REGION_PHYSICAL_DELTA_Y_SUBTAG = (0x0018, 0x602E)

def forward_pass(inputs):
    logits = backbone(inputs)["out"] # torch.Size([1, 2, 480, 640])
    # Step 1: Apply sigmoid if needed
    if DO_SIGMOID:
        logits = torch.sigmoid(logits)
    # Step 2: Apply segmentation threshold if needed
    if SEGMENTATION_THRESHOLD is not None:
        logits[logits < SEGMENTATION_THRESHOLD] = 0.0
    # Step 3: Convert segmentation map to coordinates
    predictions = segmentation_to_coordinates(
        logits,
        normalize=False,  # Set to True if you want normalized coordinates
        order="XY"
    )
    return predictions

print("Note: This script is for 2D frame-to-frame inference.\nOur model used the video with height of 480 and width of 640, respectively.")

input_type = None
VIDEO_FILE = args.file_path

if VIDEO_FILE.endswith(".avi"): input_type = "avi"
elif VIDEO_FILE.endswith(".dcm"): input_type = "dcm"

if input_type is None:
    raise ValueError("File path must be either .avi or .dcm")
if not args.output_path.endswith(".avi"):
    raise ValueError("Output path must be .avi")

# MODEL LOADING
device = "cuda:1" #cpu / cuda
weights_path = f"./weights/2D_models/{args.model_weights}_weights.ckpt"
weights = torch.load(weights_path)
backbone = deeplabv3_resnet50(num_classes=2)  # 39,633,986 params / num_classes should be 2
weights = {k.replace("m.", ""): v for k, v in weights.items()}
print(backbone.load_state_dict(weights)) #says <All keys matched successfully>
backbone = backbone.to(device)
backbone.eval()

frames = []
#Version AVI, LOAD VIDEO (AVI).
if input_type == "avi":
    video = cv2.VideoCapture(VIDEO_FILE)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    frames = np.array(frames)

#Version DICOM, LOAD VIDEO (Dicom).
elif input_type == "dcm":
    ds = pydicom.dcmread(VIDEO_FILE)
    input_dicom = ds.pixel_array #Frames shape (Frame, Height, Width, Channel)
    height, width = input_dicom.shape[1], input_dicom.shape[2]
    
    doppler_region = get_coordinates_from_dicom(ds)[0]
    if REGION_PHYSICAL_DELTA_X_SUBTAG in doppler_region:
        conversion_factor_X = abs(doppler_region[REGION_PHYSICAL_DELTA_X_SUBTAG].value)
    if REGION_PHYSICAL_DELTA_Y_SUBTAG in doppler_region:
        conversion_factor_Y = abs(doppler_region[REGION_PHYSICAL_DELTA_Y_SUBTAG].value)
    
    ratio_hight = height / 480 
    ratio_width = width / 640
    
    if ratio_hight != ratio_width:
        ValueError("Height and Width ratio should be same, Our model used 3:4 aspect videos.")

    for frame in input_dicom:
        if ds.PhotometricInterpretation == "YBR_FULL_422":
            frame =cv2.cvtColor(frame, cv2.COLOR_YUV2RGB)
        resized_frame = cv2.resize(frame, (640, 480)) 
        frames.append(resized_frame)


input_tensor = torch.tensor(frames)
input_tensor = input_tensor.float() / 255.0
input_tensor = input_tensor.to(device)
input_tensor = input_tensor.permute(0, 3, 1, 2)  # (F, C, H, W)

#In predictions, each frame-level prediction will be saved.
predictions = []
for i in range(input_tensor.shape[0]):
    batch = {"inputs": input_tensor[i].unsqueeze(0)} # torch.Size([1, 3, 480, 640])
    with torch.no_grad(): 
        model_output = forward_pass(batch["inputs"])
    predictions.append(model_output)
predictions = torch.cat(predictions, dim=0)
predictions = predictions.cpu().numpy()

#Make Output Video
output_video_path = args.output_path
out_avi = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"XVID"),
            30, #FPS defult
            (batch["inputs"].shape[-1], batch["inputs"].shape[-2]),  # Width, Height
        )

if not out_avi.isOpened():
    ValueError("Error: VideoWriter failed to open.")

frame_number, pred_x1s, pred_y1s, pred_x2s, pred_y2s =[], [], [], [], []
  
for i, (frame, prediction) in enumerate(zip(input_tensor, predictions)):
    frame = frame.permute(1, 2, 0).cpu().numpy()
    frame = np.clip(frame, 0, 1)
    frame = (frame * 255).astype(np.uint8)
    
    #Plot points (circle)
    for point, color in zip(prediction, [(235, 206, 135), (235, 206, 135)]):
        point_0, point_1 = point[0], point[1]
        cv2.circle(frame, (int(point_0), int(point_1)), 5, color, -1)
    
    #Plot line
    cv2.line(frame, 
             (int(prediction[0][0]), int(prediction[0][1])),
             (int(prediction[1][0]), int(prediction[1][1])),
             (235, 206, 135), 
             2)
    
    pred_x1s.append(prediction[0][0])
    pred_y1s.append(prediction[0][1])
    pred_x2s.append(prediction[1][0])
    pred_y2s.append(prediction[1][1])
    frame_number.append(i)
    
    out_avi.write(frame)
    
out_avi.release()
print(f"Done, please check {output_video_path}")

df = pd.DataFrame({
    "frame_number": frame_number,
    "pred_x1": pred_x1s,
    "pred_y1": pred_y1s,
    "pred_x2": pred_x2s,
    "pred_y2": pred_y2s,
})

if input_type == "avi":
    print("Completed. Distance between two points is not calculated from .avi input.")

if input_type == "dcm":
    process_video_with_diameter(video_path = output_video_path, 
                                output_path = output_video_path.replace(".avi", "_distance.avi"),
                                conversion_factor_X = conversion_factor_X,
                                conversion_factor_Y = conversion_factor_Y,
                                df = df)
    print("Distance between two points is calculated from .dcm input.")
    print(f"Completed.Please check {output_video_path.replace('.avi', '_distance.avi')}")


#SAMPLE SCRIPT

#python inference_2D_image.py --model_weights "ivs" 
#--file_path "./SAMPLE_DICOM/IVS_SAMPLE_0.dcm" 
#--output_path "./OUTPUT/AVI/IVS_SAMPLE_GENERATED.avi"

#python inference_2D_image.py --model_weights "lvid" 
#--file_path "./SAMPLE_DICOM/LVID_SAMPLE_0.dcm" 
#--output_path "./OUTPUT/AVI/LVID_SAMPLE_GENERATED.avi"

#python inference_2D_image.py --model_weights "lvpw"
#--file_path "./SAMPLE_DICOM/LVPW_SAMPLE_0.dcm"
#--output_path "./OUTPUT/AVI/LVPW_SAMPLE_GENERATED.avi"

#python inference_2D_image.py --model_weights "aorta"
#--file_path "./SAMPLE_DICOM/AORTA_SAMPLE_0.dcm"
#--output_path "./OUTPUT/AVI/AORTA_SAMPLE_GENERATED.avi"

#python inference_2D_image.py --model_weights "aortic_root"
#--file_path "./SAMPLE_DICOM/AORTIC_ROOT_SAMPLE_0.dcm"
#--output_path "./OUTPUT/AVI/AORTIC_ROOT_SAMPLE_GENERATED.avi"

#python inference_2D_image.py --model_weights "la"
#--file_path "./SAMPLE_DICOM/LA_SAMPLE_0.dcm"
#--output_path "./OUTPUT/AVI/LA_SAMPLE_GENERATED.avi"

#python inference_2D_image.py --model_weights "rv_base"
#--file_path "./SAMPLE_DICOM/RV_BASE_SAMPLE_0.dcm"
#--output_path "./OUTPUT/AVI/RV_BASE_SAMPLE_GENERATED.avi"

#python inference_2D_image.py --model_weights "pa"
#--file_path "./SAMPLE_DICOM/PA_SAMPLE_0.dcm"
#--output_path "./OUTPUT/AVI/PA_SAMPLE_GENERATED.avi"

#python inference_2D_image.py --model_weights "ivc"
#--file_path "./SAMPLE_DICOM/IVC_SAMPLE_0.dcm"
#--output_path "./OUTPUT/AVI/IVC_SAMPLE_GENERATED.avi"

#S  03_/generate_overlay.py
#S  03_/inference_video_burntin.py