from pathlib import Path
import torch
import pandas as pd
from torchvision.models.segmentation import deeplabv3_resnet50
import cv2
import numpy as np
from argparse import ArgumentParser
from utils import segmentation_to_coordinates
import pydicom

"""
This file is for 2D frame-to-frame inference.
The input is a video file (AVI or DICOM) and the output is a video file (AVI) with the predicted frame-to-frame annotation.
"""

#Configuration
parser = ArgumentParser()
parser.add_argument("--model_weights", type=str, required = True ,choices=["ivs", "ALL_PW", "SINGLE"], default=None)
parser.add_argument("--file_path", type=str, required = True, help= "Path to the video file (both AVI and DICOM)", default=None)
parser.add_argument("--output_path", type=str, required = True, help= "Output. Defalut should be AVI", default=None)
args = parser.parse_args()

SEGMENTATION_THRESHOLD = 0.0
DO_SIGMOID = True
N_POINTS = 2

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

print("Note: This script is for 2D frame-to-frame inference.\nInput height and width are 480 and 640 respectively.")
if not args.file_path.endswith(".avi") and not args.file_path.endswith(".dcm"):
    raise ValueError("File path must be either .avi or .dcm")
if not args.output_path.endswith(".avi"):
    raise ValueError("Output path must be .avi")

#MODEL LOADING
device = "cuda:1" #cpu / cuda
weights_path = f"/workspace/yuki/measurements_internal/measurements/weights/{args.model_weights}_weights.ckpt"
weights = torch.load(weights_path)
backbone = deeplabv3_resnet50(num_classes=2)  # 39,633,986 params
weights = {k.replace("m.", ""): v for k, v in weights.items()}
print(backbone.load_state_dict(weights)) #says <All keys matched successfully>
backbone = backbone.to(device)
backbone.eval()

#LOAD VIDEO (AVI). If you have a dicom video, please convert it to AVI first. Our dataset was in AVI format and ECG and Respiratory waveform was masked.
VIDEO_FILE = args.file_path #"/workspace/yuki/measurements_internal/measurements/SAMPLE_AVI/SAMPLE_PLAX.avi" #
frames = []
if VIDEO_FILE.endswith(".avi"):
    video = cv2.VideoCapture(VIDEO_FILE)
    while True:
        ret, frame = video.read()
        if not ret:
            break
        frames.append(frame)
    frames = np.array(frames)
    
elif VIDEO_FILE.endswith(".dcm"):
    ds = pydicom.dcmread(VIDEO_FILE)
    input_dicom = ds.pixel_array
    for frame in input_dicom:
        if ds.PhotometricInterpretation == "YBR_FULL_422":
            frame =cv2.cvtColor(frame, cv2.COLOR_YUV2RGB)
    resized_frame = cv2.resize(frame, (640, 480)) 
    frames.append(resized_frame)
    
input_tensor = torch.tensor(frames)
input_tensor = input_tensor.float() / 255.0
input_tensor = input_tensor.to(device)
input_tensor = input_tensor.permute(0, 3, 1, 2)  # (F: frames, C: channels, H: height, W: width) #torch.Size([FRAME, 3, 480, 640])

#In predictions, each frame-level prediction will be saved.
predictions = []
for i in range(input_tensor.shape[0]):
    batch = {"inputs": input_tensor[i].unsqueeze(0)} # torch.Size([1, 3, 480, 640])
    with torch.no_grad(): 
        model_output = forward_pass(batch["inputs"])
    predictions.append(model_output)
predictions = torch.cat(predictions, dim=0)
predictions = predictions.cpu().numpy() #((FRAME), 4)

#Make Output Video
output_video_path = args.output_path #"/workspace/yuki/measurements_internal/measurements/SAMPLE_AVI/SAMPLE_PLAX_GENERATED.avi" #
out_avi = cv2.VideoWriter(
            output_video_path,
            cv2.VideoWriter_fourcc(*"XVID"),
            30, #FPS defult
            (batch["inputs"].shape[-1], batch["inputs"].shape[-2]),  # Width, Height
        )

if not out_avi.isOpened():
    ValueError("Error: VideoWriter failed to open.")
    
for i, (frame, prediction) in enumerate(zip(input_tensor, predictions)):
    frame = frame.permute(1, 2, 0).cpu().numpy()
    frame = np.clip(frame, 0, 1)
    frame = (frame * 255).astype(np.uint8)
    
    for point, color in zip(prediction, [(235, 206, 135), (235, 206, 135)]):
        x, y = point[0], point[1]
        cv2.circle(frame, (int(x), int(y)), 5, color, -1)
    cv2.line(frame, (int(prediction[0][0]), int(prediction[0][1])),
             (int(prediction[1][0]), int(prediction[1][1])),
             (235, 206, 135), 2)
    out_avi.write(frame)
out_avi.release()
print(f"Done, please check {output_video_path}")

#SAMPLE SCRIPT
#python inference_2D_image.py --model_weights "ivs" 
#--file_path "/workspace/yuki/measurements_internal/measurements/SAMPLE_AVI/SAMPLE_PLAX.avi" 
#--output_path "/workspace/yuki/measurements_internal/measurements/SAMPLE_AVI/SAMPLE_PLAX_GENERATED.avi"
