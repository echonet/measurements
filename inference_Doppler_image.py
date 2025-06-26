import torch
import pandas as pd
from torchvision.models.segmentation import deeplabv3_resnet50
import cv2
import numpy as np
from argparse import ArgumentParser
from utils import segmentation_to_coordinates, get_coordinates_from_dicom, ybr_to_rgb
import pydicom
from pydicom.pixel_data_handlers.util import  convert_color_space
import matplotlib.pyplot as plt

"""
This script is for Doppler Velocity inference. 
The input is a DICOM file with a Doppler region. 
The script will output the Predicted Annotation and Velocity (Like TRVMAX or AVVMAX).

Please make sure your input model is for Doppler Velocity and not low-quality images.

"""

parser = ArgumentParser()
parser.add_argument("--model_weights", type=str, required = True, choices=[
            "avvmax",
            "trvmax",
            "mrvmax",
            "lvotvmax",
            "latevel", #Latral e'  
            "medevel" #Septal e'
        ])
parser.add_argument("--file_path", type=str, required = True, help= "Path to the Dicom file (.dcm)")
parser.add_argument("--output_path", type=str, required = True, help= "Output. Defalut -> jpg")
args = parser.parse_args()


#Configuration
SEGMENTATION_THRESHOLD = 0.0
DO_SIGMOID = True
N_POINTS = 1

ULTRASOUND_REGIONS_TAG = (0x0018, 0x6011)
REGION_X0_SUBTAG = (0x0018, 0x6018)  # left
REGION_Y0_SUBTAG = (0x0018, 0x601A)  # top
REGION_X1_SUBTAG = (0x0018, 0x601C)  # right
REGION_Y1_SUBTAG = (0x0018, 0x601E)  # bottom
STUDY_DESCRIPTION_TAG = (0x0008, 0x1030)
SERIES_DESCRIPTION_TAG = (0x0008, 0x103E)
PHOTOMETRIC_INTERPRETATION_TAG = (0x0028, 0x0004)
REGION_PHYSICAL_DELTA_Y_SUBTAG = (0x0018, 0x602E)
REFERENCE_LINE_TAG = (0x0018, 0x6022)  # Doppler Reference Line

def forward_pass(inputs):
    logits = backbone(inputs)["out"] # torch.Size([1, 2, 480, 640])
    # Step 1: Apply sigmoid if needed
    if DO_SIGMOID:
        logits = torch.sigmoid(logits)
    # Step 2: Apply segmentation threshold if needed
    if SEGMENTATION_THRESHOLD is not None:
        logits[logits < SEGMENTATION_THRESHOLD] = 0.0
    return logits #predictions: weighted average of logits. logits: raw output from the model

print("Note: This script is for Doppler inference. Input DICOM height and width are 768 and 1024 respectively.")

if not args.file_path.endswith(".dcm"):
    raise ValueError("File path must be .dcm since we need Dicom Tag Information to calculate the Doppler Region and Velocity")
if not args.output_path.endswith(".jpg"):
    raise ValueError("Output path must be .jpg")

#MODEL LOADING
device = "cuda:0" #cpu / cuda
weights_path = f"./weights/Doppler_models/{args.model_weights}_weights.ckpt"

weights = torch.load(weights_path, map_location=device)
backbone = deeplabv3_resnet50(num_classes=1)  # 39,633,986 params
weights = {k.replace("m.", ""): v for k, v in weights.items()}
print(backbone.load_state_dict(weights)) #<All keys matched successfully>
backbone = backbone.to(device)
backbone.eval()

#LOAD DICOM IMAGE with DOPPLER REGION
DICOM_FILE = args.file_path 

ds = pydicom.dcmread(DICOM_FILE)
input_image = ds.pixel_array
if ds.PhotometricInterpretation == 'MONOCHROME2':
    input_image = np.stack((input_image,) * 3, axis=-1)
elif ds.PhotometricInterpretation == "YBR_FULL_422" and len(input_image.shape) == 3:
    input_image = convert_color_space(arr=input_image, current="YBR_FULL_422", desired="RGB")
    #Heuristic to mask EKG (green line) / select green > 200 and blue < 100 to Mask 0
    ecg_mask = np.logical_and(input_image[:, :, 1] > 200, input_image[:, :, 0] < 100)
    input_image[ecg_mask, :] = 0
elif ds.PhotometricInterpretation == "RGB": 
    ecg_mask = np.logical_and(input_image[:, :, 1] > 200, input_image[:, :, 0] < 100)
    input_image[ecg_mask, :] = 0
    # pass
else:
    print("Unsupported Photometric Interpretation")
    
#"Need Specific DICOM Region TAG for Doppler. It is typically saved in the DICOM file."
doppler_region = get_coordinates_from_dicom(ds)[0]
if REGION_PHYSICAL_DELTA_Y_SUBTAG in doppler_region:
    conversion_factor = abs(doppler_region[REGION_PHYSICAL_DELTA_Y_SUBTAG].value)
if REGION_Y0_SUBTAG in doppler_region: 
    y0 = doppler_region[REGION_Y0_SUBTAG].value
if REGION_Y1_SUBTAG in doppler_region: 
    y1 = doppler_region[REGION_Y1_SUBTAG].value
if REGION_X0_SUBTAG in doppler_region: 
    x0 = doppler_region[REGION_X0_SUBTAG].value
if REGION_X1_SUBTAG in doppler_region: 
    x1 = doppler_region[REGION_X1_SUBTAG].value
print("Doppler Region is located: X ranged from", x0, "to ", x1, ". Y ranged from ", y0, "to", y1)

#horizontal line means the line where the Doppler signal starts
horizontal_y = doppler_region[REFERENCE_LINE_TAG].value
print("In Doppler image, Doppler baseline is located at Y=", horizontal_y)
print("If ECG is flat, the Doppler baseline is located different position.")
#Basically, the region where the Doppler signal starts is 342-345. We truncate the image from 342 to 768. Make 426*1024.
input_dicom_doppler_area = input_image[342 :,:, :] 
doppler_area_tensor = torch.tensor(input_dicom_doppler_area)
doppler_area_tensor = doppler_area_tensor.permute(2, 0, 1).unsqueeze(0)
doppler_area_tensor = doppler_area_tensor.float() / 255.0
doppler_area_tensor = doppler_area_tensor.to(device) #torch.Size([1, 3, 426, 1024])

with torch.no_grad():
    logit = forward_pass(doppler_area_tensor)
    
    max_val = logit.max().item()
    min_val = logit.min().item()
    logits_normalized = (logit - min_val) / (max_val - min_val)
    logits_normalized = logits_normalized.squeeze().cpu().numpy()
    max_coords = np.unravel_index(np.argmax(logits_normalized), logits_normalized.shape)
    
    X = max_coords[1]  # Max Logit X value
    Y = max_coords[0] # Max Logit Y value in the Doppler Region
    predicted_x = int(X) 
    predicted_y = int(Y + y0) #add y0 to get the actual y value in the original image to map
    
    peak_velocity = conversion_factor * (predicted_y - (y0 + horizontal_y))
    peak_velocity = round(peak_velocity, 2)
    plt.figure(figsize=(4, 4))
    cv2.circle(input_image, (predicted_x, predicted_y), 10, (135, 206, 235), -1)
    plt.imshow(input_image, cmap='gray')
    plt.savefig(args.output_path)

print("Peak Velocity is", peak_velocity, "cm/s")
print("Output Image is saved at", args.output_path)

#SAMPLE SCRIPT

#python inference_Doppler_image.py --model_weights "avvmax"
#--file_path "./SAMPLE_DICOM/AVVMAX_SAMPLE_0.dcm"
#--output_path "./OUTPUT/JPG/AVVMAX_SAMPLE_GENERATED.jpg"

#python inference_Doppler_image.py --model_weights "trvmax" 
#--file_path "./SAMPLE_DICOM/TRVMAX_SAMPLE_0.dcm"
#--output_path "./OUTPUT/JPG/TRVMAX_SAMPLE_GENERATED.jpg"

#python inference_Doppler_image.py --model_weights "mrvmax"
#--file_path "./SAMPLE_DICOM/MRVMAX_SAMPLE_0.dcm"
#--output_path "./OUTPUT/JPG/MRVMAX_SAMPLE_GENERATED.jpg"

#python inference_Doppler_image.py --model_weights "lvotvmax"
#--file_path "./SAMPLE_DICOM/LVOTVMAX_SAMPLE_0.dcm"
#--output_path "./OUTPUT/JPG/LVOTVMAX_SAMPLE_GENERATED.jpg"

#python inference_Doppler_image.py --model_weights "latevel"
#--file_path "./SAMPLE_DICOM/LATEVEL_SAMPLE_0.dcm"
#--output_path "./OUTPUT/JPG/LATEVEL_SAMPLE_GENERATED.jpg"

#python inference_Doppler_image.py --model_weights "medevel"
#--file_path "./SAMPLE_DICOM/MEDEVEL_SAMPLE_0.dcm"
#--output_path "./OUTPUT/JPG/MEDEVEL_SAMPLE_GENERATED.jpg"

