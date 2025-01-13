import torch
import pandas as pd
from torchvision.models.segmentation import deeplabv3_resnet50
import cv2
import numpy as np
from argparse import ArgumentParser
from utils import segmentation_to_coordinates, get_coordinates_from_dicom, find_horizontal_line
import pydicom
from pydicom.pixel_data_handlers.util import  convert_color_space
import matplotlib.pyplot as plt
import os

"""
This script is for Doppler Velocity inference with a directory which containes many Doppler Dicoms . 
The script will output the Predicted Annotation and Velocity, and metadata (Like TRVMAX or AVVMAX).
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
parser.add_argument("--folders", type=str, required = True, help= "Path to the video file folders (both AVI and DICOM)")
parser.add_argument("--output_path_folders", type=str, help= "Output folders Defalut jpg")
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
ULTRASOUND_COLOR_DATA_PRESENT_TAG = (0x0028, 0x0014)

def forward_pass(inputs):
    logits = backbone(inputs)["out"] # torch.Size([1, 2, 480, 640])
    # Step 1: Apply sigmoid if needed
    if DO_SIGMOID:
        logits = torch.sigmoid(logits)
    # Step 2: Apply segmentation threshold if needed
    if SEGMENTATION_THRESHOLD is not None:
        logits[logits < SEGMENTATION_THRESHOLD] = 0.0
    return logits #predictions: weighted average of logits. logits: raw output from the model

#MODEL LOADING
device = "cuda:0" #cpu / cuda
weights_path = f"./weights/Doppler_models/{args.model_weights}_weights.ckpt"

weights = torch.load(weights_path, map_location=device)
backbone = deeplabv3_resnet50(num_classes=1)  # 39,633,986 params
weights = {k.replace("m.", ""): v for k, v in weights.items()}
print(backbone.load_state_dict(weights)) #<All keys matched successfully>
backbone = backbone.to(device)
backbone.eval()

#Saved metadata and results
results =[]

#LOAD DICOM IMAGE with DOPPLER REGION
DICOM_FILES = [os.path.join(args.folders, f) for f in os.listdir(args.folders) if f.endswith(".dcm")]

if args.output_path_folders:
    OUTPUT_FOLDERS = args.output_path_folders
    if not os.path.exists(OUTPUT_FOLDERS): 
        os.makedirs(OUTPUT_FOLDERS)

for DICOM_FILE in DICOM_FILES:
    try:
        ds = pydicom.dcmread(DICOM_FILE)
        input_image = ds.pixel_array
        
        if PHOTOMETRIC_INTERPRETATION_TAG in ds:
            PhotometricInterpretation = ds[PHOTOMETRIC_INTERPRETATION_TAG].value
        else:
            PhotometricInterpretation = np.nan
        
        if PhotometricInterpretation == 'MONOCHROME2':
            input_image = np.stack((input_image,) * 3, axis=-1)
        elif PhotometricInterpretation == "YBR_FULL_422" and len(input_image.shape) == 3:
            input_image = convert_color_space(arr=input_image, current="YBR_FULL_422", desired="RGB")
            ecg_mask = np.logical_and(input_image[:, :, 1] > 200, input_image[:, :, 0] < 100)
            input_image[ecg_mask, :] = 0
        elif PhotometricInterpretation == "RGB": 
            ecg_mask = np.logical_and(input_image[:, :, 1] > 200, input_image[:, :, 0] < 100)
            input_image[ecg_mask, :] = 0
            # pass
        else:
            print("Unsupported Photometric Interpretation")
            continue
        
        if ULTRASOUND_COLOR_DATA_PRESENT_TAG in ds: ultrasound_color_data_present = ds[ULTRASOUND_COLOR_DATA_PRESENT_TAG].value
        else: ultrasound_color_data_present = np.nan
        
        #"Need Specific DICOM Region TAG for Doppler. It is typically saved in the DICOM file."
        doppler_region = get_coordinates_from_dicom(ds)[0]
        if REGION_PHYSICAL_DELTA_Y_SUBTAG in doppler_region: conversion_factor = abs(doppler_region[REGION_PHYSICAL_DELTA_Y_SUBTAG].value)
        if REGION_Y0_SUBTAG in doppler_region:  y0 = doppler_region[REGION_Y0_SUBTAG].value
        if REGION_Y1_SUBTAG in doppler_region:  y1 = doppler_region[REGION_Y1_SUBTAG].value
        if REGION_X0_SUBTAG in doppler_region:  x0 = doppler_region[REGION_X0_SUBTAG].value
        if REGION_X1_SUBTAG in doppler_region:  x1 = doppler_region[REGION_X1_SUBTAG].value

        if y0 <340 or y0 > 350:
            print("Error: Doppler Region is not located in the correct position. Please check the DICOM file. Our developed model is trained with y0 Doppler Region located in 342-348.")
            continue
        
        #horizontal line means the line where the Doppler signal starts
        horizontal_y = find_horizontal_line(ds.pixel_array[y0:y1, :])
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
            Y = max_coords[0]  # Max Logit Y value in the Doppler Region
            predicted_x = int(X) 
            predicted_y = int(Y + y0) #add y0 to get the actual y value in the original image to map
            
            peak_velocity = conversion_factor * (predicted_y - (y0 + horizontal_y))
            peak_velocity = round(peak_velocity, 2)
            
            if args.output_path_folders:
                OUTPUT_FILES = os.path.join(OUTPUT_FOLDERS, os.path.basename(DICOM_FILE).replace(".dcm", ".jpg"))
                plt.figure(figsize=(4, 4))
                cv2.circle(input_image, (predicted_x, predicted_y), 10, (135, 206, 235), -1)
                plt.imshow(input_image, cmap='gray')
                plt.savefig(OUTPUT_FILES)
                
        results.append({
            "filename": DICOM_FILE,
            "measurement_name": args.model_weights,
            
            #Metadatas
            "PhotometricInterpretation": PhotometricInterpretation,
            "ultrasound_color_data_present": ultrasound_color_data_present,
            "PhysicalDeltaY": conversion_factor,
            "y0": y0,
            
            #Predicted Values
            "horizontal_line": horizontal_y,
            "predicted_x": predicted_x,
            "predicted_y": predicted_y,
            "peak_velocity": peak_velocity
        })
        
    except Exception as e:
        print(f"Error:{DICOM_FILE},  {e}")

metadata = pd.DataFrame(results)

if args.output_path_folders:
    metadata.to_csv(os.path.join(OUTPUT_FOLDERS, "metadata.csv"), index=False)
    

print(metadata.head())

#SAMPLE SCRIPT
#python inference_Doppler_image_folders.py  --model_weights avvmax  --folders ./SAMPLE_DICOM/AVV_FOLDERS  --output_path_folders ./OUTPUT/AVV
