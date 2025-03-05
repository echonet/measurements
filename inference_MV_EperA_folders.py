import torch
import pandas as pd
import cv2
import numpy as np
import matplotlib.pyplot as plt
from argparse import ArgumentParser
import pydicom
from pydicom.pixel_data_handlers.util import  convert_color_space
from utils import get_coordinates_from_dicom, find_horizontal_line, calculate_weighted_centroids_with_meshgrid, ybr_to_rgb
from torchvision.models.segmentation import deeplabv3_resnet50
import os

"""
This script processes a folder of DICOM files for MV E/A Doppler inference.
It outputs annotated images and saves metadata (e.g., E/A ratios, velocities) to a CSV file.
"""

parser = ArgumentParser()
parser.add_argument("--folders", type=str, required = True, help= "Path to the video file (both AVI and DICOM)", default=None)
parser.add_argument("--output_path_folders", type=str, help= "Output. Defalut should be AVI", default=None)
args = parser.parse_args()

#Configuration
SEGMENTATION_THRESHOLD = 0.0
DO_SIGMOID = True

REGION_X0_SUBTAG = (0x0018, 0x6018)  # left
REGION_Y0_SUBTAG = (0x0018, 0x601A)  # top
REGION_X1_SUBTAG = (0x0018, 0x601C)  # right
REGION_Y1_SUBTAG = (0x0018, 0x601E)  # bottom
PHOTOMETRIC_INTERPRETATION_TAG = (0x0028, 0x0004)
REGION_PHYSICAL_DELTA_Y_SUBTAG = (0x0018, 0x602E)
ULTRASOUND_COLOR_DATA_PRESENT_TAG = (0x0028, 0x0014)

def forward_pass(inputs):
    logits = backbone(inputs)["out"]
    
    if DO_SIGMOID:
        logits = torch.sigmoid(logits)
    if SEGMENTATION_THRESHOLD is not None:
        logits[logits < SEGMENTATION_THRESHOLD] = 0.0
    
    logits_numpy = logits.squeeze().detach().cpu().numpy()
    logits_first = logits_numpy[1, :, :] #First channel is in 1. Sorry for the confusion.
    max_val_first, min_val_first = logits_first.max(), logits_first.min()
    logits_first = (logits_first - min_val_first) / (max_val_first - min_val_first)
    _, _, _, max_loc_first_channel = cv2.minMaxLoc(logits_first)
    
    logits_second = logits_numpy[0, :, :] #Second channel is in 0.
    max_val_second, min_val_second = logits_second.max(), logits_second.min()
    logits_second = (logits_second - min_val_second) / (max_val_second - min_val_second)
    _, _, _, max_loc_second_channel = cv2.minMaxLoc(logits_second)
    
    combine_logit = logits_first + logits_second
    _, _, _, max_loc_combine = cv2.minMaxLoc(combine_logit)
    
    #Check max_loc_combine is come from which channel
    #1. calculate difference between max_loc_combine and max_loc_first_channel / and max_loc_second_channel
    diff_maxloc_combine_first = np.sqrt((max_loc_combine[0] - max_loc_first_channel[0])**2 + (max_loc_combine[1] - max_loc_first_channel[1])**2)
    diff_maxloc_combine_second = np.sqrt((max_loc_combine[0] - max_loc_second_channel[0])**2 + (max_loc_combine[1] - max_loc_second_channel[1])**2)

    centroids_first, _ = calculate_weighted_centroids_with_meshgrid(logits_first)
    centroids_second, _ = calculate_weighted_centroids_with_meshgrid(logits_second)
    centroids, _ = calculate_weighted_centroids_with_meshgrid(combine_logit)
    
    #2. Pick the closest centeroid point to max_loc_combine
    #Among many points centeroids (basically, 2-4 points), get the closest point to max_loc_combine
    #Dictionary for distance between each centroid and maxlogits coordinate
    distance_centroid_btw_maxlogits = {}
    for centroid in centroids:
        distance = np.sqrt((max_loc_combine[0] - centroid[0])**2 + (max_loc_combine[1] - centroid[1])**2)
        distance_centroid_btw_maxlogits[centroid] = distance
    #Get the coordinates with minimum value of distance between maxlogits and centroid
    try:
        min_distance_coord  = min(distance_centroid_btw_maxlogits, key=distance_centroid_btw_maxlogits.get)
    except:
        raise ValueError("Error: min_distance_coord is not found, due to low prediction score. Select Good quality MVPeak Doppler data")
    
    #3. Calculate distance between min_distance_coord and other centroids
    distance_btw_centroids = {}
    if diff_maxloc_combine_second - diff_maxloc_combine_first > 15:
        # print("max_loc_combine is from the first channel. Pick Pair from the second channel")
        for centroid in centroids_second:
            distance = np.sqrt((min_distance_coord[0] - centroid[0])**2 + (min_distance_coord[1] - centroid[1])**2)
            distance_btw_centroids[centroid] = distance
            
    elif diff_maxloc_combine_first - diff_maxloc_combine_second > 15:
        # print("max_loc_combine is from the second channel. Pick Pair from the first channel")
        for centroid in centroids_first:
            distance = np.sqrt((min_distance_coord[0] - centroid[0])**2 + (min_distance_coord[1] - centroid[1])**2)
            distance_btw_centroids[centroid] = distance
    else:
        # print("Other. Get the coordinates from combined logit channel")
        distance_btw_centroids = {}
        for centroid in centroids:
            distance = np.sqrt((min_distance_coord[0] - centroid[0])**2 + (min_distance_coord[1] - centroid[1])**2)
            distance_btw_centroids[centroid] = distance
    
    #4. Get the closest Pair of Coordinates
    non_zero_distance_btw_centroids = {k:v for k, v in distance_btw_centroids.items() if v > 15}
    try:
        min_distance_paired_coord = min(non_zero_distance_btw_centroids, key=non_zero_distance_btw_centroids.get)
        pair_coords = [min_distance_coord, min_distance_paired_coord] 
    except:
        raise ValueError("Error: min_distance_coord is not found, due to low prediction score. Select Good quality MVPeak Doppler data")
    
    point_x1, point_y1= pair_coords[0][0], pair_coords[0][1] 
    point_x2, point_y2 = pair_coords[1][0], pair_coords[1][1]
            
    if point_x1 > point_x2:
        point_x1, point_y1, point_x2, point_y2 = point_x2, point_y2, point_x1, point_y1

    distance_x1_x2 = abs(point_x1 - point_x2)
    if distance_x1_x2 > 300:
        raise ValueError("Error: The distance between two points is too far. Please select the good quality Doppler data.")
            
    return point_x1, point_y1, point_x2, point_y2

print("Note: This script is for MV Peak E or E/A inference. Input DICOM height and width are 768/1024 respectively.")

#MODEL LOADING
device = "cuda:0" #cpu / cuda
weights_path = "./weights/Doppler_models/mvpeak_2c_weights.ckpt"
weights = torch.load(weights_path, map_location=device)
backbone = deeplabv3_resnet50(num_classes=2) 
weights = {k.replace("m.", ""): v for k, v in weights.items()}
print(backbone.load_state_dict(weights)) #says <All keys matched successfully>
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
        
        if len(input_image.shape) == 2:
            height, width = input_image.shape
        else:
            height, width, _ = input_image.shape
        
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
        
        if ULTRASOUND_COLOR_DATA_PRESENT_TAG in ds: 
            ultrasound_color_data_present = ds[ULTRASOUND_COLOR_DATA_PRESENT_TAG].value
        else: 
            ultrasound_color_data_present = np.nan
        
        #"Need Specific DICOM Region TAG for Doppler. It is typically saved in the DICOM file."
        doppler_region = get_coordinates_from_dicom(ds)[0]
        if REGION_PHYSICAL_DELTA_Y_SUBTAG in doppler_region: PhysicalDeltaY_doppler = abs(doppler_region[REGION_PHYSICAL_DELTA_Y_SUBTAG].value)
        if REGION_Y0_SUBTAG in doppler_region:  y0 = doppler_region[REGION_Y0_SUBTAG].value
        if REGION_Y1_SUBTAG in doppler_region:  y1 = doppler_region[REGION_Y1_SUBTAG].value
        if REGION_X0_SUBTAG in doppler_region:  x0 = doppler_region[REGION_X0_SUBTAG].value
        if REGION_X1_SUBTAG in doppler_region:  x1 = doppler_region[REGION_X1_SUBTAG].value

        if y0 <340 or y0 > 350:
            raise ValueError("Error: Doppler Region is not located in the correct position. Please check the DICOM file. Our developed model is trained with y0 Doppler Region located in 342-348.")

        #horizontal line means the line where the Doppler signal starts
        horizontal_y = find_horizontal_line(input_image[y0:y1, :])
        #Basically, the region where the Doppler signal starts is 342-345. We truncate the image from 342 to 768. Make 426*1024.
        input_dicom_doppler_area = input_image[342 :,:, :] 

        doppler_area_tensor = torch.tensor(input_dicom_doppler_area)
        doppler_area_tensor = doppler_area_tensor.permute(2, 0, 1).unsqueeze(0)
        doppler_area_tensor = doppler_area_tensor.float() / 255.0
        doppler_area_tensor = doppler_area_tensor.to(device)

        with torch.no_grad():
            point_x1, point_y1, point_x2, point_y2 = forward_pass(doppler_area_tensor)
            
            Inference_E_Vel = round(abs((point_y1 - horizontal_y) * PhysicalDeltaY_doppler),4)
            Inference_A_Vel = round(abs((point_y2 - horizontal_y) * PhysicalDeltaY_doppler),4)
            Inference_EperA = round(Inference_E_Vel / Inference_A_Vel, 3)
            
            if args.output_path_folders:
                OUTPUT_FILES = os.path.join(OUTPUT_FOLDERS, os.path.basename(DICOM_FILE).replace(".dcm", ".jpg"))
                plt.figure(figsize=(4, 4))
                plt.imshow(input_image, cmap='gray')
                plt.scatter(point_x1, point_y1 + y0, color='red', s=20)
                plt.scatter(point_x2, point_y2 + y0, color='blue', s=20)
                plt.savefig(OUTPUT_FILES)
        
        results.append({
            "filename": DICOM_FILE,
            "measurement_name": 'MV_Peak',
            
            #Metadatas
            "PhotometricInterpretation": PhotometricInterpretation,
            "ultrasound_color_data_present": ultrasound_color_data_present,
            "PhysicalDeltaY":PhysicalDeltaY_doppler,
            "y0": y0,
            
            #Predicted Values
            "horizontal_line": horizontal_y,
            "predicted_xe": point_x1,
            "predicted_ye": point_y1,
            "predicted_xa": point_x2,
            "predicted_ya": point_y2,
                
            "Inference_E_Vel": Inference_E_Vel,
            "Inference_A_Vel": Inference_A_Vel,
            "Inference_EperA": Inference_EperA,
            "height": height,
            "width": width
        })
    except Exception as e:
        print(f"Error: {DICOM_FILE}, {e}")

metadata = pd.DataFrame(results)
if args.output_path_folders:
    metadata.to_csv(os.path.join(OUTPUT_FOLDERS, "metadata_mvpeak.csv"), index=False)
    

print(metadata.head())

#SAMPLE SCRIPT
#python inference_MV_EperA_folders.py  --folders ./SAMPLE_DICOM/MVPEAK_FOLDERS  --output_path_folders ./OUTPUT/MVPEAK