from pathlib import Path
import torch
import pandas as pd
from torchvision.models.segmentation import deeplabv3_resnet50
import cv2
import numpy as np
from argparse import ArgumentParser
from utils import segmentation_to_coordinates, process_video_with_diameter, get_coordinates_from_dicom
import pydicom
from pydicom.pixel_data_handlers.util import  convert_color_space
import os
from tqdm import tqdm

"""
This script is for 2D frame-to-frame inference  with a directory which containes many Doppler Dicoms . 
The input is video files (AVI or DICOM) and the output is video files (AVI) with the predicted frame-to-frame annotation and metadata. 

Disclaimer: Please include the appropriate view corresponding to the measurement item. 
(For example, LVID is typically measured in the PLAX view and not in A4C or PSAX. 
Additionally, avoid using an excessively zoomed-out PLAX view; instead, include a standard one. The same applies to other measurement items.)
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
parser.add_argument("--folders", type=str, required = True, help= "Path to the video file folders (both AVI and DICOM)")
parser.add_argument("--output_path_folders", type=str, help= "Output folders Defalut AVI and metadata")
args = parser.parse_args()

#Configuration
SEGMENTATION_THRESHOLD = 0.0
DO_SIGMOID = True

#Dicom TAG
REGION_PHYSICAL_DELTA_X_SUBTAG = (0x0018, 0x602C)
REGION_PHYSICAL_DELTA_Y_SUBTAG = (0x0018, 0x602E)
PHOTOMETRIC_INTERPRETATION_TAG = (0x0028, 0x0004)
ULTRASOUND_COLOR_DATA_PRESENT_TAG = (0x0028, 0x0014)


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
print("Please ensure that all file extensions in the folder are unified to either .dcm or .avi. Do not combine.")
print("Note: This script is for 2D frame-to-frame inference.\nOur model used the video with height of 480 and width of 640, respectively.")


def check_extensions_uniformity(folder_path, allowed_extensions = ['.dcm', '.avi']):
    """
    Checks if all files in the folder have extensions that are uniformly one of the allowed extensions.
    """
    extensions_found = set()
    for _, _, files in os.walk(folder_path):
        for file in files:
            _, ext = os.path.splitext(file)
            if ext.lower() in allowed_extensions:
                extensions_found.add(ext.lower())
            else:
                print(f"Invalid file extension: {file}")
                return False
    if len(extensions_found) == 1:
        print(f"All files have the same valid extension")
        return True
    else:
        print("Files have mixed extensions or invalid extensions:", extensions_found)
        return False

check_extensions_uniformity(folder_path = args.folders, allowed_extensions = ['.dcm', '.avi'])

# MODEL LOADING
device = "cuda:0" #cpu / cuda
weights_path = f"./weights/2D_models/{args.model_weights}_weights.ckpt"
weights = torch.load(weights_path, map_location=device)
backbone = deeplabv3_resnet50(num_classes=2)  # 39,633,986 params / num_classes should be 2
weights = {k.replace("m.", ""): v for k, v in weights.items()}
print(backbone.load_state_dict(weights)) #says <All keys matched successfully>
backbone = backbone.to(device)
backbone.eval()


if args.output_path_folders:
    OUTPUT_FOLDERS = args.output_path_folders
    if not os.path.exists(OUTPUT_FOLDERS): 
        os.makedirs(OUTPUT_FOLDERS)


#Saved metadata and results
results_all_files =[]

#LOAD DICOM IMAGE with DOPPLER REGION
VIDEO_FILES = [os.path.join(args.folders, f) for f in os.listdir(args.folders) if f.endswith(".dcm") or f.endswith(".avi")]

#using tqdm
for VIDEO_FILE in tqdm(VIDEO_FILES):
    try:
        results_one_file =[]
        frames = []
        
        #Version AVI, LOAD VIDEO (AVI).
        if VIDEO_FILE.endswith(".avi"):
            video = cv2.VideoCapture(VIDEO_FILE)
            while True:
                ret, frame = video.read()
                if not ret:
                    break
                frames.append(frame)
            frames = np.array(frames)
            
            PhotometricInterpretation = None
            ultrasound_color_data_present = None
            diameters = None

        #Version DICOM, LOAD VIDEO (Dicom).
        elif VIDEO_FILE.endswith(".dcm"):
            ds = pydicom.dcmread(VIDEO_FILE)
            
            if ULTRASOUND_COLOR_DATA_PRESENT_TAG in ds: 
                ultrasound_color_data_present = ds[ULTRASOUND_COLOR_DATA_PRESENT_TAG].value
            else: 
                ultrasound_color_data_present = np.nan
                
            if PHOTOMETRIC_INTERPRETATION_TAG in ds:
                PhotometricInterpretation = ds[PHOTOMETRIC_INTERPRETATION_TAG].value
            else:
                PhotometricInterpretation = None
            
            input_dicom = ds.pixel_array #Frames shape (Frame, Height, Width, Channel)
            height, width = input_dicom.shape[1], input_dicom.shape[2]
            
            doppler_region = get_coordinates_from_dicom(ds)[0]
            if REGION_PHYSICAL_DELTA_X_SUBTAG in doppler_region:
                conversion_factor_X = abs(doppler_region[REGION_PHYSICAL_DELTA_X_SUBTAG].value)
            if REGION_PHYSICAL_DELTA_Y_SUBTAG in doppler_region:
                conversion_factor_Y = abs(doppler_region[REGION_PHYSICAL_DELTA_Y_SUBTAG].value)
            
            ratio_height = height / 480 
            ratio_width = width / 640
            
            if ratio_height != ratio_width:
                ValueError("Height and Width ratio should be same, Our model used 3:4 aspect videos.")

            for frame in input_dicom:
                if ds.PhotometricInterpretation == "YBR_FULL_422":
                    frame = convert_color_space(arr=frame, current="YBR_FULL_422", desired="RGB")
                    # frame =cv2.cvtColor(frame, cv2.COLOR_YUV2RGB)
                resized_frame = cv2.resize(frame, (640, 480)) 
                frames.append(resized_frame)

        #Get Prediction using loaded model
        input_tensor = torch.tensor(frames)
        input_tensor = input_tensor.float() / 255.0
        input_tensor = input_tensor.to(device)
        input_tensor = input_tensor.permute(0, 3, 1, 2)  # (F, C, H, W)

        #In predictions, each frame-level prediction will be saved.
        predictions = []
        for i in range(input_tensor.shape[0]): #[0] means number of frames.
            batch = {"inputs": input_tensor[i].unsqueeze(0)} # torch.Size([1, 3, 480, 640])
            with torch.no_grad(): 
                model_output = forward_pass(batch["inputs"])
            predictions.append(model_output)
        predictions = torch.cat(predictions, dim=0)
        predictions = predictions.cpu().numpy()

        #Make Output Video
        output_video_path = os.path.join(OUTPUT_FOLDERS, os.path.basename(VIDEO_FILE) + "_generated.avi")
        # output_video_path = os.path.join(OUTPUT_FOLDERS, os.path.basename(VIDEO_FILE).replace(".dcm", "_generated.avi"))
        out_avi = cv2.VideoWriter(
                    output_video_path,
                    cv2.VideoWriter_fourcc(*"XVID"),
                    30, #FPS defult
                    (batch["inputs"].shape[-1], batch["inputs"].shape[-2]),  # Width, Height
                )
        
        for i, (frame, prediction) in enumerate(zip(input_tensor, predictions)):
            frame = frame.permute(1, 2, 0).cpu().numpy()
            frame = np.clip(frame, 0, 1)
            frame = (frame * 255).astype(np.uint8)
            
            #Plot points (circle)
            for point, color in zip(prediction, [(235, 206, 135), (235, 206, 135)]):
                point_0, point_1 = point[0], point[1]
                cv2.circle(frame, (int(point_0), int(point_1)), 5, color, -1)
            
            x1, y1, x2, y2 = prediction[0][0], prediction[0][1], prediction[1][0], prediction[1][1]
            #Plot line
            cv2.line(frame, 
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    (235, 206, 135), 
                    2)
            delta_x = abs(x2 - x1) * ratio_height
            delta_y = abs(y2 - y1) * ratio_height
            
            if conversion_factor_X is not None and conversion_factor_Y is not None:
                diameters = np.sqrt((delta_x * conversion_factor_X)**2 + (delta_y * conversion_factor_Y)**2)
    
            results_one_file.append({
                "filename": VIDEO_FILE,
                "frame_number":i,
                "measurement_name": args.model_weights,
                
                "PhotometricInterpretation": PhotometricInterpretation,
                "ultrasound_color_data_present": ultrasound_color_data_present,
            
                "pred_x1": x1,
                "pred_y1": y1,
                "pred_x2": x2,
                "pred_y2": y2,
                "coordinates": f"{x1}:{x2}:{y1}:{y2}",
                "predicted_diameter": diameters
                #image_path   m_name  coordinates   type   video_path frame_number    mse  image_shape file_uid    mrn
                }) 
            
            out_avi.write(frame)
            
            #make dataframe from results_one_file
            df_results_one_file = pd.DataFrame(results_one_file)
            
        out_avi.release()

        process_video_with_diameter(video_path = output_video_path, 
                                output_path = output_video_path.replace(".avi", "_distance.avi"),
                                conversion_factor_X = conversion_factor_X,
                                conversion_factor_Y = conversion_factor_Y,
                                df = df_results_one_file,
                                ratio = ratio_height)
        #if you want video with predicted diameter plot, please refere process_video_with_diameter function.

    except Exception as e:
        print(f"Error:{VIDEO_FILE},  {e}")
        
    results_all_files.extend(results_one_file)

metadata = pd.DataFrame(results_all_files)

if args.output_path_folders:
    metadata.to_csv(os.path.join(OUTPUT_FOLDERS, f"metadata_{args.model_weights}.csv"), index=False)
    

print(metadata.head())

print("Completed. Please check the output folder for the generated videos and metadata.")        


#SAMPLE SCRIPT

#python inference_2D_image_folders.py --model_weights "ivs" 
#--folders "./SAMPLE_DICOM/IVS_FOLDERS" 
#--output_path_folders "./OUTPUT/IVS"

#python inference_2D_image_folders.py --model_weights "pa" 
#--folders "/workspace/yuki/measurements_internal/end_to_end_inference/mpa_videos" 
#--output_path_folders "./OUTPUT/PA_SFD"