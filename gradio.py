
################
###NOTE: We used the same code using Python Notebook (ipynb) to run gradio demo.
### Please Copy and Paste the code below to the Python Notebook.
################

import torch
from torchvision.models.segmentation import deeplabv3_resnet50
import sys
import gradio as gr
import pydicom
import numpy as np
from pydicom.pixel_data_handlers.util import apply_color_lut, convert_color_space
import cv2
import matplotlib.pyplot as plt
from utils import segmentation_to_coordinates, get_coordinates_from_dicom, find_horizontal_line

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


device = "cuda" #cpu / cuda
weights_path = "./weights/Doppler_models/trvmax_weights.ckpt"
weights = torch.load(weights_path)
backbone = deeplabv3_resnet50(num_classes=1)  # 39,633,986 params
weights = {k.replace("m.", ""): v for k, v in weights.items()}


print(backbone.load_state_dict(weights))
backbone = backbone.to(device)
backbone.eval()


def process_dicom(file):
    print("Processing dicom file", file.name)
    ds = pydicom.dcmread(file.name)
    input_image = ds.pixel_array
    if ds.PhotometricInterpretation == 'MONOCHROME2':input_image = np.stack((input_image,) * 3, axis=-1)
    elif ds.PhotometricInterpretation == "YBR_FULL_422" and len(input_image.shape) == 3:
        # img_name = f"{os.path.basename(dicom_file)}.jpg"
        input_image = convert_color_space(arr=input_image, current="YBR_FULL_422", desired="RGB")
    elif ds.PhotometricInterpretation == "RGB": pass
    else:print("Unsupported Photometric Interpretation")

    doppler_region = get_coordinates_from_dicom(ds)[0]
    if REGION_PHYSICAL_DELTA_Y_SUBTAG in doppler_region:conversion_factor = abs(doppler_region[REGION_PHYSICAL_DELTA_Y_SUBTAG].value)
    if REGION_Y0_SUBTAG in doppler_region: y0 = doppler_region[REGION_Y0_SUBTAG].value
    if REGION_Y1_SUBTAG in doppler_region: y1 = doppler_region[REGION_Y1_SUBTAG].value
    if REGION_X0_SUBTAG in doppler_region: x0 = doppler_region[REGION_X0_SUBTAG].value
    if REGION_X1_SUBTAG in doppler_region: x1 = doppler_region[REGION_X1_SUBTAG].value
     
    horizontal_y = find_horizontal_line(ds.pixel_array[y0:y1, :])
    input_dicom = ds.pixel_array[342 :,:, :]
    input_tensor = torch.tensor(input_dicom)
    input_tensor = input_tensor.permute(2, 0, 1).unsqueeze(0)
    input_tensor = input_tensor.float() / 255.0
    input_tensor = input_tensor.to(device) if device == "cuda" else input_tensor
    
    with torch.no_grad():
        model_output = forward_pass(input_tensor)
        X = model_output[0, 0, 0].item()  # Predicted X value
        Y = model_output[0, 0, 1].item()  # Predicted Y value in the Doppler Region
        predicted_x = int(X) 
        predicted_y = int(Y + y0) #add y0 to get the actual y value in the original image to map
        
        peak_velocity = conversion_factor * (predicted_y - (y0 + horizontal_y))
        peak_velocity = round(peak_velocity, 2)
        plt.figure(figsize=(6, 6))
        plt.title("TRVmax Predictin")
        plt.axis('off')
        cv2.circle(input_image, (predicted_x, predicted_y), 8, (135, 206, 235), -1)
        plt.imshow(input_image, cmap='gray')
        plt.savefig('doppler_waveform.png')
        
    return peak_velocity, "doppler_waveform.png"


iface = gr.Interface(
    fn=process_dicom, 
    inputs=gr.File(label="Upload DICOM file (Note: version 1.0.0)"), 
    outputs=[gr.Textbox(label="Predicted TRVMax Velocity (cm/s)"), gr.Image(label="Doppler Plotting on Dicom image")], 
    title="Gradio demo for CW Doppler Waveform Prediction"
)
iface.launch(share=True)