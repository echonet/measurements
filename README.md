# EchoNet-Measurement -Automatic Annotation for Echocardiography-

**Project Overview**   

This project focuses on building a deep-learning model that can automatically annotate specific measurements from echocardiographic videos or images. Annotation of measurements on images is typically a time-consuming task for cardiologists and sonographer, especially with the growing volume of echocardiographic exams being performed. Automating this process has the potential to significantly improve efficiency in clinical workflows, and potentially reduce human-error, and allow high-throughput cardiovascular research using echocardiography. Our model aims to streamline this annotation process and guide parameter measurements with faster and more accurate manner.

Herein, we show gif file. Please refer `videos/EchoNet_Annotation.mp4` if you want see video file.
![Representative Videos for 2D Auto-Annotated Echocardiography](videos/EchoNet_Annotation.gif)

<img src="https://github.com/echonet/measurements/blob/main/image/Echocardiography_2D_AutoMeasurement_vs_Human.png" alt="Sample Echo Frame" width="600"/>

Presentation: Conference information will be updated.
Preprint: link will be added once preprint is released
Paper: link will be added once preprint is released

Key Benefits:
- Efficiency: Reducing the time needed for manual annotations.
- Scalability: Addressing the increasing number of echo exams by automating routine tasks.
- Accuracy: Providing consistent and reliable annotations of echocardiography parameters.


## Model Information:
1. Segmentation Model (main purpose of this project)
- Backbone: deeplabv3_resnet50(num_classes=1 or 2, depend on the task) 
- 2D Video Model: For 2D echocardiographic videos, our model is trained and operates on videos with a resolution of 480 and 640.
- Doppler Image Model: For Doppler images, our model is designed to process videos at a resolution of 426×1080.

    - **Why 426×1080?**
    Typically, DICOM images are stored with a resolution of 768×1024 (but I think it can depend on the vendor and hospital). However, the area where Doppler information is displayed usually starts around pixel 342-344 in the vertical axis (if original is stored 768*1024). Thus, we calculate the height for the Doppler image as 426 (by subtracting 342 from 768). This ensures that the model focuses only on the relevant Doppler information, optimizing both computational resources and model performance.
    Inside the code (`inference_Doppler_image.py`), we automatically selected suitable region for Doppler assessment.

2. PLAX quality model
Sometimes, the ultrasound video is blurred or does not have appropriate quality to evaluate. The quality depends on the sonographer as well as the patient's position and the position of the heart. In these cases, we believe that the output (that is, measurement or annotations) of deep learning model may not be accurate. Therefore, we created a model to evaluate the quality of PLAX Video, and evaluated the impact of video quality on the inferred value. This is not the main purpose of this paper, but we have created a model and uploaded it here. We used 112*112 videos as Input for this model. Sample videos have been published in the past (https://echonet.github.io/dynamic/index.html).


## Repository Overview
This repository contains the deep learning model used for automatic annotation for 2D echo videos and Doppler images, along with the necessary code and weights to load the model and perform inference on your echo videos. Users can easily deploy the model and run predictions using the provided codes.

## Contents
1. model/: Contains the trained deep learning model weights for echo video annotation.
2. inference.py: Script for loading the model and running inference on input echo videos.
3. utils/:



## How to Use:
1. Clone the repository:
```sh
git clone https://github.com/echonet/measurements.git
cd your-repository-name
```

2. Install dependencies:
```sh
pip install -r requirements.txt
```

3. Run inference:

3-1. Linear Measurement (like IVS, LAD, AORTA)
for model_weights, please refer to --model_weights choices in the script.
Either AVI and DICOM format works for input.
```sh
python inference_2D_image.py --model_weights VARIABLE_LIKE_IVS --file_path "YOUR_ECHO_AVI_OR_DICOM_FILEPATH (480 and 640 resolution)"  --output_path "YOUR_OUTPUT_PATH.avi"
```

You can create frame-to-frame linear mearurement prediction.

3-2. Doppler Annotation Prediction
for model_weights, please refer to --model_weights choices in the script.
Input should be DICOM image since specific DICOM Tag information is needed.
```sh
python inference_Doppler_image.py --model_weights "VARIABLE_LIKE_TRVMAX"  --file_path "YOUR_DICOM.dcm" --output_path "YOUR_OUTPUT_PATH.jpg"
```

<img src="https://github.com/echonet/measurements/blob/main/image/sample_output_31and32.png" alt="Sample Echo Frame" width="600"/>


## Gradio Demo Sample
**Important Notes on Demo Videos**
Demo videos that illustrate how the model works in real-world applications. At this time, demo App are not publicly released because the data used contains personal information and Gradio does not fully comply with HIPAA (Health Insurance Portability and Accountability Act) standards. To protect patient privacy and sensitive information, we have decided not to make the demo videos publicly available. Herein, we used de-identified Dicom files for this demo.

![Representative Videos for 2D Auto-Annotated Echocardiography](videos/Gradio_demo_TRVMAX.gif)
