import numpy as np
import cv2
import torch
from typing import Tuple, Union, List
import math
import pydicom
import scipy.signal
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from scipy.signal import savgol_filter

import numpy as np
from scipy.signal import find_peaks


ULTRASOUND_REGIONS_TAG = (0x0018, 0x6011)
REGION_X0_SUBTAG = (0x0018, 0x6018)  # left
REGION_Y0_SUBTAG = (0x0018, 0x601A)  # top
REGION_X1_SUBTAG = (0x0018, 0x601C)  # right
REGION_Y1_SUBTAG = (0x0018, 0x601E)  # bottom
STUDY_DESCRIPTION_TAG = (0x0008, 0x1030)
SERIES_DESCRIPTION_TAG = (0x0008, 0x103E)
PHOTOMETRIC_INTERPRETATION_TAG = (0x0028, 0x0004)
REGION_PHYSICAL_DELTA_X_SUBTAG = (0x0018, 0x602C)
REGION_PHYSICAL_DELTA_Y_SUBTAG = (0x0018, 0x602E)

def segmentation_to_coordinates(
    logits: torch.Tensor, normalize: bool = True, order="YX"
):
    """
    Expects logits with shape (..., n_points, h, w). 
    Returns (..., n_points, 2) coordinate pairs in YX ordering. uses weighted averaging to compute centroid of logits. 
    It is recommended to first sigmoid your logits if they are coming right out of a model. You may want to cast the return to int if you want to use it for indexing.
    """

    predictions_rows, predictions_cols = torch.meshgrid(
        torch.arange(logits.shape[-2], device=logits.device),
        torch.arange(logits.shape[-1], device=logits.device),
        indexing="ij",
    )  # (h, w)
    predictions_rows = predictions_rows * logits  # (..., h, w)
    predictions_cols = predictions_cols * logits  # (..., h, w)
    # weighted average of y values
    predictions_rows = predictions_rows.sum(dim=(-2, -1)) / (
        logits.sum(dim=(-2, -1)) + 1e-8
    )
    # weighted average of x values
    predictions_cols = predictions_cols.sum(dim=(-2, -1)) / (
        logits.sum(dim=(-2, -1)) + 1e-8
    )
    if normalize:
        predictions_rows = predictions_rows / (logits.shape[-2])
        predictions_cols = predictions_cols / (logits.shape[-1])
    if order == "YX":
        predictions = torch.stack([predictions_rows, predictions_cols], dim=-1)
    elif order == "XY":
        predictions = torch.stack([predictions_cols, predictions_rows], dim=-1)
    else:
        raise ValueError(f"Invalid order: {order}")
    return predictions


def get_coordinates_from_dicom(
    dicom: pydicom.Dataset,
) -> tuple[tuple[int, int, int, int], tuple]:
    """
    Looks through ultrasound region tags in the DICOM file. Usually, 
    there are two regions, and the doppler image is the lower one. 
    Returns the coordinates of this region's bounding box.
    """

    REGION_COORD_SUBTAGS = [
        REGION_X0_SUBTAG,
        REGION_Y0_SUBTAG,
        REGION_X1_SUBTAG,
        REGION_Y1_SUBTAG,
    ]

    if ULTRASOUND_REGIONS_TAG in dicom:
        all_regions = dicom[ULTRASOUND_REGIONS_TAG].value
        regions_with_coords = []
        for region in all_regions:
            region_coords = []
            for coord_subtag in REGION_COORD_SUBTAGS:
                if coord_subtag in region:
                    region_coords.append(region[coord_subtag].value)
                else:
                    region_coords.append(None)

            # Keep only regions that have a full set of 4 coordinates
            if all([c is not None for c in region_coords]):
                regions_with_coords.append((region, region_coords))

        # We sort regions by their y0 coordinate, as the Doppler region we want should be the lowest region
        regions_with_coords = list(
            sorted(regions_with_coords, key=lambda x: x[1][1], reverse=True)
        )

        return regions_with_coords[0]

    else:
        print("No ultrasound regions found in DICOM file.")
        return None, None
    
# def find_horizontal_line(
#     image: np.ndarray,
#     angle_threshold: float = np.pi / 180,
#     line_threshold: float = 100,
# ) -> int:
#     """
#     Horizontal line detection for Doppler images.
    
#     Uses Canny edge detection and the Hough Transform to find the most prominent horizontal line in the image. 
#     Returns the y-coordinate of this line.
#     """

#     if len(image.shape) == 2: #Already gray image
#         gray_image = image
#     else:
#         gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#     edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
#     lines = cv2.HoughLines(edges, 1, np.pi / 180, line_threshold)

#     if lines is not None:
#         for rho, theta in lines[:, 0]:
#             if (
#                 abs(theta - np.pi / 2) < angle_threshold
#                 or abs(theta - 3 * np.pi / 2) < angle_threshold
#             ):
#                 a = np.cos(theta)
#                 b = np.sin(theta)
#                 x0 = a * rho
#                 y0 = b * rho
#                 y = int(y0)
#                 return y
#     return None

#Convert YBR_FULL_422 to RGB
lut=np.load("/workspace/yuki/measurements_internal/measurements/ybr_to_rgb_lut.npy")
def ybr_to_rgb(x):
    return lut[x[..., 0], x[..., 1], x[..., 2]]

def calculate_weighted_centroids_with_meshgrid(logits):
    """
    #Write Explanation
    From Logit input, calculate the weighted centroids of the contours.
    If the number of objects is 3, the function returns the weighted centroids of the 3 objects.
    
    Args:
        logits: np.array of shape (H, W) with values in [0, 1]
    Returns:
        pair_centroids: list of tuples [(x1, y1), (x2, y2)]
        binary_image: np.array of shape (H, W) with values in {0, 255}
    
    """
    logits = (logits / logits.max()) * 255
    logits = logits.astype(np.uint8)
    _, binary_image = cv2.threshold(logits, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    centroids = []
    contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        mask = np.zeros_like(binary_image)
        cv2.drawContours(mask, [contour], -1, (255), thickness=cv2.FILLED)
        h, w = mask.shape
        y_coords, x_coords = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
        mask_indices = mask == 255
        filtered_logits = logits[mask_indices]
        x_coords_filtered = x_coords[mask_indices]
        y_coords_filtered = y_coords[mask_indices]
        weight_sum = filtered_logits.sum()
        if weight_sum != 0:
            cx = (x_coords_filtered * filtered_logits).sum() / weight_sum
            cy = (y_coords_filtered * filtered_logits).sum() / weight_sum
            centroids.append((int(cx), int(cy)))
    centroids = [(int(x), int(y)) for x, y in centroids]
    return centroids, binary_image

def apply_lpf(signal, cutoff):
    fft = np.fft.fft(signal)
    fft[cutoff+1:-cutoff] = 0
    filtered = np.real(np.fft.ifft(fft))
    return filtered

def bpm_to_frame_freq(window_len, fps, bpm):
    beats_per_second_max = bpm / 60
    beats_per_frame_max = beats_per_second_max / fps
    beats_per_video_max = beats_per_frame_max * window_len
    return int(np.ceil(beats_per_video_max))

def process_video_with_diameter(video_path, 
                                output_path, 
                                df, 
                                conversion_factor_X,
                                conversion_factor_Y,
                                ratio,
                                systole_diastole_analysis: bool = False):
    # Load video
    cap = cv2.VideoCapture(video_path)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    frames_array = np.array(frames)

    # Get coordinates from dataframe
    x1 = df["pred_x1"].values
    y1 = df["pred_y1"].values
    x2 = df["pred_x2"].values
    y2 = df["pred_y2"].values

    delta_x = abs(x2 - x1) *ratio 
    delta_y = abs(y2 - y1) *ratio
    diameters = np.sqrt((delta_x * conversion_factor_X)**2 + (delta_y * conversion_factor_Y)**2)

    # Smooth diameters
    fps = 30  # Example FPS, modify as necessary
    cutoff = bpm_to_frame_freq(window_len=len(diameters), fps=fps, bpm=140)
    smooth_diameters = apply_lpf(diameters, cutoff)

    # Create output video
    height, width = frames_array[0].shape[:2]
    plot_height = int(width * 0.3)  # Plot height is 30% of video width
    output_height = height + plot_height
    output_width = width

    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'XVID'), fps, (output_width, output_height))

    if systole_diastole_analysis:
        systolic_i, diastolic_i = get_systole_diastole(smooth_diameters, 
                                                    smoothing=False, 
                                                    kernel=[1, 2, 3, 2, 1], 
                                                    distance= 25) #25 is default value.
        
        if systolic_i is None or diastolic_i is None:
            return print("No systolic or diastolic peaks found in the diameter signal.")
        
        systolic_diamter = smooth_diameters[systolic_i]
        diastolic_diamter = smooth_diameters[diastolic_i]
        
        #pick first pair values
        INDEX = 0
        systolic_frame = systolic_i[INDEX] if isinstance(systolic_i, np.ndarray) else systolic_i
        diastolic_frame = diastolic_i[INDEX] if isinstance(diastolic_i, np.ndarray) else diastolic_i
        systolic_diamter = systolic_diamter[INDEX] if isinstance(systolic_diamter, np.ndarray) else systolic_diamter
        diastolic_diamter = diastolic_diamter[INDEX] if isinstance(diastolic_diamter, np.ndarray) else diastolic_diamter 
        LVEF_by_teicholz = calculate_lvef_teicholz(diastolic_diameter= diastolic_diamter,
                                                systolic_diameter= systolic_diamter)
        print(f"LVEF by teicholz methods was {LVEF_by_teicholz:.3f} %")
    
    for i, frame in enumerate(tqdm(frames_array)):    
        fig, ax = plt.subplots(figsize=(8, 2))
        ax.plot(diameters, label='Raw Diameter', alpha=0.6)
        ax.plot(smooth_diameters, label='Smoothed Diameter', color='red')
        ax.axvline(x=i, color='black', linestyle='--', alpha=0.5)
        ax.scatter(systolic_frame, systolic_diamter, color='blue',  marker='o')
        ax.scatter(diastolic_frame, diastolic_diamter, color='yellow',  marker='o') 
        ax.legend()
        ax.set_ylim(0, max(diameters) * 1.1)
        ax.set_xlabel('')
        ax.set_ylabel('Diameter')

        canvas = FigureCanvas(fig)
        canvas.draw()
        try:    
            plot_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except Exception as e:
            buf = canvas.buffer_rgba()
            plot_image_rgba = np.asarray(buf)
            plot_image = plot_image_rgba[:, :, :3] 
            
        plt.close(fig)
        plot_image = cv2.resize(plot_image, (width, plot_height))
        # Stack video frame and plot vertically
        combined_image = np.vstack((frame, plot_image))

        out.write(cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))

    out.release()
    
    df["diameter"] = diameters
    df["smooth_diameter"] = smooth_diameters
    df.to_csv(output_path.replace(".avi", ".csv"), index=False)

    print(f"Output Distance avi saved to {output_path}")
    
def get_systole_diastole(diameter: np.ndarray, 
                         smoothing: bool=False,
                         kernel=[1, 2, 3, 2, 1], 
                         distance: int=25) -> Tuple[np.ndarray]:
    """Finds heart phase from a representative signal. Signal must be maximum at end diastole and
    minimum at end systole.

    Args:
        diameter (np.ndarray): Signal representing heart phase. shape=(n,)
        kernel (list, optional): Smoothing kernel used before finding peaks. Defaults to [1, 2, 3, 2, 1].
        distance (int, optional): Minimum distance between peaks in find_peaks(). Defaults to 25.

    Returns:
        systole_i (np.ndarray): Indices of end systole. shape=(n_sys,)
        diastole_i (np.ndarray): Indices of end diastole. shape=(n_dia,)
    """

    # Smooth input
    if smoothing:
        kernel = np.array(kernel)
        kernel = kernel / kernel.sum()
        diameter = np.convolve(diameter, kernel, mode='same')

    # Find peaks
    diastole_i, _ = find_peaks(diameter, distance=distance)
    systole_i, _ = find_peaks(-diameter, distance=distance)

    # Ignore first/last index if possible
    if len(systole_i) != 0 and len(diastole_i) != 0:
        start_minmax = np.concatenate([diastole_i, systole_i]).min()
        end_minmax = np.concatenate([diastole_i, systole_i]).max()
        diastole_i = np.delete(diastole_i, np.where((diastole_i == start_minmax) | (diastole_i == end_minmax)))
        systole_i = np.delete(systole_i, np.where((systole_i == start_minmax) | (systole_i == end_minmax)))
    
    return systole_i, diastole_i

def calculate_lvef_teicholz(diastolic_diameter : float, 
                            systolic_diameter: float):
    """
    Calculates Left Ventricular Ejection Fraction (LVEF) using the Teicholz formula.

    Args:
        diastolic_diameter (float): Left ventricular end-diastolic diameter (LVIDd) in cm.
        systolic_diameter (float): Left ventricular end-systolic diameter (LVIDs) in cm.

    Returns:
        float: Left Ventricular Ejection Fraction (LVEF) as a percentage.
               Returns None if input values are invalid (e.g., negative, systolic > diastolic).
    """
    if systolic_diameter > diastolic_diameter:
        print("Error: Systolic diameter cannot be greater than diastolic diameter.")
        return None
    
    # Calculate LVEDV and ESV using Teicholz formula
    lvedv = (7.0 / (2.4 + diastolic_diameter)) * (diastolic_diameter ** 3)
    lvesv = (7.0 / (2.4 + systolic_diameter)) * (systolic_diameter ** 3)
    # Calculate Stroke Volume (SV)
    sv = lvedv - lvesv
    lvef = (sv / lvedv) * 100

    return lvef