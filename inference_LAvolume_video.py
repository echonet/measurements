import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import cv2
from tqdm import tqdm
import time
from argparse import ArgumentParser

import torch
from torchvision.models.segmentation import deeplabv3_resnet50

# For DICOM support
import pydicom
from pydicom.pixel_data_handlers.util import convert_color_space

# LA Length calculation helper functions
from scipy.interpolate import splev, splprep
from skimage import filters, measure
from sklearn.linear_model import LinearRegression
import skimage.draw
from typing import Tuple

#################################################################################
# Helper Functions

# David Choi, Note:
# Most of these functions are copied from https://github.com/echonet/diastology/blob/main/utils/model_utils.py
# Sometimes modified for our use case.
#################################################################################

def check_and_shift_edge(points, p1, p2):
    idx_p1 = np.where(np.all(points == p1, axis=1))[0][0]
    idx_p2 = np.where(np.all(points == p2, axis=1))[0][0]
    min_idx = np.min([idx_p1, idx_p2])
    max_idx = np.max([idx_p1, idx_p2])
    if (max_idx > 1) & (min_idx == 0):
        points_new = points
    else:
        points_new = np.roll(points, -(min_idx + 1), axis=0)
    return points_new

def find_mitral_plane(points):
    distances = np.sqrt(np.sum(np.diff(points, axis=0, append=points[:1]) ** 2, axis=1))
    mitral_idx = np.argmax(distances)
    P1, P2 = points[mitral_idx], points[(mitral_idx + 1) % len(points)]
    return P1, P2, mitral_idx, distances[mitral_idx]

def smooth_polygon(points, smoothness=0.5, num_points=500):
    if len(points) < 4: return points
    try:
        tck, _ = splprep([points[:, 0], points[:, 1]], s=smoothness)
        u_new = np.linspace(0, 1, num_points)
        x_new, y_new = splev(u_new, tck)
        return np.column_stack([x_new, y_new])
    except:
        return points

def rasterize_polygon(smooth_points, img_shape):
    r, c = skimage.draw.polygon(
        np.rint(smooth_points[:, 1]).astype(np.int_),
        np.rint(smooth_points[:, 0]).astype(np.int_),
        img_shape,
    )
    mask = np.zeros(img_shape, np.float32)
    r = np.clip(r, 0, img_shape[0]-1)
    c = np.clip(c, 0, img_shape[1]-1)
    mask[r, c] = 1
    return mask

def vector_to_bitmap(example, mask_size, smooth=True):
    points_pure = example
    P1, P2, mitral_idx, mitral_plane_distance = find_mitral_plane(points_pure)
    points = check_and_shift_edge(points_pure, P1, P2)
    if smooth:
        smooth_points = smooth_polygon(points, num_points=500)
    else:
        smooth_points = points
    
    index_of_bottom = np.argmax(smooth_points[:, 1]) 
    point_bottom = smooth_points[index_of_bottom]
    
    mid_point = (P1 + P2) / 2
    vertical_distance = np.linalg.norm(mid_point - point_bottom)
    
    mask = rasterize_polygon(smooth_points, mask_size)
    return P1, P2, point_bottom, vertical_distance, mask

def find_contour(mask):
    if np.sum(mask) == 0:
        return None, None

    smooth_binary_array = filters.gaussian(mask, sigma=1)
    median_smoothed_image = smooth_binary_array

    max_val = np.max(median_smoothed_image)
    search_level = 0.3
    if max_val < 0.35:
        search_level = max_val * 0.5 

    contours = measure.find_contours(median_smoothed_image, level=search_level)
    if not contours:
        return None, None

    chosen_contour = max(contours, key=len)
    contour_points = chosen_contour.tolist()
    contour_points_swapped = [(x, y) for y, x in contour_points]
    contour_array = np.array(contour_points_swapped)
    _, unique_indices = np.unique(contour_array, axis=0, return_index=True)
    point_mask = contour_array[np.sort(unique_indices)]

    return median_smoothed_image, point_mask

def P2_LinearRegression_method(P1_mask, point_mask):
    forced_point = P1_mask.ravel()
    forced_index = np.argmin(np.abs(np.sum(point_mask - forced_point, axis=1)))
    filter_range = 10
    left_selected = point_mask[max(0, forced_index - filter_range) : forced_index]
    right_selected = point_mask[forced_index + 1 : forced_index + 1 + filter_range]
    
    if len(left_selected) == 0 or len(right_selected) == 0: 
        return None, None, None

    y_diff_left = np.sum(np.abs(np.diff(left_selected[:, 1], append=left_selected[0, 1])))
    y_diff_right = np.sum(np.abs(np.diff(right_selected[:, 1], append=right_selected[0, 1])))
    selected_points = left_selected if y_diff_left <= y_diff_right else right_selected

    reg = LinearRegression().fit(selected_points[:, 0].reshape(-1, 1), selected_points[:, 1])
    predicted_y = reg.predict(point_mask[:, 0].reshape(-1, 1))
    differences = np.abs(point_mask[:, 1] - predicted_y)
    threshold = np.percentile(differences, 21)
    strongly_correlated_points = point_mask[differences <= threshold]
    
    if len(strongly_correlated_points) == 0: 
        return reg, point_mask[0], point_mask[-1]

    P2_mask = strongly_correlated_points[np.argmin(strongly_correlated_points[:, 0])]
    P1_mask_new = strongly_correlated_points[np.argmax(strongly_correlated_points[:, 0])]
    return reg, P2_mask, P1_mask_new

def calculate_la_length_from_mask(mask):
    try:
        if np.sum(mask) < 100: 
            return None, None, None
        
        _, point_mask = find_contour(mask)
        if point_mask is None: 
            return None, None, None

        min_y, max_y = np.min(point_mask[:, 1]), np.max(point_mask[:, 1])
        P1_mask = point_mask[point_mask[:, 1] == min_y]
        
        _, P2_mask, P1_mask_new = P2_LinearRegression_method(P1_mask[0], point_mask)
        if P2_mask is None: 
            return None, None, None

        idx_p1 = np.where(np.all(point_mask == P1_mask_new, axis=1))[0][0]
        idx_p2 = np.where(np.all(point_mask == P2_mask, axis=1))[0][0]
        start, end = sorted([idx_p2, idx_p1])
        filtered_points = np.delete(point_mask, np.s_[start + 2 : end], axis=0)

        P1, P2, point_bottom, vertical_distance, _ = vector_to_bitmap(filtered_points, mask_size=mask.shape, smooth=True)
        mid_point = (P1 + P2) / 2
        
        return vertical_distance, mid_point, point_bottom

    except Exception as e:
        return None, None, None

def ybr_to_rgb(frame):
    """Convert YBR_FULL_422 to RGB"""
    ycbcr = frame.astype(np.float32)
    rgb = np.zeros_like(ycbcr)
    rgb[:,:,0] = ycbcr[:,:,0] + 1.402 * (ycbcr[:,:,2] - 128)
    rgb[:,:,1] = ycbcr[:,:,0] - 0.344136 * (ycbcr[:,:,1] - 128) - 0.714136 * (ycbcr[:,:,2] - 128)
    rgb[:,:,2] = ycbcr[:,:,0] + 1.772 * (ycbcr[:,:,1] - 128)
    return np.clip(rgb, 0, 255).astype(np.uint8)

def get_coordinates_from_dicom(ds):
    """Extract region sequence from DICOM"""
    if hasattr(ds, 'SequenceOfUltrasoundRegions'):
        return ds.SequenceOfUltrasoundRegions
    return []


# ===============================================
# LA Volume Curve Analysis Functions made for this use case.
# ===============================================

def get_la_max_volume_frame(volumes: np.ndarray, 
                            smoothing: bool = True,
                            kernel: list = [1, 2, 3, 2, 1], 
                            distance: int = 15) -> Tuple[np.ndarray, np.ndarray]:
    """
    Finds the frame with maximum LA volume (end-systolic phase for LA).
    LA volume is maximum at ventricular end-systole.

    Args:
        volumes (np.ndarray): LA volume signal over frames. shape=(n_frames,)
        smoothing (bool): Whether to smooth the signal before peak detection
        kernel (list): Smoothing kernel used before finding peaks
        distance (int): Minimum distance between peaks in find_peaks()

    Returns:
        max_volume_indices (np.ndarray): Indices of maximum volume frames (end-systole)
        min_volume_indices (np.ndarray): Indices of minimum volume frames (end-diastole)
    """
    from scipy.signal import find_peaks
    
    # Smooth input if requested
    if smoothing:
        kernel_array = np.array(kernel)
        kernel_array = kernel_array / kernel_array.sum()
        volumes_smoothed = np.convolve(volumes, kernel_array, mode='same')
    else:
        volumes_smoothed = volumes

    # Find peaks (maxima = end-systole for LA, minima = end-diastole for LA)
    max_volume_i, _ = find_peaks(volumes_smoothed, distance=distance)
    min_volume_i, _ = find_peaks(-volumes_smoothed, distance=distance)

    # Ignore first/last index if possible
    if len(max_volume_i) != 0 and len(min_volume_i) != 0:
        start_minmax = np.concatenate([max_volume_i, min_volume_i]).min()
        end_minmax = np.concatenate([max_volume_i, min_volume_i]).max()
        max_volume_i = np.delete(max_volume_i, np.where((max_volume_i == start_minmax) | (max_volume_i == end_minmax)))
        min_volume_i = np.delete(min_volume_i, np.where((min_volume_i == start_minmax) | (min_volume_i == end_minmax)))
    
    return max_volume_i, min_volume_i

def smooth_volume_curve(volumes: np.ndarray, cutoff: int = None, fps: int = 30, bpm: int = 100) -> np.ndarray:
    """
    Apply low-pass filter to smooth LA volume curve.
    
    Args:
        volumes: LA volume array
        cutoff: Manual cutoff frequency (if None, calculated from bpm)
        fps: Frame rate
        bpm: Heart rate for automatic cutoff calculation
        
    Returns:
        Smoothed volume array
    """
    if cutoff is None:
        # Calculate cutoff based on heart rate
        beats_per_second = bpm / 60
        beats_per_frame = beats_per_second / fps
        beats_per_video = beats_per_frame * len(volumes)
        cutoff = int(np.ceil(beats_per_video))
    
    # Apply FFT-based low-pass filter
    fft = np.fft.fft(volumes)
    fft[cutoff+1:-cutoff] = 0
    filtered = np.real(np.fft.ifft(fft))
    
    return filtered

def create_volume_curve_video(frames: list, 
                              predictions: np.ndarray,
                              volumes: np.ndarray,
                              la_lengths: list,  # LA Length [(p_start, p_end, length_cm, angle), ...]
                              output_path: str,
                              max_volume_indices: np.ndarray = None,
                              min_volume_indices: np.ndarray = None,
                              fps: int = 30):
    """
    Create a video with LA volume curve visualization below each frame,
    with AI inference overlay (segmentation mask + LA length line).
    
    Args:
        frames: List of video frames (numpy arrays)
        predictions: Segmentation masks for each frame (F, H, W)
        volumes: LA volume array for each frame
        la_lengths: List of LA length info per frame [(p_start, p_end, length_cm, angle), ...]
        output_path: Path to save output video
        max_volume_indices: Indices of maximum volume frames (end-systole)
        min_volume_indices: Indices of minimum volume frames (end-diastole)
        fps: Output video frame rate
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
    
    if len(frames) == 0:
        print("Error: No frames to create volume curve video")
        return
    
    height, width = frames[0].shape[:2]
    plot_height = int(width * 0.35)
    output_height = height + plot_height
    output_width = width
    
    # Determine output format
    if output_path.endswith('.mp4'):
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif output_path.endswith('.avi'):
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
    else:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        output_path += '.mp4'
    
    out = cv2.VideoWriter(output_path, fourcc, fps, (output_width, output_height))
    
    for i, frame in enumerate(tqdm(frames, desc="Creating volume curve video with AI inference")):
        # Convert frame to RGB if needed
        if len(frame.shape) == 2:  # Grayscale
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        elif frame.shape[2] == 3:
            frame_rgb = frame.copy()
        else:
            frame_rgb = frame[:, :, :3].copy()
        
        # Normalize frame to [0, 1] for display
        frame_display = frame_rgb / 255.0 if frame_rgb.max() > 1 else frame_rgb.copy()
        
        # === Add AI Inference Overlay ===
        # 1. Draw segmentation mask
        if predictions is not None and i < len(predictions):
            pred_mask = predictions[i]
            
            # Create colored overlay for mask (semi-transparent blue)
            mask_overlay = np.zeros_like(frame_display)
            mask_overlay[pred_mask == 1] = [0, 0.5, 1.0]  # Blue color
            
            # Blend with original frame
            alpha = 0.4
            frame_display = frame_display * (1 - alpha * (pred_mask[:, :, None] == 1)) + mask_overlay * alpha
        
        # 2. Draw LA Length line
        if la_lengths is not None and i < len(la_lengths):
            la_info = la_lengths[i]
            if la_info is not None:
                la_p_start, la_p_end, length_cm, angle = la_info
                if la_p_start is not None and la_p_end is not None:
                    # Convert to image coordinates (scale to [0, 1] range)
                    xs = [la_p_start[0], la_p_end[0]]
                    ys = [la_p_start[1], la_p_end[1]]
                    
                    # Draw line on frame_display using matplotlib
                    import matplotlib.pyplot as plt
                    from matplotlib.figure import Figure
                    
                    # Create figure for frame with overlay
                    fig_frame = Figure(figsize=(width/100, height/100), dpi=100)
                    ax_frame = fig_frame.add_axes([0, 0, 1, 1])
                    ax_frame.imshow(frame_display)
                    ax_frame.plot(xs, ys, color='yellow', linewidth=3, linestyle='-', marker='o', markersize=6)
                    
                    # Add text annotation
                    ax_frame.text(xs[0], ys[0]-10, f'{length_cm:.1f}cm\n{angle:.1f}°', 
                                 color='yellow', fontsize=12, fontweight='bold',
                                 bbox=dict(boxstyle='round,pad=0.5', facecolor='black', alpha=0.7))
                    ax_frame.axis('off')
                    
                    # Convert figure to array
                    canvas_frame = FigureCanvas(fig_frame)
                    canvas_frame.draw()
                    try:
                        frame_with_line = np.frombuffer(canvas_frame.tostring_rgb(), dtype='uint8')
                        frame_with_line = frame_with_line.reshape((height, width, 3))
                        frame_display = frame_with_line / 255.0
                    except:
                        pass
                    plt.close(fig_frame)
        
        # === Create Volume Curve Plot ===
        fig, ax = plt.subplots(figsize=(8, 2.5))
        ax.plot(volumes, color='skyblue', linewidth=2.5, label='LA Volume')
        ax.axvline(x=i, color='red', linestyle='--', alpha=0.8, linewidth=2.5, label='Current Frame')
        
        # Mark peak frames
        if max_volume_indices is not None and len(max_volume_indices) > 0:
            ax.scatter(max_volume_indices, volumes[max_volume_indices], 
                      color='red', marker='o', s=100, label='Max Volume (ES)', zorder=5, edgecolors='darkred', linewidths=2)
        
        if min_volume_indices is not None and len(min_volume_indices) > 0:
            ax.scatter(min_volume_indices, volumes[min_volume_indices], 
                      color='blue', marker='o', s=100, label='Min Volume (ED)', zorder=5, edgecolors='darkblue', linewidths=2)
        
        ax.set_ylim(0, max(volumes) * 1.15 if max(volumes) > 0 else 1.0)
        ax.set_xlim(0, len(volumes))
        ax.set_xlabel('Frame Number', fontsize=11, fontweight='bold')
        ax.set_ylabel('LA Volume (mm²)', fontsize=11, fontweight='bold')
        ax.set_title(f'Left Atrial Volume Curve - Frame {i}/{len(volumes)-1}', fontsize=12, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3, linestyle='--')
        
        # Convert plot to image
        canvas = FigureCanvas(fig)
        canvas.draw()
        try:
            plot_image = np.frombuffer(canvas.tostring_rgb(), dtype='uint8')
            plot_image = plot_image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
        except Exception:
            buf = canvas.buffer_rgba()
            plot_image_rgba = np.asarray(buf)
            plot_image = plot_image_rgba[:, :, :3]
        
        plt.close(fig)
        
        # Resize plot to match video width
        plot_image = cv2.resize(plot_image, (width, plot_height))
        
        # Convert frame_display back to uint8
        frame_uint8 = (frame_display * 255).astype(np.uint8)
        
        # Stack video frame and plot vertically
        combined_image = np.vstack((frame_uint8, plot_image))
        
        # Write frame
        out.write(cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
    
    out.release()
    print(f"Volume curve video with inference saved to: {output_path}")

#################################################################################
# Configuration
#################################################################################

parser = ArgumentParser()
parser.add_argument("--model_weights", type=str, required=True, help="Path to model checkpoint")
parser.add_argument("--input_video", type=str, help="Single video file (.avi or .dcm) to process")
parser.add_argument("--input_csv", type=str, help="CSV with video paths and metadata")
parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
parser.add_argument("--inference_width", type=int, default=640)
parser.add_argument("--inference_height", type=int, default=480)
parser.add_argument("--metrics_threshold", type=float, default=0.5)
parser.add_argument("--debug", action='store_true', help="Debug mode")
args = parser.parse_args()

# Check that either input_video or input_csv is provided
if not args.input_video and not args.input_csv:
    raise ValueError("Either --input_video or --input_csv must be provided")
if args.input_video and args.input_csv:
    raise ValueError("Cannot use both --input_video and --input_csv at the same time")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
NUM_CLASSES = 1
INFERENCE_WIDTH = args.inference_width
INFERENCE_HEIGHT = args.inference_height

# DICOM Tags
REGION_PHYSICAL_DELTA_X_SUBTAG = (0x0018, 0x602C)
REGION_PHYSICAL_DELTA_Y_SUBTAG = (0x0018, 0x602E)

print(f"Starting LA Volume Video Inference...")
print(f"Device: {DEVICE}")
print(f"Debug mode: {args.debug}")

#################################################################################
# Model Loading
#################################################################################

def extract_key_hook(module, input, output): 
    return output["out"]

weights_path = args.model_weights
if not Path(weights_path).exists(): 
    raise FileNotFoundError(f"Checkpoint not found: {weights_path}")

checkpoint = torch.load(weights_path, map_location=DEVICE)
if 'state_dict' in checkpoint:
    state_dict = checkpoint['state_dict']
else:
    state_dict = checkpoint

# Remove 'm.' prefix if exists
state_dict = {k.replace('m.', ''): v for k, v in state_dict.items()}

backbone = deeplabv3_resnet50(num_classes=NUM_CLASSES)
backbone.register_forward_hook(extract_key_hook)
print(backbone.load_state_dict(state_dict, strict=False))
backbone = backbone.to(DEVICE)
backbone.eval()

#################################################################################
# Prepare input list
#################################################################################

videos_to_process = []

if args.input_video:
    # Single video mode
    input_path = Path(args.input_video)
    if not input_path.exists():
        raise FileNotFoundError(f"Video file not found: {input_path}")
    
    # Determine input type
    if input_path.suffix.lower() == '.avi':
        input_type = 'avi'
    elif input_path.suffix.lower() == '.dcm':
        input_type = 'dcm'
    else:
        raise ValueError(f"Unsupported file format: {input_path.suffix}. Use .avi or .dcm")
    
    # Default values for single video (no physical scale info for AVI)
    videos_to_process.append({
        'video_path': str(input_path),
        'file_uid': input_path.stem,
        'input_type': input_type,
        'PhysicalDeltaX': None,
        'PhysicalDeltaY': None,
        'height_in_dicom': None,
        'width_in_dicom': None,
    })
    
    print(f"Processing single video: {input_path}")

# elif args.input_csv:
# # David Choi Note:
# # Please Ignore at this time, I think this will be needed for future case, input from CSV.
#     # CSV mode
#     df_input = pd.read_csv(args.input_csv)
#     print(f"Loaded {len(df_input)} videos from CSV")
    
#     required_cols = ['path_column', 'file_uid', 'PhysicalDeltaX_doppler', 'PhysicalDeltaY_doppler', 'image_size']
#     for col in required_cols:
#         if col not in df_input.columns:
#             raise ValueError(f"Required column '{col}' not found in CSV")
    
#     for idx, row in df_input.iterrows():
#         video_path = row['path_column']
        
#         # Determine input type from file extension
#         if video_path.lower().endswith('.avi'):
#             input_type = 'avi'
#         elif video_path.lower().endswith('.dcm'):
#             input_type = 'dcm'
#         else:
#             print(f"Warning: Unknown file type for {video_path}, skipping")
#             continue
        
#         # Parse image_size
#         image_size_str = row['image_size'].strip('() ')
#         sizes = image_size_str.split(', ')
#         height_in_dicom = int(sizes[0])
#         width_in_dicom = int(sizes[1])
        
#         videos_to_process.append({
#             'video_path': video_path,
#             'file_uid': row['file_uid'],
#             'input_type': input_type,
#             'PhysicalDeltaX': row['PhysicalDeltaX_doppler'],
#             'PhysicalDeltaY': row['PhysicalDeltaY_doppler'],
#             'height_in_dicom': height_in_dicom,
#             'width_in_dicom': width_in_dicom,
#         })

#################################################################################
# Output directories

output_dir = Path(args.output_dir)
output_dir.mkdir(parents=True, exist_ok=True)
overlay_output_dir = output_dir / "overlays"
overlay_output_dir.mkdir(parents=True, exist_ok=True)

print(f"Output directory: {output_dir}")

#################################################################################
# Inference Loop


results_list = []

for idx, video_info in enumerate(tqdm(videos_to_process, desc="Processing videos")):
    try:
        video_path = video_info['video_path']
        file_uid = video_info['file_uid']
        input_type = video_info['input_type']
        
        frames = []
        PhysicalDeltaX = video_info['PhysicalDeltaX']
        PhysicalDeltaY = video_info['PhysicalDeltaY']
        
        # Load video based on input type
        if input_type == 'avi':
            # Load AVI
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                print(f"Error: Could not open AVI video {video_path}")
                continue
            
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_resized = cv2.resize(frame, (INFERENCE_WIDTH, INFERENCE_HEIGHT))
                frames.append(frame_resized)
            cap.release()
            
            # For AVI without DICOM metadata, use pixel-based measurements
            if PhysicalDeltaX is None:
                PhysicalDeltaX = 1.0  # Default: 1 pixel = 1 unit
            if PhysicalDeltaY is None:
                PhysicalDeltaY = 1.0
            
            # Use current frame size as "original" size
            height_in_dicom = INFERENCE_HEIGHT
            width_in_dicom = INFERENCE_WIDTH
            
        elif input_type == 'dcm':
            # Load DICOM
            ds = pydicom.dcmread(video_path)
            input_dicom = ds.pixel_array  # (F, H, W, C)
            
            if len(input_dicom.shape) == 3:
                # Single frame: (H, W, C)
                input_dicom = input_dicom[np.newaxis, ...]
            
            # Get physical deltas from DICOM if not provided
            if PhysicalDeltaX is None or PhysicalDeltaY is None:
                doppler_region = get_coordinates_from_dicom(ds)
                if doppler_region:
                    region = doppler_region[0]
                    if REGION_PHYSICAL_DELTA_X_SUBTAG in region:
                        PhysicalDeltaX = abs(region[REGION_PHYSICAL_DELTA_X_SUBTAG].value)
                    if REGION_PHYSICAL_DELTA_Y_SUBTAG in region:
                        PhysicalDeltaY = abs(region[REGION_PHYSICAL_DELTA_Y_SUBTAG].value)
            
            # Fallback to pixel-based if still None
            if PhysicalDeltaX is None:
                PhysicalDeltaX = 1.0
            if PhysicalDeltaY is None:
                PhysicalDeltaY = 1.0
            
            # Get original dimensions
            if video_info['height_in_dicom'] is not None:
                height_in_dicom = video_info['height_in_dicom']
                width_in_dicom = video_info['width_in_dicom']
            else:
                height_in_dicom = input_dicom.shape[1]
                width_in_dicom = input_dicom.shape[2]
            
            # Process frames
            for frame in input_dicom:
                if hasattr(ds, 'PhotometricInterpretation') and ds.PhotometricInterpretation == "YBR_FULL_422":
                    frame = ybr_to_rgb(frame)
                resized_frame = cv2.resize(frame, (INFERENCE_WIDTH, INFERENCE_HEIGHT))
                frames.append(resized_frame)
        
        if len(frames) == 0:
            print(f"Error: No frames in video {video_path}")
            continue
        
        # Calculate scaling factors
        scaling_x = width_in_dicom / INFERENCE_WIDTH
        scaling_y = height_in_dicom / INFERENCE_HEIGHT
        
        # Convert to tensor
        input_tensor = torch.tensor(np.array(frames))  # (F, H, W, C)
        input_tensor = input_tensor.float() / 255.0
        input_tensor = input_tensor.permute(0, 3, 1, 2)  # (F, C, H, W)
        input_tensor = input_tensor.to(DEVICE)
        
        # Frame-by-frame inference
        predictions = []
        la_lengths_info = []
        for i in range(input_tensor.shape[0]):
            frame_input = input_tensor[i].unsqueeze(0)  # (1, C, H, W)
            with torch.no_grad():
                logits = backbone(frame_input)  # (1, NUM_CLASSES, H, W)
                pred_mask = (torch.sigmoid(logits) > args.metrics_threshold).squeeze().cpu().numpy()
            predictions.append(pred_mask)
        
        predictions = np.array(predictions)  # (F, H, W)
        
        # Process each frame
        for frame_idx in range(len(predictions)):
            pred_mask_np = predictions[frame_idx]  # (H, W)
            frame_rgb = frames[frame_idx]  # Already resized
            
            # Calculate area
            white_pixel_count = int(np.sum(pred_mask_np))
            area_per_pixel = PhysicalDeltaX * PhysicalDeltaY
            AI_calculated_area_mm2 = white_pixel_count * area_per_pixel * (scaling_x * scaling_y)
            
            # Calculate LA Length
            AI_calculated_length_cm = None
            la_p_start = None
            la_p_end = None
            la_line_angle_deg = None
            
            try:
                la_length_pixels, la_p_start, la_p_end = calculate_la_length_from_mask(pred_mask_np)
                
                if la_length_pixels is not None:
                    AI_calculated_length_cm = la_length_pixels * scaling_y * PhysicalDeltaY / 10  # mm to cm
                    
                    dx = la_p_end[0] - la_p_start[0]
                    dy = la_p_end[1] - la_p_start[1]
                    la_line_angle_deg = np.degrees(np.arctan2(dy, dx))
                    
            except Exception as e:
                pass
            
            la_lengths_info.append((la_p_start, la_p_end, AI_calculated_length_cm, la_line_angle_deg))
            
            # Create overlay
            fig, ax = plt.subplots(1, 1, figsize=(8, 8), dpi=100)
            frame_display = frame_rgb / 255.0 if frame_rgb.max() > 1 else frame_rgb
            
            ax.imshow(frame_display)
            ax.imshow(np.ma.masked_where(pred_mask_np != 1, pred_mask_np), cmap='jet', alpha=0.5)
            
            # Draw LA Length line
            if la_p_start is not None and la_p_end is not None:
                xs = [la_p_start[0], la_p_end[0]]
                ys = [la_p_start[1], la_p_end[1]]
                ax.plot(xs, ys, color='yellow', linewidth=2, linestyle='-', marker='o', markersize=4)
                ax.text(xs[0], ys[0], f'{AI_calculated_length_cm:.1f}cm, {la_line_angle_deg:.1f}°', 
                       color='white', fontsize=10, bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))
            
            title_text = f"{file_uid} Frame {frame_idx}\nArea: {AI_calculated_area_mm2:.1f}mm²"
            if AI_calculated_length_cm:
                title_text += f"\nLength: {AI_calculated_length_cm:.1f}cm, Angle: {la_line_angle_deg:.1f}°"
            ax.set_title(title_text, fontsize=9)
            ax.axis('off')
            
            overlay_save_path = overlay_output_dir / f"{file_uid}_frame{frame_idx:04d}.png"
            plt.tight_layout(pad=0)
            plt.savefig(overlay_save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            
            # Save results
            results_list.append({
                'file_uid': file_uid,
                'frame_number': frame_idx,
                'video_path': video_path,
                'input_type': input_type,
                'overlay_path': str(overlay_save_path),
                'foreground_pixel_count': white_pixel_count,
                'AI_calculated_area_mm2': AI_calculated_area_mm2,
                'AI_calculated_length_cm': AI_calculated_length_cm,
                'LA_line_angle_deg': la_line_angle_deg,
                'PhysicalDeltaX': PhysicalDeltaX,
                'PhysicalDeltaY': PhysicalDeltaY,
            })
            
                # ===== After processing all frames for this video =====
        # (Add this before "if args.debug and idx >= 0:" line)
        
        if len(results_list) > 0:
            # Extract volume data for this video
            video_results = [r for r in results_list if r['file_uid'] == file_uid]
            
            if len(video_results) > 0:
                volumes_ml = np.array([r['AI_calculated_area_mm2'] for r in video_results])
                
                # Smooth volume curve
                smooth_volumes = smooth_volume_curve(volumes_ml, fps=30, bpm=100)
                
                # Find max volume frames (end-systole for LA)
                max_vol_indices, min_vol_indices = get_la_max_volume_frame(
                    smooth_volumes, 
                    smoothing=True,
                    distance=15
                )
                
                # Update results with smoothed volumes and phase info
                for idx_frame, result in enumerate(video_results):
                    result['AI_calculated_volume_smoothed_ml'] = smooth_volumes[idx_frame]
                    result['is_max_volume_frame'] = idx_frame in max_vol_indices
                    result['is_min_volume_frame'] = idx_frame in min_vol_indices
                
                # Create volume curve video
                volume_video_path = overlay_output_dir / f"{file_uid}_volume_curve.mp4"
                create_volume_curve_video(
                    frames=frames,
                    predictions=predictions,
                    volumes=smooth_volumes,
                    la_lengths=la_lengths_info,
                    output_path=str(volume_video_path),
                    max_volume_indices=max_vol_indices,
                    min_volume_indices=min_vol_indices,
                    fps=30
                )
                
                # Print summary
                if len(max_vol_indices) > 0:
                    max_vol_frame = max_vol_indices[0]
                    max_vol_value = smooth_volumes[max_vol_frame]
                    print(f"[{file_uid}] Max LA Volume (End-Systole): {max_vol_value:.1f} mL at frame {max_vol_frame}")
                
                if len(min_vol_indices) > 0:
                    min_vol_frame = min_vol_indices[0]
                    min_vol_value = smooth_volumes[min_vol_frame]
                    print(f"[{file_uid}] Min LA Volume (End-Diastole): {min_vol_value:.1f} mL at frame {min_vol_frame}")
                
                # Calculate LA ejection fraction if both max and min exist
                if len(max_vol_indices) > 0 and len(min_vol_indices) > 0:
                    la_max = smooth_volumes[max_vol_indices[0]]
                    la_min = smooth_volumes[min_vol_indices[0]]
                    if la_max > 0:
                        la_ef = ((la_max - la_min) / la_max) * 100
                        # print(f"[{file_uid}] LA Ejection Fraction: {la_ef:.1f}%")
                        # David Choi Note: Commented out above print to reduce verbosity. I am assuming this la-ef could be useful for qc, but basically does not need for calculating LAVi.
            
            
        
        # Debug mode: process only first video
        if args.debug and idx >= 0:
            print("Debug mode: Stopping after first video")
            break
            
    except Exception as e:
        print(f"Error processing video {file_uid}: {e}")
        import traceback
        traceback.print_exc()
        continue

#################################################################################
# Save Results
#################################################################################

results_df = pd.DataFrame(results_list)
results_csv_path = output_dir / "inference_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"\nSaved {len(results_df)} frame results to {results_csv_path}")
print(f"Processed {len(results_df['file_uid'].unique())} videos")
print("Inference completed!")


# Single AVI file:
# python inference_LAvolume_video.py \
#   --model_weights "/path/to/model.ckpt" \
#   --input_video "/path/to/video.avi" \
#   --output_dir "./output" \
#   --debug

# Single DICOM file:
# python inference_LAvolume_video.py \
#   --model_weights "/path/to/model.ckpt" \
#   --input_video "/path/to/video.dcm" \
#   --output_dir "./output"

# Yuki / David Choi Example:
# python inference_LAvolume_video.py   --model_weights "/workspace/yuki/measurements_internal/wandb/run-20250620_140151-la_area_train_finetune_250620_bi/weights/model_best_epoch_loss.ckpt"   --input_video "/workspace/yuki/measurements_internal/measurements/SAMPLE_DICOM/RV_BASE_SAMPLE_0.dcm"   --output_dir "./AAA"



# David, Note:
# Current code is just for single video input. I think you need to modify for batch processing (CSV or multiple videos).
# 1. Run All for getting LA Area inference results.
# 2. For calculating LAVi, I think AI_calculated_area_mm2 and AI_calculated_length_cm are enough.
# 3. Do QC by LA_line_angle_deg if necessary. I would suggest excluding cases with abs(LA_line_angle_deg) < 55 and > 125 degrees. and Length cm < 2 cm or > 10 is weird.
# 4. merge your csv with filename and predicted view (Basically A4C/A4C-LA and A2C/A2C-LA) with this inference results csv on file_uid column.
# XX. For calculating LAVi, please select L (length) from the A4c-la or A2C-LA view with SHORTER length between the two views.
# For QC, LAEF could be useful, if this is too big, something is wrong with area due to segmentation error by low-quality or wrong view.