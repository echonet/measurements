import numpy as np
import cv2
import torch
from typing import Tuple, Union, List
import torch
import numpy as np
import pydicom

ULTRASOUND_REGIONS_TAG = (0x0018, 0x6011)
REGION_X0_SUBTAG = (0x0018, 0x6018)  # left
REGION_Y0_SUBTAG = (0x0018, 0x601A)  # top
REGION_X1_SUBTAG = (0x0018, 0x601C)  # right
REGION_Y1_SUBTAG = (0x0018, 0x601E)  # bottom
STUDY_DESCRIPTION_TAG = (0x0008, 0x1030)
SERIES_DESCRIPTION_TAG = (0x0008, 0x103E)
PHOTOMETRIC_INTERPRETATION_TAG = (0x0028, 0x0004)
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
        return None, None
    
def find_horizontal_line(
    image: np.ndarray,
    angle_threshold: float = np.pi / 180,
    line_threshold: float = 100,
) -> int:
    """
    Horizontal line detection for Doppler images.
    
    Uses Canny edge detection and the Hough Transform to find the most prominent horizontal line in the image. 
    Returns the y-coordinate of this line.
    """

    if len(image.shape) == 2: #Already gray image
        gray_image = image
    else:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    edges = cv2.Canny(gray_image, 50, 150, apertureSize=3)
    lines = cv2.HoughLines(edges, 1, np.pi / 180, line_threshold)

    if lines is not None:
        for rho, theta in lines[:, 0]:
            if (
                abs(theta - np.pi / 2) < angle_threshold
                or abs(theta - 3 * np.pi / 2) < angle_threshold
            ):
                a = np.cos(theta)
                b = np.sin(theta)
                x0 = a * rho
                y0 = b * rho
                y = int(y0)
                return y
    return None