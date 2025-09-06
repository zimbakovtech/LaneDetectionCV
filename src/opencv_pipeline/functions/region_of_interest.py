import cv2
import numpy as np

def region_of_interest(img, vertices):
    # Create a blank mask matching the image dimensions
    mask = np.zeros_like(img)
    # Determine mask color based on image channels
    if len(img.shape) > 2:
        channel_count = img.shape[2]
        match_mask_color = (255,) * channel_count
    else:
        match_mask_color = 255
    # Fill the polygon defined by vertices
    cv2.fillPoly(mask, [vertices], match_mask_color)
    # Return the image only in the masked region
    return cv2.bitwise_and(img, mask)
