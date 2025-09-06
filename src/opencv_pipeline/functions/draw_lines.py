import cv2
import numpy as np

def draw_lines(img, lines, color=(0, 0, 255), thickness=5):
    # Make a blank image to draw lines on
    line_img = np.zeros_like(img)
    if lines is None:
        return img
    # Draw each line segment
    for x1, y1, x2, y2 in lines:
        cv2.line(line_img, (x1, y1), (x2, y2), color, thickness)
    # Overlay the lines on the original image
    return cv2.addWeighted(img, 0.8, line_img, 1.0, 0.0)