import cv2
import numpy as np
import os

params = {"canny_low": 100, "canny_high": 250, "hough_rho": 1, "hough_theta": 0.017453292519943295, "hough_threshold": 100, "min_line_length": 100, "max_line_gap": 10, "slope_threshold": 0.5}

def detect_lanes(frame, params):
    canny_low = params['canny_low']
    canny_high = params['canny_high']
    hough_rho = params['hough_rho']
    hough_theta = params['hough_theta']
    hough_threshold = params['hough_threshold']
    min_line_length = params['min_line_length']
    max_line_gap = params['max_line_gap']
    slope_threshold = params['slope_threshold']
    
    # Convert to grayscale and apply blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Edge detection
    edges = cv2.Canny(blur, canny_low, canny_high)
    
    # Define region of interest (ROI)
    height, width = frame.shape[:2]
    mask = np.zeros_like(edges)
    polygon = np.array([[(0, height), (width, height), (width//2, height//2)]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)
    
    # Hough transform to detect lines
    lines = cv2.HoughLinesP(masked_edges, hough_rho, hough_theta, hough_threshold, 
                            minLineLength=min_line_length, maxLineGap=max_line_gap)
    
    # Separate lines into left and right lanes based on slope
    left_lines = []
    right_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0:  # Avoid division by zero
                continue
            slope = (y2 - y1) / (x2 - x1)
            if abs(slope) < slope_threshold:  # Filter out near-horizontal lines
                continue
            if slope > 0:  # Positive slope for left lane
                left_lines.append(line)
            else:  # Negative slope for right lane
                right_lines.append(line)
    return left_lines, right_lines

def compute_x_intercept(line, height):
    x1, y1, x2, y2 = line[0]
    if x2 - x1 == 0:
        return x1
    slope = (y2 - y1) / (x2 - x1)
    intercept = x1 - slope * y1
    return intercept + slope * height

# Initialize video capture (replace with your test video)
input_path = 'data/raw/road_video_3.mp4'
cap = cv2.VideoCapture(input_path)
if not cap.isOpened():
    print("Error: Could not open video file.")
    exit(1)

# Prepare video writer (preserve FPS & size)
os.makedirs('results', exist_ok=True)
base = os.path.basename(input_path)  # e.g., road_video_5.mp4
out_path = os.path.join('results', f"lane_{base}")

fps = cap.get(cv2.CAP_PROP_FPS)
if not fps or fps <= 0:
    fps = 20.0
writer = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect lanes using optimized parameters
    left_lines, right_lines = detect_lanes(frame, params)
    
    # Draw detected lanes on the frame
    for line in left_lines + right_lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(frame, (x1, y1), (x2, y2), (0, 0, 255), thickness=8)
    
    # Init writer once size is known
    if writer is None:
        h, w = frame.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

    # Write and display the frame
    writer.write(frame)
    cv2.imshow('Lane Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
if writer is not None:
    writer.release()
cv2.destroyAllWindows()
print(f"Saved output → {out_path}")