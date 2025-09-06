import cv2
import numpy as np
from collections import deque
from functions.region_of_interest import region_of_interest
from functions.draw_lines import draw_lines

DRAW_Y_TOP_RATIO = 0.65
DRAW_Y_BOTTOM_RATIO = 0.98  

def detect(frame):
    height, width = frame.shape[:2]

    # Define a triangular region of interest
    vertices = np.array([
        (0, height),
        (width // 2, height * 3 // 5),
        (width, height)
    ], dtype=np.int32)

    # 1. Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 2. Apply Gaussian blur
    blur = cv2.GaussianBlur(gray, (5, 5), 0)

    # 3. Perform Canny edge detection
    edges = cv2.Canny(blur, 100, 200)

    # 4. Mask edges image to region of interest
    masked = region_of_interest(edges, vertices)

    # 5. Run Hough transform to find line segments
    segments = cv2.HoughLinesP(
        masked,
        rho=6,
        theta=np.pi / 60,
        threshold=160,
        minLineLength=40,
        maxLineGap=25
    )

    if segments is None:
        return frame

    # 6. Separate segments into left and right based on slope
    left_x, left_y, right_x, right_y = [], [], [], []
    for seg in segments:
        for x1, y1, x2, y2 in seg:
            if x2 == x1:
                continue
            slope = (y2 - y1) / (x2 - x1)
            # Filter out nearly horizontal lines
            if abs(slope) < 0.5:
                continue
            if slope < 0:
                left_x.extend([x1, x2])
                left_y.extend([y1, y2])
            else:
                right_x.extend([x1, x2])
                right_y.extend([y1, y2])

    # 7. Fit a single line to each side using polyfit
    # Use tighter vertical band so rendered lines are shorter and stay on-road
    y_min = int(height * DRAW_Y_TOP_RATIO)
    y_max = int(height * DRAW_Y_BOTTOM_RATIO)
    # Safety: ensure ordering and bounds
    y_min = max(0, min(height - 1, y_min))
    y_max = max(0, min(height - 1, y_max))
    if y_max <= y_min:
        y_min = int(height * 0.70)
        y_max = height - 1
    lines = []

    left_line = None
    right_line = None

    if left_x and left_y:
        left_fit = np.poly1d(np.polyfit(left_y, left_x, deg=1))
        lx_bottom = int(left_fit(y_max))
        lx_top = int(left_fit(y_min))
        lx_bottom = max(0, min(width - 1, lx_bottom))
        lx_top = max(0, min(width - 1, lx_top))
        left_line = (lx_bottom, y_max, lx_top, y_min)

    if right_x and right_y:
        right_fit = np.poly1d(np.polyfit(right_y, right_x, deg=1))
        rx_bottom = int(right_fit(y_max))
        rx_top = int(right_fit(y_min))
        rx_bottom = max(0, min(width - 1, rx_bottom))
        rx_top = max(0, min(width - 1, rx_top))
        right_line = (rx_bottom, y_max, rx_top, y_min)

    # 7b. Impute missing lines using short-term history
    left_line, right_line = _LANE_IMPUTER.update_and_impute(left_line, right_line, y_min, y_max, width)

    if left_line is not None:
        lines.append(left_line)
    if right_line is not None:
        lines.append(right_line)

    # 8. Draw the lane lines back onto the original frame
    output = draw_lines(frame, lines)
    return output


class MissingLaneImputer:
    def __init__(self, window: int = 12, max_gap: int = 6):
        self.window = int(window)
        self.max_gap = int(max_gap)
        self.frame_idx = 0
        self.left_hist: deque[tuple[int, int, int]] = deque(maxlen=self.window)
        self.right_hist: deque[tuple[int, int, int]] = deque(maxlen=self.window)

    def _predict(self, hist: deque, cur_idx: int):
        if len(hist) == 0:
            return None
        
        idxs = np.array([h[0] for h in hist], dtype=np.float32)
        xb = np.array([h[1] for h in hist], dtype=np.float32)
        xt = np.array([h[2] for h in hist], dtype=np.float32)

        if len(hist) == 1:
            return int(xb[-1]), int(xt[-1])
        
        pb = np.polyfit(idxs, xb, deg=1)
        pt = np.polyfit(idxs, xt, deg=1)
        x_bottom = int(np.poly1d(pb)(cur_idx))
        x_top = int(np.poly1d(pt)(cur_idx))
        return x_bottom, x_top

    def update_and_impute(self, left_line, right_line, y_min: int, y_max: int, width: int):
        cur_idx = self.frame_idx

        if left_line is not None:
            self.left_hist.append((cur_idx, left_line[0], left_line[2]))
        if right_line is not None:
            self.right_hist.append((cur_idx, right_line[0], right_line[2]))

        def within_gap(hist):
            return len(hist) > 0 and (cur_idx - hist[-1][0]) <= self.max_gap

        if left_line is None and within_gap(self.left_hist):
            pred = self._predict(self.left_hist, cur_idx)
            if pred is not None:
                xb, xt = pred
                xb = max(0, min(width - 1, xb))
                xt = max(0, min(width - 1, xt))
                left_line = (xb, y_max, xt, y_min)

        if right_line is None and within_gap(self.right_hist):
            pred = self._predict(self.right_hist, cur_idx)
            if pred is not None:
                xb, xt = pred
                xb = max(0, min(width - 1, xb))
                xt = max(0, min(width - 1, xt))
                right_line = (xb, y_max, xt, y_min)

        self.frame_idx += 1
        return left_line, right_line


_LANE_IMPUTER = MissingLaneImputer(window=12, max_gap=6)