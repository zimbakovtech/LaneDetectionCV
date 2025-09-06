import os
import json
from typing import Tuple, List, Optional
import numpy as np
from skimage import img_as_ubyte
from skimage.morphology import binary_closing, square
from skimage.color import gray2rgb
from skimage.morphology import skeletonize
import cv2


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def save_config(config: dict, path: str):
    ensure_dir(os.path.dirname(path))
    with open(path, 'w') as f:
        json.dump(config, f, indent=2)


def dice_coefficient(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    intersection = np.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (np.sum(y_true_f) + np.sum(y_pred_f) + smooth)


def iou_score(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    intersection = np.sum(y_true_f * y_pred_f)
    union = np.sum(y_true_f) + np.sum(y_pred_f) - intersection
    return (intersection + smooth) / (union + smooth)


def precision_recall(y_true, y_pred, smooth=1e-6):
    y_true_f = y_true.reshape(-1)
    y_pred_f = y_pred.reshape(-1)
    tp = np.sum(y_true_f * y_pred_f)
    fp = np.sum((1 - y_true_f) * y_pred_f)
    fn = np.sum(y_true_f * (1 - y_pred_f))
    precision = (tp + smooth) / (tp + fp + smooth)
    recall = (tp + smooth) / (tp + fn + smooth)
    return precision, recall


def postprocess_mask(mask: np.ndarray, threshold: float = 0.5, morph: bool = True) -> np.ndarray:
    """
    Convert probability mask to binary, optionally apply morphological closing.
    mask: HxW or HxWx1 float array in [0,1].
    """
    if mask.ndim == 3:
        mask = mask[..., 0]
    bin_mask = (mask >= threshold).astype(np.uint8)
    if morph:
        bin_mask = binary_closing(bin_mask, square(3)).astype(np.uint8)
    return bin_mask


def overlay_mask_on_image(image: np.ndarray, mask: np.ndarray, color=(255, 0, 0), alpha=0.4) -> np.ndarray:
    """
    Overlay a binary mask on an RGB image.
    """
    if image.ndim == 2:
        image = gray2rgb(image)
    overlay = image.copy()
    color_arr = np.array(color, dtype=np.uint8)
    mask_bool = mask.astype(bool)
    overlay[mask_bool] = (alpha * color_arr + (1 - alpha) * overlay[mask_bool]).astype(np.uint8)
    return overlay


def normalize_image(img: np.ndarray, to_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    img = img.astype(np.float32) / 255.0
    if to_range == (-1.0, 1.0):
        img = img * 2.0 - 1.0
    return img


def denormalize_image(img: np.ndarray, from_range: Tuple[float, float] = (0.0, 1.0)) -> np.ndarray:
    if from_range == (-1.0, 1.0):
        img = (img + 1.0) / 2.0
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


# ---------------- Lane line post-processing helpers ---------------- #

def _default_road_roi(h: int, w: int) -> np.ndarray:
    """Return a trapezoid ROI mask polygon for a typical forward-facing dashcam.
    Coordinates are in image reference frame (y down, x right).
    """
    y_top = int(0.58 * h)
    y_bottom = h - 1
    x_top_left = int(0.42 * w)
    x_top_right = int(0.58 * w)
    margin = int(0.03 * w)
    x_bottom_left = margin
    x_bottom_right = w - margin
    poly = np.array([
        [x_bottom_left, y_bottom],
        [x_top_left, y_top],
        [x_top_right, y_top],
        [x_bottom_right, y_bottom],
    ], dtype=np.int32)
    return poly


def apply_roi_mask(mask: np.ndarray, roi_polygon: Optional[np.ndarray] = None) -> np.ndarray:
    """Apply a polygonal ROI to the binary mask, zeroing out pixels outside the ROI."""
    h, w = mask.shape[:2]
    if roi_polygon is None:
        roi_polygon = _default_road_roi(h, w)
    roi = np.zeros((h, w), dtype=np.uint8)
    cv2.fillPoly(roi, [roi_polygon], 1)
    return (mask.astype(np.uint8) & roi).astype(np.uint8)


def _line_length(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    return float(np.hypot(p2[0]-p1[0], p2[1]-p1[1]))


def _fit_average_line(lines: List[Tuple[Tuple[int, int], Tuple[int, int]]], h: int, side: str) -> Optional[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Average multiple short line segments into a single long line spanning from bottom to ~60% height.
    side: 'left' or 'right' controls bottom x bound fallback.
    Returns ((x1,y1),(x2,y2)) or None.
    """
    if not lines:
        return None
    # Fit slope-intercept weighted by length
    slopes, intercepts, weights = [], [], []
    for (x1, y1), (x2, y2) in lines:
        if x2 == x1:
            continue
        m = (y2 - y1) / (x2 - x1)
        if abs(m) < 0.2:  # discard nearly horizontal
            continue
        b = y1 - m * x1
        w = _line_length((x1, y1), (x2, y2))
        slopes.append(m)
        intercepts.append(b)
        weights.append(w)
    if not weights:
        return None
    slopes = np.array(slopes)
    intercepts = np.array(intercepts)
    weights = np.array(weights)
    m_avg = np.average(slopes, weights=weights)
    b_avg = np.average(intercepts, weights=weights)
    # Define two y's and compute x's on the line: bottom and top-of-ROI
    y1 = h - 1
    y2 = int(0.62 * h)
    if abs(m_avg) < 1e-3:
        return None
    x1 = int((y1 - b_avg) / m_avg)
    x2 = int((y2 - b_avg) / m_avg)
    # Clip to image bounds
    x1 = int(np.clip(x1, 0, 10**9))
    x2 = int(np.clip(x2, 0, 10**9))
    return (x1, y1), (x2, y2)


def extract_lane_lines(binary_mask: np.ndarray, use_skeleton: bool = True,
                       roi_polygon: Optional[np.ndarray] = None) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    """Extract two main lane lines from a binary segmentation mask.

    Returns up to two lines as endpoints: [((x1,y1),(x2,y2)), ...]
    """
    if binary_mask.ndim == 3:
        binary_mask = binary_mask[..., 0]
    mask = (binary_mask > 0).astype(np.uint8)

    # Focus on the road region to suppress trees/sky
    mask = apply_roi_mask(mask, roi_polygon)

    # Optional skeletonization to get crisp line structures
    if use_skeleton:
        skel = skeletonize(mask.astype(bool)).astype(np.uint8) * 255
        edges = skel
    else:
        edges = cv2.Canny((mask*255).astype(np.uint8), 50, 150)

    h, w = mask.shape

    # Probabilistic Hough Transform via OpenCV
    lines_p = cv2.HoughLinesP(
        edges,
        rho=1,
        theta=np.pi/180,
        threshold=max(20, w//32),
        minLineLength=max(30, w//8),
        maxLineGap=20,
    )

    if lines_p is None:
        return []

    left_segs: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    right_segs: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    center_x = w / 2.0

    for l in lines_p[:, 0, :]:
        x1, y1, x2, y2 = map(int, l.tolist())
        if x2 == x1:
            m = np.inf
        else:
            m = (y2 - y1) / (x2 - x1)
        # Reject nearly horizontal or upward lines
        if not np.isfinite(m) or abs(m) < 0.2:
            continue
        # Decide side by x at bottom-most endpoint
        xb, yb = (x1, y1) if y1 > y2 else (x2, y2)
        seg = ((x1, y1), (x2, y2))
        if m < 0 and xb < center_x:
            left_segs.append(seg)
        elif m > 0 and xb > center_x:
            right_segs.append(seg)

    left_line = _fit_average_line(left_segs, h, side='left')
    right_line = _fit_average_line(right_segs, h, side='right')

    lines: List[Tuple[Tuple[int, int], Tuple[int, int]]] = []
    if left_line is not None:
        lines.append(left_line)
    if right_line is not None:
        lines.append(right_line)
    return lines


def draw_lane_lines(image: np.ndarray,
                    lines: List[Tuple[Tuple[int, int], Tuple[int, int]]],
                    color: Tuple[int, int, int] = (0, 255, 0),
                    thickness: int = 6,
                    antialias: bool = True) -> np.ndarray:
    """Draw lane lines on a copy of the image. Returns the annotated image."""
    if image.ndim == 2:
        image = gray2rgb(image)
    out = image.copy()
    # If antialias not available, cv2.line uses 8-connected default
    line_type = cv2.LINE_AA if antialias else cv2.LINE_8
    for (x1, y1), (x2, y2) in lines:
        cv2.line(out, (int(x1), int(y1)), (int(x2), int(y2)), color, thickness, line_type)
    return out

