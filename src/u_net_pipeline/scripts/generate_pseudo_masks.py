import os
import argparse
from glob import glob
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm import tqdm
from skimage import io, color, filters, feature, morphology, exposure
import cv2
from src.utils import ensure_dir


def generate_mask(img: np.ndarray) -> np.ndarray:
    # Convert to HLS for white/yellow detection
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    h, l, s = cv2.split(hls)
    # Threshold white and yellow
    white = (l > 200) & (s < 80)
    yellow = ((h > 15) & (h < 40) & (s > 80))
    color_mask = (white | yellow).astype(np.uint8)

    # Canny edges
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)

    # Combine cues
    combined = cv2.bitwise_or(edges, color_mask * 255)

    # Region of Interest (bottom trapezoid)
    hgt, wdt = combined.shape
    mask_roi = np.zeros_like(combined)
    pts = np.array([
        [int(0.1*wdt), hgt],
        [int(0.45*wdt), int(0.6*hgt)],
        [int(0.55*wdt), int(0.6*hgt)],
        [int(0.9*wdt), hgt]
    ], dtype=np.int32)
    cv2.fillPoly(mask_roi, [pts], 255)
    combined = cv2.bitwise_and(combined, mask_roi)

    # Hough lines to enhance linear structures
    lines = cv2.HoughLinesP(combined, 1, np.pi/180, threshold=50, minLineLength=40, maxLineGap=100)
    line_img = np.zeros_like(combined)
    if lines is not None:
        for l in lines:
            x1, y1, x2, y2 = l[0]
            cv2.line(line_img, (x1,y1), (x2,y2), 255, 3)
    merged = cv2.bitwise_or(combined, line_img)

    # Morphological operations
    kernel = np.ones((3,3), np.uint8)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, kernel, iterations=2)
    merged = cv2.morphologyEx(merged, cv2.MORPH_DILATE, kernel, iterations=1)

    # Binarize
    bin_mask = (merged > 0).astype(np.uint8)
    return bin_mask


def main():
    parser = argparse.ArgumentParser(description='Generate pseudo masks using classic CV heuristics')
    parser.add_argument('--frames', type=str, default='datasets/raw', help='Directory containing subfolders of frames per video')
    parser.add_argument('--out-masks', type=str, default='datasets/raw_masks', help='Output directory for masks (mirror structure)')
    args = parser.parse_args()

    ensure_dir(args.out_masks)

    folders = [d for d in sorted(os.listdir(args.frames)) if os.path.isdir(os.path.join(args.frames, d))]
    for folder in folders:
        src_dir = os.path.join(args.frames, folder)
        dst_dir = os.path.join(args.out_masks, folder)
        ensure_dir(dst_dir)
        images = sorted(glob(os.path.join(src_dir, '*.png')))
        for img_path in tqdm(images, desc=f"Masks for {folder}"):
            img = io.imread(img_path)
            mask = generate_mask(img)
            out_path = os.path.join(dst_dir, os.path.basename(img_path))
            io.imsave(out_path, (mask*255).astype(np.uint8))


if __name__ == '__main__':
    main()
