import os
import argparse
from glob import glob
import cv2
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm import tqdm
from src.utils import ensure_dir
import numpy as np
from PIL import Image


def extract_frames_from_video(video_path: str, out_dir: str, fps: int = 5, img_size=None):
    ensure_dir(out_dir)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open {video_path}")
        return
    native_fps = cap.get(cv2.CAP_PROP_FPS) or 30
    stride = max(1, int(round(native_fps / fps)))
    basename = os.path.splitext(os.path.basename(video_path))[0]
    frame_dir = os.path.join(out_dir, basename)
    ensure_dir(frame_dir)

    frame_idx = 0
    saved_idx = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    with tqdm(total=total_frames//stride if total_frames>0 else 0, desc=f"Extracting {basename}") as pbar:
        while True:
            ret, frame_bgr = cap.read()
            if not ret:
                break
            if frame_idx % stride == 0:
                frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                if img_size:
                    frame = cv2.resize(frame, (img_size[0], img_size[1]))
                out_path = os.path.join(frame_dir, f"{basename}_{saved_idx:06d}.png")
                Image.fromarray(frame).save(out_path)
                saved_idx += 1
                pbar.update(1)
            frame_idx += 1
    cap.release()


def main():
    parser = argparse.ArgumentParser(description="Extract frames from mp4 videos")
    parser.add_argument('--data-dir', type=str, default='data', help='Directory containing .mp4 files')
    parser.add_argument('--out-dir', type=str, default='datasets/raw', help='Output directory for frames')
    parser.add_argument('--fps', type=int, default=5, help='Sampling rate in frames per second')
    parser.add_argument('--width', type=int, default=None, help='Optional width to resize frames')
    parser.add_argument('--height', type=int, default=None, help='Optional height to resize frames')
    args = parser.parse_args()

    ensure_dir(args.out_dir)
    video_files = glob(os.path.join(args.data_dir, '*.mp4'))
    if not video_files:
        print('No .mp4 files found in', args.data_dir)
        return
    img_size = None
    if args.width and args.height:
        img_size = (args.width, args.height)

    for vf in video_files:
        extract_frames_from_video(vf, args.out_dir, fps=args.fps, img_size=img_size)


if __name__ == '__main__':
    main()
