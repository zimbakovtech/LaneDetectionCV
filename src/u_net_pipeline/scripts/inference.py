import os
import argparse
from glob import glob
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from moviepy import VideoFileClip, ImageSequenceClip
from skimage import io
import tensorflow as tf
from tqdm import tqdm
from PIL import Image
from src.utils import (
    postprocess_mask,
    overlay_mask_on_image,
    extract_lane_lines,
    draw_lane_lines,
)


def load_model(model_path: str):
    return tf.keras.models.load_model(model_path, compile=False)


def process_image(model, img, img_size):
    h, w = img_size
    img_r = tf.image.resize(tf.convert_to_tensor(img, dtype=tf.float32)/255.0, (h, w))
    pred = model.predict(tf.expand_dims(img_r, 0), verbose=0)[0]
    pred_bin_small = postprocess_mask(pred, threshold=0.5, morph=True)
    # resize mask back to original size
    mask_orig = tf.image.resize(tf.expand_dims(pred_bin_small, -1), (img.shape[0], img.shape[1]), method='nearest').numpy().astype(np.uint8)
    mask_orig = mask_orig[..., 0]

    # Extract lane lines and draw them cleanly
    lines = extract_lane_lines(mask_orig, use_skeleton=True)
    if len(lines) == 0:
        # Fallback: faint overlay of mask if lines not confidently found
        return overlay_mask_on_image(img, mask_orig, color=(0, 255, 255), alpha=0.25)
    out = draw_lane_lines(img, lines, color=(0, 255, 0), thickness=6, antialias=True)
    return out


def run_on_video(model, input_path, output_path, img_size):
    clip = VideoFileClip(input_path)
    fps = clip.fps
    frames = []
    for frame in tqdm(clip.iter_frames(dtype='uint8'), desc='Infer video'):
        overlay = process_image(model, frame, img_size)
        frames.append(overlay)
    out = ImageSequenceClip(frames, fps=fps)
    out.write_videofile(output_path, codec='libx264', audio=False)


def main():
    parser = argparse.ArgumentParser(description='Inference on video/image/folder')
    parser.add_argument('--input', type=str, required=True, help='Path to input video (.mp4) or image or folder')
    parser.add_argument('--model', type=str, default='models/best_model.h5')
    parser.add_argument('--output', type=str, default='outputs/out.mp4')
    parser.add_argument('--img-size', type=int, nargs=2, default=[256, 512], help='height width')
    args = parser.parse_args()

    model = load_model(args.model)
    h, w = args.img_size

    if os.path.isdir(args.input):
        out_dir = args.output if os.path.splitext(args.output)[1]=='' else 'outputs/images'
        os.makedirs(out_dir, exist_ok=True)
        img_files = sorted(glob(os.path.join(args.input, '*.png')) + glob(os.path.join(args.input, '*.jpg')))
        for f in tqdm(img_files, desc='Infer images'):
            img = io.imread(f)
            overlay = process_image(model, img, (h, w))
            io.imsave(os.path.join(out_dir, os.path.basename(f)), overlay)
    elif args.input.lower().endswith('.mp4'):
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        run_on_video(model, args.input, args.output, (h, w))
    else:
        img = io.imread(args.input)
        overlay = process_image(model, img, (h, w))
        out_path = args.output if args.output.lower().endswith(('.png','.jpg')) else 'outputs/infer.png'
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        io.imsave(out_path, overlay)


if __name__ == '__main__':
    main()
