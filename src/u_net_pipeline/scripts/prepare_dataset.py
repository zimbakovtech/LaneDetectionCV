import os
import argparse
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from glob import glob
from sklearn.model_selection import train_test_split
from src.utils import ensure_dir
from skimage import io, transform
import numpy as np


def list_frames_and_masks(frames_root: str, masks_root: str):
    pairs = []
    video_folders = [d for d in sorted(os.listdir(frames_root)) if os.path.isdir(os.path.join(frames_root, d))]
    for vf in video_folders:
        fdir = os.path.join(frames_root, vf)
        mdir = os.path.join(masks_root, vf)
        imgs = sorted(glob(os.path.join(fdir, '*.png')))
        for img_path in imgs:
            mask_path = os.path.join(mdir, os.path.basename(img_path))
            if os.path.exists(mask_path):
                pairs.append((img_path, mask_path, vf))
    return pairs


def save_resized(img_path, mask_path, out_img_path, out_mask_path, img_size):
    img = io.imread(img_path)
    mask = io.imread(mask_path)
    img_r = transform.resize(img, (img_size[1], img_size[0]), preserve_range=True, anti_aliasing=True).astype(np.uint8)
    mask_r = transform.resize(mask, (img_size[1], img_size[0]), order=0, preserve_range=True, anti_aliasing=False)
    mask_r = (mask_r > 127).astype(np.uint8) * 255
    io.imsave(out_img_path, img_r)
    io.imsave(out_mask_path, mask_r)


def main():
    parser = argparse.ArgumentParser(description='Prepare dataset: split into train/val/test and resize')
    parser.add_argument('--frames', type=str, default='datasets/raw')
    parser.add_argument('--masks', type=str, default='datasets/raw_masks')
    parser.add_argument('--out', type=str, default='datasets')
    parser.add_argument('--img-size', type=int, nargs=2, default=[512, 256], help='width height')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    pairs = list_frames_and_masks(args.frames, args.masks)
    if not pairs:
        print('No frame/mask pairs found. Did you run extract_frames and generate_pseudo_masks?')
        return

    # Group by video to avoid leakage
    video_to_pairs = {}
    for img, msk, vid in pairs:
        video_to_pairs.setdefault(vid, []).append((img, msk))

    videos = list(video_to_pairs.keys())
    train_vids, temp_vids = train_test_split(videos, test_size=0.30, random_state=args.seed)
    val_vids, test_vids = train_test_split(temp_vids, test_size=0.5, random_state=args.seed)

    splits = {
        'train': train_vids,
        'val': val_vids,
        'test': test_vids,
    }

    for split, vids in splits.items():
        out_img_dir = os.path.join(args.out, split, 'images')
        out_mask_dir = os.path.join(args.out, split, 'masks')
        ensure_dir(out_img_dir)
        ensure_dir(out_mask_dir)
        for vid in vids:
            for img_path, mask_path in video_to_pairs[vid]:
                base = os.path.basename(img_path)
                out_img_path = os.path.join(out_img_dir, base)
                out_mask_path = os.path.join(out_mask_dir, base)
                save_resized(img_path, mask_path, out_img_path, out_mask_path, args.img_size)

    print('Dataset prepared at', args.out)


if __name__ == '__main__':
    main()
