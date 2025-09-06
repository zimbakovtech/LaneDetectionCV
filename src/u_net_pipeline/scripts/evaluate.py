import os
import argparse
import numpy as np
import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from glob import glob
from skimage import io
import tensorflow as tf
from tqdm import tqdm
from src.utils import iou_score, dice_coefficient, precision_recall, postprocess_mask, overlay_mask_on_image


def load_model(model_path: str):
    custom_objects = {}
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
    except Exception:
        model = tf.keras.models.load_model(model_path, custom_objects=custom_objects, compile=False)
    return model


def main():
    parser = argparse.ArgumentParser(description='Evaluate model on test set and save overlays')
    parser.add_argument('--model', type=str, default='models/best_model.h5')
    parser.add_argument('--data-dir', type=str, default='datasets/test')
    parser.add_argument('--out-dir', type=str, default='outputs/visuals')
    parser.add_argument('--img-size', type=int, nargs=2, default=[256, 512], help='height width')
    args = parser.parse_args()

    model = load_model(args.model)

    img_dir = os.path.join(args.data_dir, 'images')
    mask_dir = os.path.join(args.data_dir, 'masks')
    img_files = sorted(glob(os.path.join(img_dir, '*.png')))

    ious, dices, precisions, recalls = [], [], [], []

    os.makedirs(args.out_dir, exist_ok=True)

    for img_path in tqdm(img_files, desc='Evaluating'):
        mask_path = os.path.join(mask_dir, os.path.basename(img_path))
        img = io.imread(img_path)
        mask = io.imread(mask_path)
        h, w = args.img_size
        img_r = tf.image.resize(tf.convert_to_tensor(img, dtype=tf.float32)/255.0, (h, w))
        pred = model.predict(tf.expand_dims(img_r, 0), verbose=0)[0]
        pred_bin = postprocess_mask(pred, threshold=0.5, morph=True)
        mask_arr = (mask>127).astype(np.float32)
        if mask_arr.ndim == 2:
            mask_arr = np.expand_dims(mask_arr, -1)
        mask_r = tf.image.resize(tf.convert_to_tensor(mask_arr), (h, w), method='nearest').numpy()
        mask_bin = (mask_r[...,0] > 0.5).astype(np.uint8)

        ious.append(iou_score(mask_bin, pred_bin))
        dices.append(dice_coefficient(mask_bin, pred_bin))
        p, r = precision_recall(mask_bin, pred_bin)
        precisions.append(p); recalls.append(r)

        overlay = overlay_mask_on_image(img, pred_bin)
        io.imsave(os.path.join(args.out_dir, os.path.basename(img_path)), overlay)

    print(f"Mean IoU: {np.mean(ious):.4f}")
    print(f"Mean Dice: {np.mean(dices):.4f}")
    print(f"Mean Precision: {np.mean(precisions):.4f}")
    print(f"Mean Recall: {np.mean(recalls):.4f}")


if __name__ == '__main__':
    main()
