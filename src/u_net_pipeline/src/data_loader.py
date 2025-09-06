import os
from typing import Tuple, Optional
import tensorflow as tf
import numpy as np

AUTOTUNE = tf.data.AUTOTUNE


def _load_image_mask(img_path, mask_path, img_size: Tuple[int, int], norm_range=(0.0, 1.0)):
    img = tf.io.read_file(img_path)
    img = tf.io.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    if norm_range == (-1.0, 1.0):
        img = img * 2.0 - 1.0
    mask = tf.io.read_file(mask_path)
    mask = tf.io.decode_png(mask, channels=1)
    mask = tf.image.convert_image_dtype(mask, tf.float32)
    mask = tf.where(mask > 0.5, 1.0, 0.0)
    img = tf.image.resize(img, img_size, method='bilinear')
    mask = tf.image.resize(mask, img_size, method='nearest')
    return img, mask


def _augment(img, mask, seed: Optional[int] = None):
    # Random horizontal flip
    flip = tf.random.uniform(()) > 0.5
    img, mask = tf.cond(
        flip,
        lambda: (tf.image.flip_left_right(img), tf.image.flip_left_right(mask)),
        lambda: (img, mask)
    )
    # Brightness and contrast on image only
    img = tf.image.random_brightness(img, max_delta=0.1)
    img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
    # Small random zoom via central crop then resize back
    scale = tf.random.uniform((), 0.9, 1.0)
    new_h = tf.cast(scale * tf.cast(tf.shape(img)[0], tf.float32), tf.int32)
    new_w = tf.cast(scale * tf.cast(tf.shape(img)[1], tf.float32), tf.int32)
    img_c = tf.image.resize_with_crop_or_pad(img, new_h, new_w)
    mask_c = tf.image.resize_with_crop_or_pad(mask, new_h, new_w)
    img = tf.image.resize(img_c, tf.shape(img)[0:2])
    mask = tf.image.resize(mask_c, tf.shape(mask)[0:2], method='nearest')
    return img, mask


def _make_dataset(images_dir: str, masks_dir: str, img_size: Tuple[int, int], batch_size: int,
                  shuffle: bool = True, augment: bool = False, seed: int = 42, norm_range=(0.0, 1.0)):
    image_files = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    mask_files = [os.path.join(masks_dir, os.path.basename(f).rsplit('.', 1)[0] + '.png') for f in image_files]

    ds = tf.data.Dataset.from_tensor_slices((image_files, mask_files))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(image_files), seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(lambda x, y: _load_image_mask(x, y, img_size, norm_range), num_parallel_calls=AUTOTUNE)
    if augment:
        ds = ds.map(lambda x, y: _augment(x, y), num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(AUTOTUNE)
    return ds, len(image_files)


def get_datasets(data_root: str, img_size: Tuple[int, int] = (256, 512), batch_size: int = 8, augment: bool = True,
                 seed: int = 42, norm_range=(0.0, 1.0)):
    train_images = os.path.join(data_root, 'train', 'images')
    train_masks = os.path.join(data_root, 'train', 'masks')
    val_images = os.path.join(data_root, 'val', 'images')
    val_masks = os.path.join(data_root, 'val', 'masks')

    train_ds, n_train = _make_dataset(train_images, train_masks, img_size, batch_size, shuffle=True, augment=augment, seed=seed, norm_range=norm_range)
    val_ds, n_val = _make_dataset(val_images, val_masks, img_size, batch_size, shuffle=False, augment=False, seed=seed, norm_range=norm_range)
    steps_per_epoch = max(1, n_train // batch_size)
    val_steps = max(1, n_val // batch_size)
    return train_ds, val_ds, steps_per_epoch, val_steps


def get_test_dataset(test_root: str, img_size: Tuple[int, int] = (256, 512), batch_size: int = 8, norm_range=(0.0,1.0)):
    images_dir = os.path.join(test_root, 'images')
    masks_dir = os.path.join(test_root, 'masks')
    ds, n = _make_dataset(images_dir, masks_dir, img_size, batch_size, shuffle=False, augment=False, seed=42, norm_range=norm_range)
    return ds, n
