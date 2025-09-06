import os
import argparse
import json
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import tensorflow as tf
from keras import optimizers, callbacks, losses
from src.model import build_unet
from src.data_loader import get_datasets
from src.utils import ensure_dir


def dice_loss(y_true, y_pred, smooth=1e-6):
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    return 1.0 - (2.0 * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def bce_dice_loss(y_true, y_pred):
    bce = losses.BinaryCrossentropy()(y_true, y_pred)
    dl = dice_loss(y_true, y_pred)
    return bce + dl


def iou_metric(y_true, y_pred, smooth=1e-6):
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred_bin, [-1])
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    union = tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) - inter
    return (inter + smooth) / (union + smooth)


def dice_metric(y_true, y_pred, smooth=1e-6):
    y_pred_bin = tf.cast(y_pred > 0.5, tf.float32)
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred_bin, [-1])
    inter = tf.reduce_sum(y_true_f * y_pred_f)
    return (2.0 * inter + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)


def main():
    parser = argparse.ArgumentParser(description='Train U-Net for lane segmentation')
    parser.add_argument('--data-dir', type=str, default='datasets')
    parser.add_argument('--img-size', type=int, nargs=2, default=[256, 512], help='height width for training pipeline')
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--base-filters', type=int, default=32)
    parser.add_argument('--depth', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--no-bn', action='store_true', help='Disable batch normalization')
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--model-out', type=str, default='models/best_model.h5')
    parser.add_argument('--norm-range', type=str, choices=['0_1','-1_1'], default='0_1')
    args = parser.parse_args()

    ensure_dir(os.path.dirname(args.model_out))

    height, width = args.img_size
    norm_range = (0.0,1.0) if args.norm_range=='0_1' else (-1.0,1.0)

    strategy = tf.distribute.MirroredStrategy() if len(tf.config.list_physical_devices('GPU')) > 1 else None
    if strategy:
        print('Using MirroredStrategy across GPUs')

    def build_and_compile():
        model = build_unet(input_shape=(height, width, 3), base_filters=args.base_filters, depth=args.depth,
                           use_bn=not args.no_bn, dropout=args.dropout)
        opt = optimizers.Adam(learning_rate=args.lr)
        model.compile(optimizer=opt, loss=bce_dice_loss, metrics=[iou_metric, dice_metric])
        return model

    if strategy:
        with strategy.scope():
            model = build_and_compile()
    else:
        model = build_and_compile()

    train_ds, val_ds, steps_per_epoch, val_steps = get_datasets(args.data_dir, img_size=(height, width), batch_size=args.batch_size, augment=True, norm_range=norm_range)

    # IoU metric callback via validation after each epoch can be custom, but keep Keras metrics for speed
    lr_cb = callbacks.ReduceLROnPlateau(monitor='val_iou_metric', mode='max', factor=0.5, patience=5, min_lr=1e-6, verbose=1)
    es_cb = callbacks.EarlyStopping(monitor='val_iou_metric', mode='max', patience=args.patience, restore_best_weights=True)
    ck_cb = callbacks.ModelCheckpoint(args.model_out, monitor='val_iou_metric', mode='max', save_best_only=True, save_weights_only=False)

    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=args.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=val_steps,
        callbacks=[lr_cb, es_cb, ck_cb]
    )

    # Save training history and config
    out_dir = os.path.dirname(args.model_out)
    hist_path = os.path.join(out_dir, 'train_history.json')
    with open(hist_path, 'w') as f:
        json.dump({k: [float(x) for x in v] for k, v in history.history.items()}, f, indent=2)

    config = {
        'data_dir': args.data_dir,
        'img_size': args.img_size,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'base_filters': args.base_filters,
        'depth': args.depth,
        'dropout': args.dropout,
        'use_bn': not args.no_bn,
        'lr': args.lr,
        'norm_range': args.norm_range,
        'best_model_path': args.model_out
    }
    with open(os.path.join(out_dir, 'train_config.json'), 'w') as f:
        json.dump(config, f, indent=2)


if __name__ == '__main__':
    main()
