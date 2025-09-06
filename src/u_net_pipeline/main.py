import argparse
import subprocess
import sys


def run(cmd):
    print('>',' '.join(cmd))
    res = subprocess.run(cmd)
    if res.returncode != 0:
        sys.exit(res.returncode)


def main():
    parser = argparse.ArgumentParser(description='Lane Detection Pipeline CLI')
    sub = parser.add_subparsers(dest='command')

    p_prep = sub.add_parser('prepare', help='Extract frames, generate masks, and prepare dataset')
    p_prep.add_argument('--fps', type=int, default=5)
    p_prep.add_argument('--img-width', type=int, default=512)
    p_prep.add_argument('--img-height', type=int, default=256)

    p_train = sub.add_parser('train', help='Train model')
    p_train.add_argument('--epochs', type=int, default=50)
    p_train.add_argument('--batch-size', type=int, default=8)
    p_train.add_argument('--model-out', type=str, default='models/best_model.h5')

    p_eval = sub.add_parser('evaluate', help='Evaluate model')
    p_eval.add_argument('--model', type=str, default='models/best_model.h5')

    p_infer = sub.add_parser('infer', help='Run inference')
    p_infer.add_argument('--input', type=str, required=True)
    p_infer.add_argument('--model', type=str, default='models/best_model.h5')
    p_infer.add_argument('--output', type=str, default='outputs/out.mp4')

    args = parser.parse_args()

    if args.command == 'prepare':
        run([sys.executable, 'scripts/extract_frames.py', '--data-dir', 'data', '--out-dir', 'datasets/raw', '--fps', str(args.fps), '--width', str(args.img_width), '--height', str(args.img_height)])
        run([sys.executable, 'scripts/generate_pseudo_masks.py', '--frames', 'datasets/raw', '--out-masks', 'datasets/raw_masks'])
        run([sys.executable, 'scripts/prepare_dataset.py', '--frames', 'datasets/raw', '--masks', 'datasets/raw_masks', '--out', 'datasets', '--img-size', str(args.img_width), str(args.img_height)])
    elif args.command == 'train':
        run([sys.executable, 'scripts/train.py', '--data-dir', 'datasets', '--epochs', str(args.epochs), '--batch-size', str(args.batch_size), '--model-out', args.model_out])
    elif args.command == 'evaluate':
        run([sys.executable, 'scripts/evaluate.py', '--model', args.model, '--data-dir', 'datasets/test'])
    elif args.command == 'infer':
        run([sys.executable, 'scripts/inference.py', '--input', args.input, '--model', args.model, '--output', args.output])
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
