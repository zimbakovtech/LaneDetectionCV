import os
import cv2
from detect import detect
from functions.preprocess import load_video

INPUT_DIR='data/raw'
OUTPUT_DIR='results'

if __name__ == '__main__':
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    for fname in os.listdir(INPUT_DIR):
        input_path = os.path.join(INPUT_DIR, fname)
        out_path = os.path.join(OUTPUT_DIR, fname)
        writer = None

        for frame in load_video(input_path):
            vis = detect(frame)
            if writer is None:
                h, w = vis.shape[:2]
                writer = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), 20, (w, h))
            writer.write(vis)
            
        writer.release()
        print(f"Processed {fname} → {out_path}")
