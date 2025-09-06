# Road Lane Detection (OpenCV + U-Net)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)  
[![Python Version](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)

Open-source lane detection with two complementary approaches:

- A classical OpenCV pipeline (fast, lightweight).
- A deep-learning U-Net segmentation pipeline (accurate once trained).

## Table of contents

- [Quick start](#quick-start)
- [How the OpenCV pipeline works](#how-the-opencv-pipeline-works)
- [How the U-Net pipeline works](#how-the-u-net-pipeline-works)
- [Repository layout](#repository-layout)
- [Tips and troubleshooting](#tips-and-troubleshooting)
- [License](#license)
- [Authors](#authors)

## Quick start

1. **Install dependencies**

```bash
git clone https://github.com/zimbakovtech/LaneDetectionCV.git
cd LaneDetectionCV
python -m venv venv
pip install -r requirements.txt
```

2. **Run the OpenCV pipeline** on all videos in `data/raw/` and write outputs to `results/`

```bash
python src/opencv_pipeline/main.py
```

3. **U-Net pipeline** (training/inference)

The U-Net code and scripts are under `src/u_net_pipeline/`. See the script help for exact arguments.

```bash
# Prepare dataset and visualize samples
python main.py prepare --fps 5 --img-width 512 --img-height 256

# Train U-Net model
python main.py train --epochs 10 --batch-size 4 --model-out models/best_model.h5

# Run U-Net inference
python main.py infer --input data/raw/road_video_5.mp4 --model models/best_model.h5 --output results/output_video.mp4
```

## How the OpenCV pipeline works

Entry points:

- `src/opencv_pipeline/detect.py` — per-frame lane detection and drawing.
- `src/opencv_pipeline/main.py` — batch process all videos in `data/raw/`.

Processing steps in `detect.py`:

1. **Grayscale** conversion

   - `cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)` produces a single-channel image that’s less sensitive to color changes and faster to process.

2. **Gaussian** blur

   - `cv2.GaussianBlur(gray, (5, 5), 0)` reduces noise and small textures that create spurious edges.

3. **Canny** edge detection

   - `cv2.Canny(blur, 100, 200)` highlights strong intensity gradients (lane paint boundaries). Thresholds can be tuned for your camera/lighting.

4. **Region of interest** (ROI)

   - We apply a polygon mask (a roadway triangle) to focus on the road area and ignore sky/hood.

5. **Hough transform** (probabilistic)

   - `cv2.HoughLinesP(...)` extracts short line segments from the edge map.
   - Parameters (`rho`, `theta`, `threshold`, `minLineLength`, `maxLineGap`) control sensitivity and segment continuity.

6. **Left/right separation** and slope filtering

   - Segments with |slope| < 0.5 are rejected (nearly horizontal).
   - Negative slope → left, positive slope → right, in the camera reference frame.

7. **Robust line fitting**

   - We fit a single line per side with `np.polyfit(y, x, 1)` to combine many segments into one stable lane line.
   - Drawing is intentionally limited to a vertical band of the image (`DRAW_Y_TOP_RATIO`, `DRAW_Y_BOTTOM_RATIO`) so lines don’t extend too far off-road.

8. **Handling dashed/broken lines** (gap filling)

   - A small stateful imputer predicts missing lane endpoints over short gaps using recent history (linear extrapolation over time).
   - This reduces blinking when dashed lines appear/disappear across frames.

9. **Rendering**
   - Final lines are drawn in red on the original frame. Output can be previewed live and written to video.

Key files and helpers (may be used by the pipeline):

- `src/opencv_pipeline/detect.py` — main logic described above (ROI, Canny, Hough, fit, impute, draw).
- `src/opencv_pipeline/functions/region_of_interest.py` and `draw_lines.py` — modular ROI/drawing helpers.

### Tuning the OpenCV pipeline

In `src/opencv_pipeline/detect.py`:

- **Canny thresholds**: increase for fewer edges/noise, decrease to pick up faint paint.
- **Hough parameters**: raise `threshold`/`minLineLength` to reduce false positives; increase `maxLineGap` to bridge gaps.
- `slope` filter (currently 0.5): raise to ignore more shallow segments; lower to accept flatter lanes.
- `DRAW_Y_TOP_RATIO`, `DRAW_Y_BOTTOM_RATIO`: control how long lines are drawn (shorter to avoid leaving the road).
- Missing imputer window/gap: widen if your dashed lines are longer and you need more persistence.

## How the U-Net pipeline works

The U-Net approach performs per-pixel segmentation to classify lane markings, then post-processes the mask to visualize lanes. Typical steps:

1. **Data preparation**

   - Extract frames and, if available, masks (ground truth) from `data/raw/` into a training set.

2. **Model**

   - A U-Net model implemented in Keras/TensorFlow under `src/u_net_pipeline/` (see `src`, `scripts`, and `models` subfolders).

3. **Training**

   - Use the provided training script(s) under `src/u_net_pipeline/scripts/` to train on your dataset. Check each script’s `-h` for exact args.

4. **Inference**
   - Load a trained model (e.g., `src/u_net_pipeline/models/best_model.h5`) and run inference on a video to produce a lane mask and overlay.

Notes:

- Deep models can outperform classical methods on difficult lighting and worn markings, but require annotated data and GPU time to train.
- Start with the OpenCV pipeline for quick results, then move to U-Net if you need higher robustness.

## Repository layout

```
RoadLaneDetection/
├── data/
│   ├── raw/                      # Input videos
│   └── processed/                # (optional) prepared frames/masks
├── src/
│   ├── opencv_pipeline/
│   │   ├── detect.py             # OpenCV lane detection (Canny + Hough + fit + impute)
│   │   ├── main.py               # Batch process all videos in data/raw
│   │   └── functions/
│   │       ├── region_of_interest.py
│   │       ├── draw_lines.py
│   │       └── preprocess.py
│   └── u_net_pipeline/
│       ├── main.py               # Prepare dataset, train model & infer video
│       ├── models/
│       │   └── best_model.h5     # Example trained model (if present)
│       ├── scripts/              # Training & inference helpers
│       └── src/                  # Model and utils
├── results/                      # Outputs (videos, models, logs)
├── requirements.txt
├── .gitignore
├── LICENSE
└── README.md
```

## Tips and troubleshooting

- If output writes but playback looks choppy, ensure output FPS matches input FPS (the scripts handle this automatically where possible).
- If lanes are noisy: increase Canny thresholds and Hough `threshold`, or narrow the ROI polygon.
- If dashed lines blink: increase the imputer window/gap slightly.
- For different camera FOVs/heights, adjust the ROI triangle.

## License

This project is licensed under the MIT License.  
See the [LICENSE](LICENSE) file for details.

## **Authors**

_Prepared by Damjan Zimbakov & Efimija Cuneva_  
_July 2025_
