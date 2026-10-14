# Video Processing with OpenCV

Python project for the Multimedia Systems course (Islamic University of Gaza). It processes a short video with OpenCV: reads it, extracts frames, applies simple image processing, and (bonus) recognizes the video content with a pre-trained MobileNetV2 model.

## Features

- Read a video and print its resolution, FPS, number of frames, and duration
- Extract one frame every 0.5 seconds and save them as JPG files
- Apply grayscale conversion and brightness adjustment to 3 frames
- Visualize the extracted and processed frames with Matplotlib
- **Bonus:** classify 5 frames with MobileNetV2 (ONNX, run with OpenCV DNN), using full-frame and multi-crop (3 x 2 tiles) classification

## Project structure

```
.
├── main.py            # frame extraction and image processing
├── bonus.py           # video content recognition (bonus)
├── requirements.txt
├── videos/video.mp4   # your input video (not included)
├── models/            # downloaded automatically by bonus.py
└── output/            # results
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Put your video at `videos/video.mp4`.

## Usage

```bash
python main.py     # extraction + processing
python bonus.py    # bonus: content recognition
```

The first run of `bonus.py` downloads the MobileNetV2 model (~14 MB) and the ImageNet class names into `models/`.

## Output

**`main.py`** writes to `output/`:

| Output | Description |
|---|---|
| `extracted_frames/` | one JPG every 0.5 seconds (72 frames for a 36 s video at 59.94 FPS) |
| `processed_frames/` | grayscale and brightness versions of 3 frames |
| `video_information.json`, `extracted_frames.json` | video info and the list of extracted frames |
| `extracted_frames_visualization.png` | grid of 6 extracted frames |
| `processing_comparison.png` | original vs grayscale vs brightness |

**`bonus.py`** writes to `output/bonus_output/`:

| Output | Description |
|---|---|
| `frame_N.jpg` (5 files) | the analyzed frames |
| `frame_N_tiles.jpg` (5 files) | frames with the 6 tiles and each tile's predicted label |
| `bonus_predictions.png` | the 5 frames with their predictions |
| `bonus_predictions.csv` | table of predictions and confidence |

## How the bonus works

ImageNet models recognize one dominant object per image, so classifying a whole street scene gives weak and inconsistent labels. Each frame is therefore classified in two ways:

1. **Full frame:** the whole image resized to 224 x 224 (baseline).
2. **Multi-crop:** the frame is split into 6 tiles, each tile is classified, and the probabilities of the 6 tiles and the full frame are averaged.

Example results on a city street video:

| Time (s) | Full frame | Multi-crop top-1 |
|---|---|---|
| 0.00 | jinrikisha (10.7%) | pole (8.2%) |
| 8.98 | shovel (13.3%) | pole (10.6%) |
| 17.97 | jinrikisha (16.1%) | pole (9.3%) |
| 26.96 | horse cart (10.5%) | pole (8.4%) |
| 35.95 | cinema (9.3%) | patio (8.0%) |

Multi-crop gave more consistent labels that match real objects in the scene (poles, parking meters, outdoor seating). The model still cannot recognize people or cars as the main content, because ImageNet has no "person" class.

## Requirements

Python 3 with `opencv-python`, `numpy`, `pandas`, and `matplotlib` (see `requirements.txt`). No TensorFlow or PyTorch is needed.
