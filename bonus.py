import os
import urllib.request

import cv2
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

VIDEO_PATH = "videos/video.mp4"
OUTPUT_DIR = "output/bonus_output"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "mobilenetv2-12.onnx")
LABELS_PATH = os.path.join(MODEL_DIR, "imagenet_classes.txt")

MODEL_URL = (
    "https://github.com/onnx/models/raw/main/validated/vision/"
    "classification/mobilenet/model/mobilenetv2-12.onnx"
)
LABELS_URL = (
    "https://raw.githubusercontent.com/pytorch/hub/master/"
    "imagenet_classes.txt"
)

NUM_FRAMES = 5          
TOP_PREDICTIONS = 3     
GRID_COLS = 3           
GRID_ROWS = 2           

# ImageNet normalization values
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Download the file only if it is not already there
def download_if_missing(url, path):
    if os.path.isfile(path):
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading {os.path.basename(path)} ...")
    urllib.request.urlretrieve(url, path)

# Resize to 224x224 and normalize for the model
def preprocess(rgb):
    img = cv2.resize(rgb, (224, 224)).astype(np.float32) / 255.0
    img = (img - MEAN) / STD
    img = np.transpose(img, (2, 0, 1))[np.newaxis, ...]
    return np.ascontiguousarray(img, dtype=np.float32)

# Turn the model scores into probabilities
def softmax(x):
    e = np.exp(x - np.max(x))
    return e / e.sum()

# Return the probability of each of the 1000 classes for one image
def classify(net, rgb):
    net.setInput(preprocess(rgb))
    return softmax(net.forward().flatten())

# Split the image into 3 x 2 tiles and return their boxes
def get_tiles(rgb):
    h, w = rgb.shape[:2]
    boxes = []
    for r in range(GRID_ROWS):
        for c in range(GRID_COLS):
            boxes.append((
                c * w // GRID_COLS, r * h // GRID_ROWS,
                (c + 1) * w // GRID_COLS, (r + 1) * h // GRID_ROWS,
            ))
    return boxes

# Write white text with a black outline
def draw_label(img, text, org, scale=1.1, thickness=3):
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, scale,
                (255, 255, 255), thickness, cv2.LINE_AA)

# analyze one frame: full frame, and multi-crop (average of tiles + full frame)
def analyze_frame(net, labels, frame):
    # Classify one frame: full frame, and multi-crop (average of tiles + full frame)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Full frame prediction
    full_probs = classify(net, rgb)
    full_top = int(np.argmax(full_probs))

    # Tile predictions (also drawn on a copy of the frame)
    tile_probs = []
    annotated = frame.copy()

    for (x1, y1, x2, y2) in get_tiles(rgb):
        probs = classify(net, rgb[y1:y2, x1:x2])
        tile_probs.append(probs)

        t_idx = int(np.argmax(probs))
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 255), 3)
        draw_label(annotated, f"{labels[t_idx]} {probs[t_idx] * 100:.0f}%",
                   (x1 + 15, y1 + 50))

    # Multi-crop: average all probabilities and take the top classes
    all_probs = np.mean(tile_probs + [full_probs], axis=0)
    top_idx = np.argsort(all_probs)[::-1][:TOP_PREDICTIONS]
    top3 = [(labels[k], float(all_probs[k])) for k in top_idx]

    return {
        "full_label": labels[full_top],
        "full_conf": float(full_probs[full_top]) * 100,
        "top3": top3,
        "annotated": annotated,
    }


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isfile(VIDEO_PATH):
        raise FileNotFoundError(f"Video not found: {VIDEO_PATH}")

    # Load the model and class names
    download_if_missing(MODEL_URL, MODEL_PATH)
    download_if_missing(LABELS_URL, LABELS_PATH)

    print("Loading MobileNetV2 model...")
    net = cv2.dnn.readNetFromONNX(MODEL_PATH)

    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = [line.strip() for line in f if line.strip()]

    # Open the video
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError("Could not open the video.")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps if fps > 0 else 0

    print("\nVIDEO INFORMATION")
    print(f"Video: {VIDEO_PATH}")
    print(f"Duration: {duration:.2f} seconds")
    print(f"FPS: {fps:.2f}")
    print(f"Total frames: {total_frames}")

    if total_frames <= 0:
        cap.release()
        raise RuntimeError("The video contains no readable frames.")

    # Pick 5 frames spread across the video
    frame_indices = np.linspace(0, total_frames - 1, NUM_FRAMES, dtype=int)

    results = []
    fig, axes = plt.subplots(1, NUM_FRAMES, figsize=(24, 6))

    # Classify each frame
    for i, frame_index in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        success, frame = cap.read()

        if not success:
            print(f"Could not read frame {frame_index}")
            axes[i].axis("off")
            continue

        timestamp = frame_index / fps if fps > 0 else 0

        analysis = analyze_frame(net, labels, frame)
        top3 = analysis["top3"]
        top_label, top_score = top3[0]
        confidence = top_score * 100
        full_label = analysis["full_label"]
        full_conf = analysis["full_conf"]

        results.append({
            "Frame": int(frame_index),
            "Timestamp (seconds)": round(timestamp, 2),
            "Multi-crop prediction": top_label,
            "Confidence (%)": round(confidence, 2),
            "Second prediction": top3[1][0],
            "Third prediction": top3[2][0],
            "Full-frame prediction": full_label,
            "Full-frame confidence (%)": round(full_conf, 2),
        })

        print(f"\nFrame {frame_index} | Time: {timestamp:.2f}s")
        print(f"Full frame : {full_label} ({full_conf:.2f}%)")
        print("Multi-crop (average of tiles + full frame):")
        for rank, (label, score) in enumerate(top3, start=1):
            print(f"  {rank}. {label}: {score * 100:.2f}%")

        # Show the frame with its predictions
        axes[i].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        axes[i].set_title(
            f"{timestamp:.1f}s\n"
            f"Multi-crop: {top_label} ({confidence:.1f}%)\n"
            f"Full frame: {full_label} ({full_conf:.1f}%)",
            fontsize=9,
        )
        axes[i].axis("off")

        # Save the frame and its tile view
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{i + 1}.jpg"), frame)
        cv2.imwrite(os.path.join(OUTPUT_DIR, f"frame_{i + 1}_tiles.jpg"),
                    analysis["annotated"])

    cap.release()

    # Save the figure and the CSV table
    plt.tight_layout()
    image_path = os.path.join(OUTPUT_DIR, "bonus_predictions.png")
    plt.savefig(image_path, dpi=150, bbox_inches="tight")
    plt.close(fig)

    df = pd.DataFrame(results)
    csv_path = os.path.join(OUTPUT_DIR, "bonus_predictions.csv")
    df.to_csv(csv_path, index=False)

    print("\nCLASSIFICATION RESULTS")
    print("-" * 40)
    print(df.to_string(index=False))

    print("\nBONUS TASK COMPLETED")
    print(f"Visualization: {image_path}")
    print(f"CSV results: {csv_path}")
    print(f"Tile views: {OUTPUT_DIR}/frame_N_tiles.jpg")
    print(f"Output folder: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()