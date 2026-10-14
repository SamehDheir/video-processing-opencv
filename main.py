import json
import os

import cv2
import matplotlib.pyplot as plt
import numpy as np

VIDEO_PATH = "videos/video.mp4"
OUTPUT_DIR = "output"
FRAMES_DIR = os.path.join(OUTPUT_DIR, "extracted_frames")
PROCESSED_DIR = os.path.join(OUTPUT_DIR, "processed_frames")

INTERVAL_SECONDS = 0.5   
BRIGHTNESS_BETA = 40 

# Open the video file
def open_video(path):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open '{path}'.")
    return cap

# get the video information
def get_video_info(cap):
    # Read resolution, FPS, number of frames, and compute duration
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0 or total_frames <= 0:
        cap.release()
        raise ValueError("The video metadata is invalid.")

    return {
        "video_path": VIDEO_PATH,
        "width": width,
        "height": height,
        "fps": fps,
        "total_frames": total_frames,
        "duration_seconds": total_frames / fps,
    }

# Print the basic video information
def print_video_info(info):
    print("VIDEO INFORMATION")
    print(f"Resolution: {info['width']} x {info['height']}")
    print(f"Frame rate: {info['fps']:.2f} FPS")
    print(f"Total frames: {info['total_frames']}")
    print(f"Duration: {info['duration_seconds']:.2f} seconds")

# Extract frames, process them, and save results
def extract_frames(cap, fps, interval_seconds, frames_dir):
    # Save one frame every 0.5 seconds
    frame_interval = max(1, round(fps * interval_seconds))
    extracted_frames = []
    frame_index = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        if frame_index % frame_interval == 0:
            timestamp = frame_index / fps
            frame_path = os.path.join(frames_dir, f"frame_{frame_index:06d}.jpg")

            if cv2.imwrite(frame_path, frame):
                extracted_frames.append({
                    "frame_index": frame_index,
                    "timestamp": round(timestamp, 3),
                    "path": frame_path,
                })

        frame_index += 1

    if not extracted_frames:
        raise RuntimeError("No frames were extracted from the video")

    return extracted_frames

# save the extracted frames and their timestamps to a JSON file
def save_json(data, path):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, indent=4)

# visualize the extracted frames in a grid and save the figure
def visualize_extracted_frames(extracted_frames, output_path):
    count = min(6, len(extracted_frames))
    indices = np.linspace(0, len(extracted_frames) - 1, count, dtype=int)

    fig, axes = plt.subplots(2, 3, figsize=(15, 8))
    axes = axes.flatten()

    for ax, idx in zip(axes, indices):
        item = extracted_frames[idx]
        image = cv2.imread(item["path"])
        ax.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)) 
        ax.set_title(f"Time: {item['timestamp']:.2f} seconds")
        ax.axis("off")

    for ax in axes[count:]:
        ax.axis("off")

    fig.suptitle("Extracted Video Frames", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()

# process the selected frames by applying grayscale and brightness adjustments, and save the results
def process_selected_frames(extracted_frames, processed_dir):
    indices = np.linspace(
        0, len(extracted_frames) - 1, min(3, len(extracted_frames)), dtype=int
    )
    results = []

    for number, idx in enumerate(indices, start=1):
        item = extracted_frames[idx]
        original = cv2.imread(item["path"])

        grayscale = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
        bright = cv2.convertScaleAbs(original, alpha=1.0, beta=BRIGHTNESS_BETA)

        cv2.imwrite(os.path.join(processed_dir, f"frame_{number}_grayscale.jpg"), grayscale)
        cv2.imwrite(os.path.join(processed_dir, f"frame_{number}_brightness.jpg"), bright)

        results.append({
            "original": original,
            "grayscale": grayscale,
            "bright": bright,
            "timestamp": item["timestamp"],
        })

    return results

# plot the original, grayscale, and brightness-adjusted images side by side for comparison
def plot_processing_comparison(results, output_path):
    fig, axes = plt.subplots(len(results), 3, figsize=(15, 5 * len(results)))

    if len(results) == 1:
        axes = np.array([axes])

    for row, result in enumerate(results):
        axes[row, 0].imshow(cv2.cvtColor(result["original"], cv2.COLOR_BGR2RGB))
        axes[row, 0].set_title(f"Original ({result['timestamp']:.2f}s)")

        axes[row, 1].imshow(result["grayscale"], cmap="gray")
        axes[row, 1].set_title("Grayscale")

        axes[row, 2].imshow(cv2.cvtColor(result["bright"], cv2.COLOR_BGR2RGB))
        axes[row, 2].set_title(f"Brightness +{BRIGHTNESS_BETA}")

        for ax in axes[row]:
            ax.axis("off")

    fig.suptitle("Image Processing Results", fontsize=16)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.show()


def main():
    os.makedirs(FRAMES_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    # 1. Load the video and show its information
    cap = open_video(VIDEO_PATH)
    info = get_video_info(cap)
    print_video_info(info)

    # 2. Extract and save frames
    extracted_frames = extract_frames(cap, info["fps"], INTERVAL_SECONDS, FRAMES_DIR)
    cap.release()

    info["extracted_frames"] = len(extracted_frames)
    save_json(info, os.path.join(OUTPUT_DIR, "video_information.json"))
    save_json(extracted_frames, os.path.join(OUTPUT_DIR, "extracted_frames.json"))

    print(f"\nExtracted frames: {len(extracted_frames)}")
    print(f"Frames saved in: {FRAMES_DIR}")

    # 3. Visualize some extracted frames
    visualize_extracted_frames(
        extracted_frames,
        os.path.join(OUTPUT_DIR, "extracted_frames_visualization.png"),
    )

    # 4. Process 3 frames and compare with the originals
    results = process_selected_frames(extracted_frames, PROCESSED_DIR)
    plot_processing_comparison(
        results, os.path.join(OUTPUT_DIR, "processing_comparison.png")
    )

    print("\nImage processing completed successfully.")
    print(f"Processed images saved in: {PROCESSED_DIR}")
    print(f"All results saved in: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()