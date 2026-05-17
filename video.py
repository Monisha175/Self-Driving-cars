"""
video.py — Compile a folder of driving frames into an MP4 video.

Usage:
    python video.py run1              # creates run1.mp4 at default 60 FPS
    python video.py run1 --fps 48
"""

import argparse
import glob
import os

import cv2
from tqdm import tqdm


def frames_to_video(folder: str, fps: int = 60) -> str:
    """
    Read all JPEG frames from `folder` (sorted by filename/timestamp),
    compile into an MP4 at `fps` frames per second.

    Returns:
        Path to the output MP4 file.
    """
    frame_paths = sorted(glob.glob(os.path.join(folder, "*.jpg")))
    if not frame_paths:
        raise FileNotFoundError(f"No .jpg frames found in '{folder}'")

    # Read first frame to get resolution
    sample = cv2.imread(frame_paths[0])
    h, w   = sample.shape[:2]

    output_path = f"{folder.rstrip('/')}.mp4"
    fourcc      = cv2.VideoWriter_fourcc(*"mp4v")
    writer      = cv2.VideoWriter(output_path, fourcc, fps, (w, h))

    print(f"Compiling {len(frame_paths)} frames → {output_path}  ({fps} FPS)")
    for path in tqdm(frame_paths, unit="frame"):
        frame = cv2.imread(path)
        writer.write(frame)

    writer.release()
    size_mb = os.path.getsize(output_path) / 1e6
    print(f"Done. {output_path} ({size_mb:.1f} MB)")
    return output_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", help="Folder containing recorded .jpg frames")
    parser.add_argument("--fps",  type=int, default=60, help="Output FPS (default 60)")
    args = parser.parse_args()
    frames_to_video(args.folder, args.fps)
