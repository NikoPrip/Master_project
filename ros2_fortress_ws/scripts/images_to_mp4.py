#!/usr/bin/env python3
"""Convert a folder of sequentially-named images into an MP4 video."""

import argparse
import os
import sys

import cv2


def images_to_mp4(recording_dir: str, fps: float = 30.0, output_name: str = "output.mp4") -> None:
    images_dir = os.path.join(recording_dir, "images")
    if not os.path.isdir(images_dir):
        print(f"Error: images directory not found at '{images_dir}'", file=sys.stderr)
        sys.exit(1)

    image_files = sorted(
        [f for f in os.listdir(images_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))],
    )

    if not image_files:
        print(f"Error: no images found in '{images_dir}'", file=sys.stderr)
        sys.exit(1)

    first_frame = cv2.imread(os.path.join(images_dir, image_files[0]))
    if first_frame is None:
        print(f"Error: could not read '{image_files[0]}'", file=sys.stderr)
        sys.exit(1)

    height, width = first_frame.shape[:2]
    output_path = os.path.join(recording_dir, output_name)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    print(f"Writing {len(image_files)} frames ({width}x{height}) at {fps} fps -> {output_path}")

    for i, fname in enumerate(image_files):
        frame = cv2.imread(os.path.join(images_dir, fname))
        if frame is None:
            print(f"Warning: skipping unreadable frame '{fname}'", file=sys.stderr)
            continue
        writer.write(frame)
        if (i + 1) % 100 == 0:
            print(f"  {i + 1}/{len(image_files)}")

    writer.release()
    print("Done.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert sequential images to an MP4 video.")
    parser.add_argument("recording_dir", help="Path to the recording directory (must contain an 'images/' subfolder).")
    parser.add_argument("--fps", type=float, default=30.0, help="Frames per second (default: 30).")
    parser.add_argument("--output", default="output.mp4", help="Output filename inside recording_dir (default: output.mp4).")
    args = parser.parse_args()

    images_to_mp4(args.recording_dir, fps=args.fps, output_name=args.output)


if __name__ == "__main__":
    main()
