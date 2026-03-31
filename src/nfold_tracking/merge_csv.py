#!/usr/bin/env python3
"""
Merge output.csv and joint_states.csv for a given test_videos folder.

Usage:
    python merge_csv.py --folder Aruco_1_01
    python merge_csv.py --folder Aruco_1_01 --input my_output.csv
"""
import sys
import argparse
import pandas as pd
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser(
        description='Merge output.csv and joint_states.csv on matching frame/img_filename')
    parser.add_argument('--folder', type=str,
                        help='Folder name inside test_videos/ (e.g. Aruco_1_01)')
    parser.add_argument('--input', type=str, default='output.csv',
                        help='Input pose CSV filename inside the folder (default: output.csv)')
    parser.add_argument('--output', type=str, default='merged.csv',
                        help='Output filename (default: merged.csv)')
    return parser.parse_args()


def main():
    args = parse_args()

    base = Path(__file__).parent / 'test_videos' / args.folder

    joint_path = base / 'joint_states.csv'
    output_path = base / args.input

    if not joint_path.exists():
        print(f"Error: {joint_path} not found")
        sys.exit(1)
    if not output_path.exists():
        print(f"Error: {output_path} not found (use --input to specify the pose CSV filename)")
        sys.exit(1)

    joint = pd.read_csv(joint_path)
    output = pd.read_csv(output_path)

    # Extract integer frame number from img_filename (e.g. "000822.png" -> 822)
    joint['frame'] = joint['img_filename'].str.replace('.png', '', regex=False).astype(int)

    # Keep only needed columns from joint_states
    joint_trim = joint[['frame', 'test_time_s', 'img_filename', 'position_rad']]

    # Inner join on frame — only rows with a match in both files
    merged = pd.merge(output, joint_trim, on='frame', how='inner')

    # Collect non-matching frames from each side
    output_frames = set(output['frame'])
    joint_frames = set(joint_trim['frame'])
    only_in_output = sorted(output_frames - joint_frames)
    only_in_joint = sorted(joint_frames - output_frames)

    # Reorder: joint metadata first, then pose data
    cols = ['frame', 'test_time_s', 'img_filename', 'position_rad',
            'time_s', 'tx', 'ty', 'tz', 'qx', 'qy', 'qz', 'qw']
    merged = merged[cols]

    out_path = base / args.output
    merged.to_csv(out_path, index=False)

    print(f"Rows in {args.input}:      {len(output)}")
    print(f"Rows in joint_states.csv:  {len(joint)}")
    print(f"Matched rows in merged:    {len(merged)}")
    print(f"Saved to: {out_path}")

    if only_in_output:
        print(f"\nFrames in {args.input} with no match in joint_states.csv ({len(only_in_output)}):")
        print(only_in_output)
    if only_in_joint:
        print(f"\nFrames in joint_states.csv with no match in {args.input} ({len(only_in_joint)}):")
        print(only_in_joint)
    if not only_in_output and not only_in_joint:
        print("\nAll frames matched.")


if __name__ == '__main__':
    main()
