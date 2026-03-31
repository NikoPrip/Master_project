#!/usr/bin/env python3
"""
Plot joint position over time from a recorded joint_states.csv file.

Usage:
    python3 scripts/plot_joint_state.py --csv scripts/recordings/<timestamp>/joint_states.csv
    python3 scripts/plot_joint_state.py  # auto-picks the most recent recording
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import argparse
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import config.initial_link_poses as Grader_blade_measurements

def find_latest_csv(recordings_dir: str) -> str:
    """Return the joint_states.csv from the most recently created recording folder."""
    folders = sorted(
        [f for f in os.listdir(recordings_dir)
         if os.path.isdir(os.path.join(recordings_dir, f))],
        reverse=True,
    )
    for folder in folders:
        candidate = os.path.join(recordings_dir, folder, "joint_states.csv")
        if os.path.isfile(candidate):
            return candidate
    raise FileNotFoundError(f"No joint_states.csv found in {recordings_dir}")




def main():
    default_recordings = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "test_data")

    parser = argparse.ArgumentParser(description="Plot joint position vs time from CSV.")
    parser.add_argument(
        "--csv",
        default=None,
        help="Path to joint_states.csv (default: most recent recording)",
    )
    args = parser.parse_args()

    csv_path = args.csv if args.csv else find_latest_csv(default_recordings)
    print(f"Reading: {csv_path}")

    df = pd.read_csv(csv_path)

    if "ros_time_s" not in df.columns or "position_rad" not in df.columns:
        raise ValueError("CSV must contain 'ros_time_s' and 'position_rad' columns.")

    # Drop rows with missing position data
    df = df.dropna(subset=["ros_time_s", "position_rad"])

    # Offset time so recording starts at t=0
    t0 = df["ros_time_s"].iloc[0]
    df["time_s"] = df["ros_time_s"] - t0

    joint_name = df["joint_name"].iloc[0] if "joint_name" in df.columns else "joint"

    # ── Plot ─────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.plot(df["time_s"].to_numpy(), df["position_rad"].to_numpy(),
            linewidth=1.5, color="royalblue", label="Actual position")

    # Plot commanded position if available in this recording
    if "commanded_position_rad" in df.columns:
        cmd = df["commanded_position_rad"].to_numpy()
        ax.plot(df["time_s"].to_numpy(), cmd,
                linewidth=1.5, color="orangered", linestyle="--", label="Commanded position")

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Position (rad)", fontsize=12)
    ax.set_title(f"Joint position over time — {joint_name}", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.6)
    ax.set_xlim(left=0)
    ax.legend(fontsize=10)

    # Annotate start/end values
#    ax.annotate(
#        f"start: {df['position_rad'].iloc[0]:.3f} rad",
#        xy=(df["time_s"].iloc[0], df["position_rad"].iloc[0]),
#        xytext=(df["time_s"].iloc[0] + 0.2, df["position_rad"].iloc[0]),
#        fontsize=9, color="green",
#    )
#    ax.annotate(
#        f"end: {df['position_rad'].iloc[-1]:.3f} rad",
#        xy=(df["time_s"].iloc[-1], df["position_rad"].iloc[-1]),
#        xytext=(df["time_s"].iloc[-1] - 1.5, df["position_rad"].iloc[-1]),
#        fontsize=9, color="red",
#    )

    plt.tight_layout()

    # Save next to the CSV
    #out_path = os.path.join(os.path.dirname(csv_path), "joint_position_plot.png")
    #plt.savefig(out_path, dpi=150)
    #print(f"Plot saved to: {out_path}")

    plt.show()


if __name__ == "__main__":
    main()
