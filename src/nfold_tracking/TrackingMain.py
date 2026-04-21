#!/usr/bin/env python3
"""
Main Pose Tracking Runner

Run from terminal with mode selection:
    python TrackingMain.py --mode aruco
    python TrackingMain.py --mode hybrid
    python TrackingMain.py --mode nfold
"""
import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from HybridPoseTracker import HybridPoseTracker
from ArucoPoseTracker import ArucoPoseTracker
from NfoldPoseTracker import NfoldPoseTracker


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Run pose tracking with different marker types')
    parser.add_argument('--mode', type=str, choices=['aruco', 'hybrid', 'nfold'],
                       default='nfold', help='Tracking mode: aruco, hybrid, or nfold')
    parser.add_argument('--video90', type=str, default=None, help='Path to video files (optional)')
    parser.add_argument('--video91', type=str, default=None, help='Path to video files (optional)')
    parser.add_argument('--calib', type=str, choices=['indoor', 'outdoor', 'phone', 'Simulation_test', 'sim_cam_80_deg', 'sim_cam_actual_calib'], default='indoor', help='Path to calibration files (optional)')
    parser.add_argument('--config', type=str, default="indoor", choices=['indoor', 'outdoor', 'optitrack','Simulation_test'], help='Config file path (optional, default: indoor)')
    parser.add_argument('--output', type=str, default=None, help='Path to output CSV file for pose logging (optional)')
    return parser.parse_args()


def main():
    """Run pose tracker with configured settings."""
    args = parse_args()

    # Setup paths based on mode
    base_path = Path(__file__).parent
    calib_path = Path(__file__).parent.parent / 'camera_calibration'
    default_90 = {'aruco': 'Aruco_cam_90.mp4', 'hybrid': 'Hybrid_cam_90.mp4', 'nfold': 'Nfold_cam_90.mp4'}
    default_91 = {'aruco': 'Aruco_cam_91.mp4', 'hybrid': 'Hybrid_cam_91.mp4', 'nfold': 'Nfold_cam_91.mp4'}
    video_90 = args.video90 if args.video90 else str(base_path / 'test_videos' / default_90[args.mode])
    video_91 = args.video91 if args.video91 else str(base_path / 'test_videos' / default_91[args.mode])

    calib_file = str(calib_path / args.calib / 'calib_files')

    try:
        config_prefix = {'indoor': 'indoor_test', 'outdoor': 'outdoor_test', 'optitrack': 'optitrack', 'Simulation_test': 'Simulation_test'}[args.config]
        if args.mode == 'aruco':
            tracker = ArucoPoseTracker(calib_file, video_90, config_module=f'{config_prefix}.aruco_config', csv_path=args.output)
        elif args.mode == 'hybrid':
            tracker = HybridPoseTracker(calib_file, video_90, config_module=f'{config_prefix}.hybrid_config', csv_path=args.output)
        elif args.mode == 'nfold':
            tracker = NfoldPoseTracker(calib_file, video_90, config_module=f'{config_prefix}.nfold_config', csv_path=args.output)

        tracker.run()
        return 0

    except KeyboardInterrupt:
        return 0
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())