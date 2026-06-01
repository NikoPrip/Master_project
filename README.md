# Grader Blade Identification and Localization Using Computer Vision

Master thesis project — University of Southern Denmark (SDU), Maersk Mc-Kinney Moller Institute.

**Authors:** Johan Egtved Nissen & Nikolai Prip
**Supervisor:** Henrik Skov Midtiby

---

## Overview

This project implements a computer vision pipeline for real-time 6-DOF pose estimation of a grader blade relative to the machine chassis. The goal is to replace blade-mounted GNSS antenna masts with a passive fiducial marker, restoring the full range of motion of the blade while maintaining accurate position tracking.

A camera mounted on the grader chassis observes a fiducial marker attached to the blade. The estimated marker pose, combined with the chassis-mounted GNSS position, provides full knowledge of the blade pose in the world frame.

Three marker types are implemented and evaluated:

| Mode | Marker | PnP Solver |
|------|--------|------------|
| `aruco` | Single ArUco marker | IPPE + Levenberg-Marquardt |
| `nfold` | Constellation of 3-fold rotationally symmetric markers | SQPNP + Levenberg-Marquardt |
| `hybrid` | ArUco marker + Nfold constellation combined | SQPNP + Levenberg-Marquardt |

All three modes share a 6-DOF Kalman filter for temporal smoothing (constant-velocity translation + quaternion rotation kinematics).

---

## Project Structure

```
src/
├── nfold_tracking/         # Core pose estimation pipeline
│   ├── TrackingMain.py     # Entry point — run tracking from here
│   ├── ArucoPoseTracker.py
│   ├── NfoldPoseTracker.py
│   ├── HybridPoseTracker.py
│   ├── MarkerTracker.py    # Nfold convolution-based marker detector
│   ├── PoseKalmanFilter.py
│   ├── rotation_utils.py
│   ├── indoor_test/        # Configs for indoor lab tests
│   ├── outdoor_test/       # Configs for outdoor tests
│   ├── optitrack/          # Configs and analysis for OptiTrack validation
│   └── final_test/         # Configs and analysis for grader GNSS tests
├── camera_calibration/     # ChArUco-based camera calibration pipeline
└── gnss/                   # Leica iCA202 GNSS logger
RunCalibration.py           # Calibration entry point
Makefile                    # All run/log/capture targets
```

---

## Running the Tracker

```bash
python src/nfold_tracking/TrackingMain.py \
    --mode <aruco|nfold|hybrid> \
    --calib <indoor|outdoor|phone> \
    --config <indoor|outdoor|optitrack|final_test> \
    --video <path_to_video.mp4>
```

Or use the Makefile targets:

```bash
make aruco-indoor
make nfold-outdoor-1
make hybrid-optitrack-2
make log-all          # runs all 9 optitrack CSV logging jobs
make log-all-final    # runs all final test CSV logging jobs
```

To run on the grader (live capture + GNSS logging):

```bash
make capture NAME=aruco_1
make gnss-log NAME=aruco_1
make align NAME=aruco_1 MODE=aruco
```

---

## Camera Calibration

Uses a ChArUco board (ArUco DICT_5x5, 12×9 layout). Calibration files are stored in `src/camera_calibration/<dataset>/calib_files/`.

```bash
python RunCalibration.py --dataset <indoor|outdoor|phone|final_test>
```

---

## GNSS Logger

Connects to a Leica iCA202 NMEA TCP stream and logs position, heading and RTK fix quality to CSV.

```bash
python src/gnss/gnss_logger.py --host 192.168.151.2 --port 5001 --name <session_name>
```

---

## Dependencies

- Python 3.8+
- OpenCV (`opencv-contrib-python`) — ArUco and ChArUco support
- NumPy
- SciPy
- GStreamer (for live H.264 camera streams from Leica CRS360)

---

## GStreamer Camera Commands

The Leica CRS360 streams H.264 video over UDP. Camera IPs: `192.168.40.90` (port 58004) and `192.168.40.91` (port 58006).

**View stream (camera 90):**
```bash
gst-launch-1.0 -v udpsrc port=58004 buffer-size=212992 do-timestamp=true \
  ! application/x-rtp,media=video,clock-rate=90000,payload=96,encoding-name=H264 \
  ! rtph264depay ! avdec_h264 ! xvimagesink sync=true
```

**View and save to file:**
```bash
gst-launch-1.0 -v udpsrc port=58004 buffer-size=212992 do-timestamp=true \
  ! application/x-rtp,media=video,clock-rate=90000,payload=96,encoding-name=H264 \
  ! rtph264depay ! h264parse ! tee name=t \
    t. ! queue ! qtmux fragment-duration=1000 ! filesink location=output.mp4 \
    t. ! queue ! avdec_h264 ! xvimagesink sync=true
```

Replace port `58004` with `58006` for camera 91.

**Capture single image:**
```bash
gst-launch-1.0 -v udpsrc port=58004 buffer-size=212992 do-timestamp=true \
  ! application/x-rtp,media=video,clock-rate=90000,payload=96,encoding-name=H264 \
  ! rtph264depay ! avdec_h264 ! jpegenc ! filesink location=test.jpeg
```

---

## Source Code

[github.com/NikoPrip/Master_project](https://github.com/NikoPrip/Master_project)
