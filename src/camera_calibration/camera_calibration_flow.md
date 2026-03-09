# Camera Calibration System - Data Flow

```mermaid
graph TB
    subgraph "1. Video Capture - CaptureVideo.py"
        A1[Camera Streams<br/>UDP RTP H.264] --> A2[DualVideoCapture]
        A2 --> A3{Pipeline Type}
        A3 --> A4[Preview Pipeline<br/>Countdown Display]
        A3 --> A5[Recording Pipeline<br/>H.264 → MP4]
        A4 --> A6[Frame Synchronization<br/>timestamp-based]
        A5 --> A6
        A6 --> A7[Synchronized Videos<br/>cam_left_*.mp4<br/>cam_right_*.mp4]
    end

    subgraph "2. Frame Extraction - video_processing.py"
        B1[Input Videos<br/>MP4 files] --> B2[VideoProcessor]
        B2 --> B3[Extract Frames<br/>every 30th frame]
        B3 --> B4{ChArUco Board<br/>Detection}
        B4 -->|Detected<br/>≥15 corners| B5[Save Frame<br/>as JPEG]
        B4 -->|Not Detected| B6[Skip Frame]
        B5 --> B7[Frame Images<br/>frame_0000.jpeg<br/>frame_0001.jpeg<br/>...]
    end

    subgraph "3. Calibration - calibration.py"
        C1[Frame Images] --> C2{Calibration Type}

        C2 -->|Monocular| C3[Calibrator]
        C3 --> C4[Detect ChArUco<br/>Corners & IDs]
        C4 --> C5{Valid Corners?<br/>≥30 corners}
        C5 -->|Yes| C6[Collect Corner Data]
        C5 -->|No| C7[Skip Image]
        C6 --> C8[Camera Calibration<br/>calibrateCameraCharuco]
        C8 --> C9[Camera Matrix<br/>Distortion Coeffs<br/>RMS Error]
        C9 --> C10[Save Calibration<br/>*.npz file]

        C2 -->|Stereo| C11[StereoCalibrator]
        C11 --> C12[Load Left/Right<br/>Calibration Data]
        C12 --> C13[Detect Stereo<br/>ChArUco Corners]
        C13 --> C14{Find Common<br/>IDs ≥10?}
        C14 -->|Yes| C15[Match Corner Pairs]
        C14 -->|No| C16[Skip Image Pair]
        C15 --> C17[Stereo Calibration<br/>stereoCalibrate]
        C17 --> C18[Rotation Matrix R<br/>Translation Vector T<br/>Essential/Fundamental<br/>Matrices E, F]
        C18 --> C19[Save Stereo<br/>Calibration *.npz]
    end

    A7 --> B1
    B7 --> C1

    style A7 fill:#e1f5ff
    style B7 fill:#e1f5ff
    style C10 fill:#c8e6c9
    style C19 fill:#c8e6c9
```

## Component Details

### 1. CaptureVideo.py - DualVideoCapture Class
**Purpose**: Capture synchronized video streams from multiple cameras

**Key Features**:
- Receives H.264 video via UDP/RTP from multiple camera ports
- Uses GStreamer pipelines for video processing
- Two-phase approach: preview pipeline (countdown) → recording pipeline (actual recording)
- Frame synchronization using timestamps with 33ms tolerance
- Outputs timestamped MP4 files for each camera

**Data Flow**:
- Input: UDP RTP streams (H.264) from cameras
- Output: Synchronized MP4 video files (`cam_name_timestamp.mp4`)

---

### 2. video_processing.py - VideoProcessor Class
**Purpose**: Extract calibration frames from recorded videos

**Key Features**:
- Processes MP4 videos frame-by-frame
- Detects ChArUco board presence in each frame
- Extracts frames at regular intervals (every 30th frame / 1 second at 30fps)
- Only saves frames where ChArUco board is detected with ≥15 corners
- Clears output directory before extraction

**Data Flow**:
- Input: MP4 video files
- Output: JPEG images containing ChArUco boards (`frame_XXXX.jpeg`)

---

### 3. calibration.py - Calibrator & StereoCalibrator Classes
**Purpose**: Perform camera calibration using extracted frames

#### Calibrator (Monocular Calibration)
**Key Features**:
- Detects ChArUco markers in images
- Interpolates ChArUco corners from detected markers
- Requires ≥30 corners per image for valid calibration data
- Uses `aruco.calibrateCameraCharuco()` for camera calibration
- Outputs camera matrix, distortion coefficients, and RMS error

**Data Flow**:
- Input: Directory of JPEG images with ChArUco boards
- Output: NPZ file with camera calibration data (mtx, dist, rvecs, tvecs, rms_error)

#### StereoCalibrator (Stereo Calibration)
**Key Features**:
- Loads pre-calibrated left and right camera data
- Detects ChArUco corners in synchronized image pairs
- Matches common corner IDs between left/right images (requires ≥10 common IDs)
- Uses `cv.stereoCalibrate()` with fixed intrinsics
- Outputs rotation (R), translation (T), essential (E), and fundamental (F) matrices

**Data Flow**:
- Input: Left/right camera calibration data + stereo image pairs
- Output: NPZ file with stereo calibration data (R, T, E, F, stereo_error)

---

## Typical Workflow

1. **Record Videos**: Use CaptureVideo.py to capture synchronized videos from stereo camera setup
2. **Extract Frames**: Run video_processing.py to extract frames containing ChArUco boards
3. **Monocular Calibration**: Use Calibrator to calibrate each camera individually
4. **Stereo Calibration**: Use StereoCalibrator to compute stereo relationship between cameras
