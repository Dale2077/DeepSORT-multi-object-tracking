# DeepSORT Multi-Object Tracking

Deep Simple Online and Realtime Tracking using YOLOv5.

## Overview

This repository implements DeepSORT (Deep Simple Online and Realtime Tracking) algorithm for multi-object tracking using YOLOv5 as the object detector. DeepSORT combines object detection with appearance feature extraction and Kalman filtering to track multiple objects across video frames.

## Features

- **YOLOv5 Integration**: Uses YOLOv5 for fast and accurate object detection
- **DeepSORT Tracking**: Robust multi-object tracking with appearance features
- **Real-time Processing**: Optimized for real-time tracking on videos and webcam
- **Person Detection**: Specifically configured for tracking people (can be extended to other classes)
- **Easy to Use**: Simple command-line interface

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Dale2077/DeepSORT-multi-object-tracking.git
cd DeepSORT-multi-object-tracking
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage

Track objects in a video file:
```bash
python track.py --source video.mp4 --output output.avi --display
```

Track objects from webcam:
```bash
python track.py --source 0 --output webcam_output.avi --display
```

### Command Line Arguments

- `--source`: Video file path or camera index (0 for webcam). Default: '0'
- `--output`: Output video file path. Default: 'output.avi'
- `--yolo-model`: YOLOv5 model variant (yolov5s, yolov5m, yolov5l, yolov5x). Default: 'yolov5s'
- `--conf-thresh`: Confidence threshold for detections. Default: 0.4
- `--iou-thresh`: IOU threshold for NMS. Default: 0.5
- `--max-age`: Maximum frames to keep track alive without detection. Default: 30
- `--n-init`: Number of frames to confirm a track. Default: 3
- `--nn-budget`: Maximum size of feature gallery for matching. Default: 100
- `--display`: Display tracking results in real-time
- `--device`: Device to run on ('cuda' or 'cpu'). Default: auto-detect

### Examples

High accuracy tracking with larger model:
```bash
python track.py --source video.mp4 --yolo-model yolov5l --conf-thresh 0.5 --display
```

Fast tracking with smaller model:
```bash
python track.py --source video.mp4 --yolo-model yolov5s --conf-thresh 0.3 --display
```

## Architecture

### Components

1. **YOLOv5 Detector** (`detector.py`): Object detection using YOLOv5
2. **Feature Extractor** (`feature_extractor.py`): CNN-based appearance feature extraction
3. **DeepSORT Tracker** (`deep_sort/`): Multi-object tracking implementation
   - Kalman Filter: State prediction and update
   - Hungarian Algorithm: Data association
   - Track Management: Track lifecycle management
4. **Visualization** (`utils.py`): Drawing utilities for bounding boxes and IDs

### Workflow

1. **Detection**: YOLOv5 detects objects in each frame
2. **Feature Extraction**: Extract appearance features from detected bounding boxes
3. **Prediction**: Kalman filter predicts track states
4. **Matching**: Associate detections with existing tracks using appearance and motion
5. **Update**: Update track states with matched detections
6. **Management**: Initialize new tracks, delete old tracks

## Requirements

- Python >= 3.7
- PyTorch >= 1.7.0
- OpenCV >= 4.5.0
- NumPy >= 1.19.0
- SciPy >= 1.5.0

See `requirements.txt` for complete list.

## References

- [DeepSORT Paper](https://arxiv.org/abs/1703.07402)
- [YOLOv5](https://github.com/ultralytics/yolov5)
- [SORT Paper](https://arxiv.org/abs/1602.00763)

## License

This project is for educational and research purposes.

