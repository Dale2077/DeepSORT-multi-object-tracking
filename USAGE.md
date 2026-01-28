# DeepSORT Multi-Object Tracking - Usage Examples

This document provides detailed examples of how to use the DeepSORT multi-object tracking system.

## Basic Usage

### Track Objects in a Video File

```bash
python track.py --source path/to/video.mp4 --output tracked_output.avi --display
```

### Track Objects from Webcam

```bash
python track.py --source 0 --output webcam_tracking.avi --display
```

### Run Without Display (Background Processing)

```bash
python track.py --source video.mp4 --output output.avi
```

## Advanced Usage

### Use Different YOLOv5 Models

#### YOLOv5s (Small, Fast)
```bash
python track.py --source video.mp4 --yolo-model yolov5s --display
```

#### YOLOv5m (Medium, Balanced)
```bash
python track.py --source video.mp4 --yolo-model yolov5m --display
```

#### YOLOv5l (Large, Accurate)
```bash
python track.py --source video.mp4 --yolo-model yolov5l --display
```

#### YOLOv5x (Extra Large, Most Accurate)
```bash
python track.py --source video.mp4 --yolo-model yolov5x --display
```

### Adjust Detection Thresholds

#### High Precision (Fewer False Positives)
```bash
python track.py --source video.mp4 --conf-thresh 0.6 --display
```

#### High Recall (More Detections)
```bash
python track.py --source video.mp4 --conf-thresh 0.3 --display
```

### Configure Tracking Parameters

#### Long-Term Tracking (Keep tracks longer)
```bash
python track.py --source video.mp4 --max-age 50 --display
```

#### Fast Confirmation (Quick track initialization)
```bash
python track.py --source video.mp4 --n-init 2 --display
```

#### Strict Confirmation (More reliable tracks)
```bash
python track.py --source video.mp4 --n-init 5 --display
```

### Feature Gallery Size

#### Large Gallery (Better matching, more memory)
```bash
python track.py --source video.mp4 --nn-budget 200 --display
```

#### Small Gallery (Less memory, faster)
```bash
python track.py --source video.mp4 --nn-budget 50 --display
```

### Force CPU or GPU

#### Force CPU Processing
```bash
python track.py --source video.mp4 --device cpu --display
```

#### Force CUDA (GPU) Processing
```bash
python track.py --source video.mp4 --device cuda --display
```

## Testing and Demo

### Run Component Tests
```bash
python test_implementation.py
```

### Run Demo with Synthetic Data
```bash
python demo.py
```

## Common Use Cases

### High-Quality Tracking for Analysis
```bash
python track.py \
  --source video.mp4 \
  --yolo-model yolov5l \
  --conf-thresh 0.5 \
  --max-age 40 \
  --n-init 3 \
  --nn-budget 150 \
  --output high_quality_tracking.avi \
  --display
```

### Real-Time Webcam Tracking
```bash
python track.py \
  --source 0 \
  --yolo-model yolov5s \
  --conf-thresh 0.4 \
  --max-age 30 \
  --output webcam_live.avi \
  --display
```

### Fast Processing for Large Videos
```bash
python track.py \
  --source large_video.mp4 \
  --yolo-model yolov5s \
  --conf-thresh 0.4 \
  --device cuda \
  --output fast_tracking.avi
```

### Crowded Scene Tracking
```bash
python track.py \
  --source crowded_scene.mp4 \
  --yolo-model yolov5x \
  --conf-thresh 0.3 \
  --max-age 20 \
  --n-init 4 \
  --nn-budget 200 \
  --output crowded_tracking.avi \
  --display
```

## Using the Code Programmatically

### Example: Custom Tracking Script

```python
import cv2
import numpy as np
from detector import YOLOv5Detector
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from feature_extractor import Extractor
from utils import draw_boxes

# Initialize components
detector = YOLOv5Detector(model_name='yolov5s', conf_thresh=0.4)
extractor = Extractor(use_cuda=True)
metric = NearestNeighborDistanceMetric('cosine', 0.2, 100)
tracker = Tracker(metric, max_age=30, n_init=3)

# Open video
cap = cv2.VideoCapture('video.mp4')

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    # Detect objects
    detections = detector.detect_person(frame)
    
    # Extract features
    bbox_list = detections[:, :4]
    features = extractor.extract(frame, bbox_list)
    
    # Create Detection objects
    detection_list = []
    for i, bbox in enumerate(bbox_list):
        x1, y1, x2, y2 = bbox
        tlwh = np.array([x1, y1, x2 - x1, y2 - y1])
        confidence = detections[i, 4]
        feature = features[i]
        detection_list.append(Detection(tlwh, confidence, feature))
    
    # Update tracker
    tracker.predict()
    tracker.update(detection_list)
    
    # Get tracking results
    for track in tracker.tracks:
        if not track.is_confirmed():
            continue
        bbox = track.to_tlbr()
        track_id = track.track_id
        # Use bbox and track_id for your application
        print(f"Track {track_id}: {bbox}")

cap.release()
```

## Tips and Best Practices

1. **Model Selection**: 
   - Use `yolov5s` for real-time applications
   - Use `yolov5l` or `yolov5x` for offline analysis where accuracy is critical

2. **Confidence Threshold**:
   - Higher values (0.5-0.7) reduce false positives
   - Lower values (0.3-0.4) catch more objects but may include false detections

3. **Max Age**:
   - Increase for scenarios with frequent occlusions
   - Decrease for fast-moving objects to avoid ghost tracks

4. **N Init**:
   - Higher values ensure more reliable tracks
   - Lower values provide faster track initialization

5. **NN Budget**:
   - Larger budgets improve re-identification after occlusions
   - Smaller budgets reduce memory usage and speed up matching

## Troubleshooting

### Issue: Low FPS
**Solution**: Use a smaller YOLOv5 model (yolov5s) or enable GPU with `--device cuda`

### Issue: Lost Tracks
**Solution**: Increase `--max-age` or decrease `--conf-thresh`

### Issue: Too Many False Tracks
**Solution**: Increase `--conf-thresh` or `--n-init`

### Issue: Memory Error
**Solution**: Reduce `--nn-budget` or use a smaller YOLOv5 model

## Output Format

The tracking output is a video file (`.avi`) with:
- Bounding boxes around tracked objects
- Unique ID for each track
- Frame counter
- Active track count

Press 'q' during playback (with `--display`) to stop early.
