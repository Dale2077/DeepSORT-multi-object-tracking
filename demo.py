"""
Demo script to show DeepSORT tracking on a synthetic video
This creates a simple demo without needing an actual video file
"""
import cv2
import numpy as np
from detector import YOLOv5Detector
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from feature_extractor import Extractor
from utils import draw_boxes


def create_demo_frame(frame_num, width=640, height=480):
    """Create a synthetic demo frame with moving objects"""
    # Create a blank frame
    frame = np.ones((height, width, 3), dtype=np.uint8) * 240
    
    # Add title
    cv2.putText(frame, 'DeepSORT Tracking Demo', (width//2 - 150, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    
    # Add some moving "objects" (colored rectangles)
    objects = []
    
    # Object 1: moves horizontally
    x1 = (frame_num * 3) % width
    y1 = 100
    bbox1 = [x1, y1, x1 + 50, y1 + 100]
    cv2.rectangle(frame, (int(bbox1[0]), int(bbox1[1])), 
                 (int(bbox1[2]), int(bbox1[3])), (255, 0, 0), -1)
    objects.append(bbox1)
    
    # Object 2: moves vertically
    x2 = 200
    y2 = (frame_num * 2) % (height - 100)
    bbox2 = [x2, y2, x2 + 50, y2 + 100]
    cv2.rectangle(frame, (int(bbox2[0]), int(bbox2[1])), 
                 (int(bbox2[2]), int(bbox2[3])), (0, 255, 0), -1)
    objects.append(bbox2)
    
    # Object 3: moves diagonally
    x3 = (frame_num * 2) % (width - 50)
    y3 = (frame_num * 1.5) % (height - 100)
    bbox3 = [x3, y3, x3 + 50, y3 + 100]
    cv2.rectangle(frame, (int(bbox3[0]), int(bbox3[1])), 
                 (int(bbox3[2]), int(bbox3[3])), (0, 0, 255), -1)
    objects.append(bbox3)
    
    return frame, objects


def demo_tracking():
    """Run a simple tracking demo"""
    print("=" * 60)
    print("DeepSORT Multi-Object Tracking - Demo")
    print("=" * 60)
    print("\nThis demo shows DeepSORT tracking on synthetic moving objects.")
    print("Press 'q' to quit.\n")
    
    # Initialize components
    print("Initializing components...")
    extractor = Extractor(use_cuda=False)
    metric = NearestNeighborDistanceMetric('cosine', 0.2, 100)
    tracker = Tracker(metric, max_age=30, n_init=3)
    
    # Video parameters
    width, height = 640, 480
    fps = 30
    
    # Create output video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('demo_output.avi', fourcc, fps, (width, height))
    
    print("Running tracking demo...")
    
    num_frames = 300  # 10 seconds at 30 FPS
    
    for frame_num in range(num_frames):
        # Create synthetic frame with moving objects
        frame, objects = create_demo_frame(frame_num, width, height)
        
        # Extract features for objects
        features = extractor.extract(frame, objects)
        
        # Create Detection objects
        detection_list = []
        for i, bbox in enumerate(objects):
            x1, y1, x2, y2 = bbox
            tlwh = np.array([x1, y1, x2 - x1, y2 - y1])
            confidence = 0.9
            feature = features[i] if len(features) > 0 else np.zeros(128)
            detection_list.append(Detection(tlwh, confidence, feature))
        
        # Update tracker
        tracker.predict()
        tracker.update(detection_list)
        
        # Draw tracking results
        bbox_xyxy = []
        identities = []
        for track in tracker.tracks:
            if not track.is_confirmed() or track.time_since_update > 1:
                continue
            bbox = track.to_tlbr()
            bbox_xyxy.append(bbox)
            identities.append(track.track_id)
        
        if len(bbox_xyxy) > 0:
            frame = draw_boxes(frame, bbox_xyxy, identities)
        
        # Add frame info
        cv2.putText(frame, f'Frame: {frame_num}/{num_frames}', (10, height - 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        cv2.putText(frame, f'Active Tracks: {len(identities)}', (10, height - 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
        
        # Write frame
        out.write(frame)
        
        # Display frame
        cv2.imshow('DeepSORT Demo', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        if (frame_num + 1) % 30 == 0:
            print(f"  Processed {frame_num + 1}/{num_frames} frames, Active tracks: {len(identities)}")
    
    # Cleanup
    out.release()
    cv2.destroyAllWindows()
    
    print("\n" + "=" * 60)
    print("✓ Demo completed successfully!")
    print(f"  - Total frames: {num_frames}")
    print(f"  - Output saved to: demo_output.avi")
    print("=" * 60)


if __name__ == '__main__':
    demo_tracking()
