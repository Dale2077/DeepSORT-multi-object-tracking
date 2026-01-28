"""
Main script for DeepSORT tracking with YOLOv5
"""
import argparse
import cv2
import numpy as np
from detector import YOLOv5Detector
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from feature_extractor import Extractor
from utils import draw_boxes


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(description='DeepSORT tracking with YOLOv5')
    parser.add_argument('--source', type=str, default='0', 
                       help='Video file path or camera index (0 for webcam)')
    parser.add_argument('--output', type=str, default='output.avi',
                       help='Output video file path')
    parser.add_argument('--yolo-model', type=str, default='yolov5s',
                       help='YOLOv5 model variant (yolov5s, yolov5m, yolov5l, yolov5x)')
    parser.add_argument('--conf-thresh', type=float, default=0.4,
                       help='Confidence threshold for detections')
    parser.add_argument('--iou-thresh', type=float, default=0.5,
                       help='IOU threshold for NMS')
    parser.add_argument('--max-age', type=int, default=30,
                       help='Maximum number of frames to keep track alive')
    parser.add_argument('--n-init', type=int, default=3,
                       help='Number of frames to confirm track')
    parser.add_argument('--nn-budget', type=int, default=100,
                       help='Maximum size of feature gallery')
    parser.add_argument('--display', action='store_true',
                       help='Display tracking results in real-time')
    parser.add_argument('--device', type=str, default='',
                       help='Device to run on (cuda or cpu)')
    
    return parser.parse_args()


def main():
    """Main tracking function"""
    args = parse_args()
    
    # Initialize detector
    print('Initializing YOLOv5 detector...')
    detector = YOLOv5Detector(
        model_name=args.yolo_model,
        device=args.device,
        conf_thresh=args.conf_thresh,
        iou_thresh=args.iou_thresh
    )
    
    # Initialize feature extractor
    print('Initializing feature extractor...')
    extractor = Extractor(use_cuda=(args.device != 'cpu'))
    
    # Initialize tracker
    print('Initializing DeepSORT tracker...')
    metric = NearestNeighborDistanceMetric('cosine', 0.2, args.nn_budget)
    tracker = Tracker(metric, max_age=args.max_age, n_init=args.n_init)
    
    # Open video source
    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)
    
    if not cap.isOpened():
        print(f'Error: Could not open video source {args.source}')
        return
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter(args.output, fourcc, fps, (width, height))
    
    print(f'Processing video: {width}x{height} @ {fps} FPS')
    print('Press "q" to quit')
    
    frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            
            # Detect objects
            detections = detector.detect_person(frame)
            
            # Extract features for detected objects
            bbox_list = detections[:, :4]
            features = extractor.extract(frame, bbox_list)
            
            # Create Detection objects
            detection_list = []
            for i, bbox in enumerate(bbox_list):
                x1, y1, x2, y2 = bbox
                # Convert to tlwh format
                tlwh = np.array([x1, y1, x2 - x1, y2 - y1])
                confidence = detections[i, 4]
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
            cv2.putText(frame, f'Frame: {frame_count}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f'Tracks: {len(identities)}', (10, 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Write frame
            out.write(frame)
            
            # Display
            if args.display:
                cv2.imshow('DeepSORT Tracking', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
            
            if frame_count % 30 == 0:
                print(f'Processed {frame_count} frames, Active tracks: {len(identities)}')
    
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
        
        print(f'\nTracking complete!')
        print(f'Total frames processed: {frame_count}')
        print(f'Output saved to: {args.output}')


if __name__ == '__main__':
    main()
