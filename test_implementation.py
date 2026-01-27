"""
Test script to verify DeepSORT implementation
"""
import numpy as np
import torch
from detector import YOLOv5Detector
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from deep_sort.nn_matching import NearestNeighborDistanceMetric
from feature_extractor import Extractor


def test_detector():
    """Test YOLOv5 detector initialization"""
    print("Testing YOLOv5 Detector...")
    try:
        detector = YOLOv5Detector(model_name='yolov5s', device='cpu', conf_thresh=0.4)
        print("✓ YOLOv5 detector initialized successfully")
        
        # Test detection on dummy image
        dummy_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        detections = detector.detect(dummy_image)
        print(f"✓ Detection works, found {len(detections)} objects in dummy image")
        
        return True
    except Exception as e:
        print(f"✗ Error in detector: {e}")
        return False


def test_feature_extractor():
    """Test feature extractor"""
    print("\nTesting Feature Extractor...")
    try:
        extractor = Extractor(use_cuda=False)
        print("✓ Feature extractor initialized successfully")
        
        # Test feature extraction
        dummy_image = np.random.randint(0, 255, (640, 480, 3), dtype=np.uint8)
        bbox_list = [[10, 10, 100, 200], [200, 200, 300, 400]]
        features = extractor.extract(dummy_image, bbox_list)
        print(f"✓ Feature extraction works, shape: {features.shape}")
        
        return True
    except Exception as e:
        print(f"✗ Error in feature extractor: {e}")
        return False


def test_tracker():
    """Test DeepSORT tracker"""
    print("\nTesting DeepSORT Tracker...")
    try:
        metric = NearestNeighborDistanceMetric('cosine', 0.2, 100)
        tracker = Tracker(metric, max_age=30, n_init=3)
        print("✓ Tracker initialized successfully")
        
        # Test with dummy detections
        detection1 = Detection(np.array([10, 10, 50, 100]), 0.9, np.random.randn(128))
        detection2 = Detection(np.array([200, 200, 50, 100]), 0.8, np.random.randn(128))
        
        tracker.predict()
        tracker.update([detection1, detection2])
        print(f"✓ Tracking works, active tracks: {len(tracker.tracks)}")
        
        return True
    except Exception as e:
        print(f"✗ Error in tracker: {e}")
        return False


def test_detection_class():
    """Test Detection class"""
    print("\nTesting Detection Class...")
    try:
        tlwh = np.array([10, 10, 50, 100])
        confidence = 0.9
        feature = np.random.randn(128)
        
        det = Detection(tlwh, confidence, feature)
        tlbr = det.to_tlbr()
        xyah = det.to_xyah()
        
        print(f"✓ Detection class works")
        print(f"  - TLWH: {det.tlwh}")
        print(f"  - TLBR: {tlbr}")
        print(f"  - XYAH: {xyah}")
        
        return True
    except Exception as e:
        print(f"✗ Error in detection class: {e}")
        return False


def main():
    """Run all tests"""
    print("=" * 60)
    print("DeepSORT Multi-Object Tracking - Component Tests")
    print("=" * 60)
    
    results = []
    
    results.append(("Detection Class", test_detection_class()))
    results.append(("Feature Extractor", test_feature_extractor()))
    results.append(("Tracker", test_tracker()))
    results.append(("YOLOv5 Detector", test_detector()))
    
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    
    for name, result in results:
        status = "PASS" if result else "FAIL"
        symbol = "✓" if result else "✗"
        print(f"{symbol} {name}: {status}")
    
    all_passed = all(result for _, result in results)
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ All tests passed! The implementation is working correctly.")
    else:
        print("✗ Some tests failed. Please check the errors above.")
    print("=" * 60)
    
    return all_passed


if __name__ == '__main__':
    success = main()
    exit(0 if success else 1)
