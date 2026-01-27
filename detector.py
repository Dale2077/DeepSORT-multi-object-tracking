"""
YOLOv5 Detector for object detection
"""
import torch
import numpy as np


class YOLOv5Detector:
    """
    YOLOv5 object detector wrapper
    """
    
    def __init__(self, model_name='yolov5s', device='', conf_thresh=0.4, iou_thresh=0.5):
        """
        Initialize YOLOv5 detector
        
        Args:
            model_name: YOLOv5 model variant (yolov5s, yolov5m, yolov5l, yolov5x)
            device: Device to run on ('cuda' or 'cpu')
            conf_thresh: Confidence threshold for detections
            iou_thresh: IOU threshold for NMS
        """
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        
        # Set device
        if device == '':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        # Load YOLOv5 model from torch hub
        self.model = torch.hub.load('ultralytics/yolov5', model_name, pretrained=True)
        self.model.to(self.device)
        self.model.eval()
        
        # Set model parameters
        self.model.conf = conf_thresh
        self.model.iou = iou_thresh
        
    def detect(self, image):
        """
        Detect objects in image
        
        Args:
            image: numpy array (BGR format from OpenCV)
            
        Returns:
            detections: numpy array of shape (N, 6) where each row is [x1, y1, x2, y2, conf, class]
        """
        # YOLOv5 expects RGB
        results = self.model(image[..., ::-1])
        
        # Get predictions
        predictions = results.xyxy[0].cpu().numpy()
        
        return predictions
    
    def detect_person(self, image):
        """
        Detect only person class (class 0 in COCO)
        
        Args:
            image: numpy array (BGR format from OpenCV)
            
        Returns:
            detections: numpy array of shape (N, 6) where each row is [x1, y1, x2, y2, conf, class]
        """
        detections = self.detect(image)
        
        # Filter for person class (class 0)
        person_detections = detections[detections[:, 5] == 0]
        
        return person_detections
