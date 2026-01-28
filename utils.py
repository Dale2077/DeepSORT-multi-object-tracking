"""
Utility functions for visualization
"""
import cv2
import numpy as np


def draw_boxes(image, bbox_list, identities=None, offset=(0, 0)):
    """
    Draw bounding boxes on image
    
    Args:
        image: numpy array (BGR format)
        bbox_list: list of bounding boxes in format [x1, y1, x2, y2]
        identities: list of track IDs
        offset: offset for drawing
    
    Returns:
        image with bounding boxes drawn
    """
    for i, box in enumerate(bbox_list):
        x1, y1, x2, y2 = [int(i) for i in box]
        x1 += offset[0]
        x2 += offset[0]
        y1 += offset[1]
        y2 += offset[1]
        
        # Get track ID
        if identities is not None:
            track_id = int(identities[i]) if i < len(identities) else 0
        else:
            track_id = 0
        
        # Generate color based on ID
        color = compute_color_for_id(track_id)
        
        # Draw bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
        
        # Draw label
        label = f'ID: {track_id}'
        label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
        y1_label = max(y1, label_size[1])
        cv2.rectangle(image, (x1, y1_label - label_size[1]), 
                     (x1 + label_size[0], y1_label + base_line), color, cv2.FILLED)
        cv2.putText(image, label, (x1, y1_label), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.5, (255, 255, 255), 2)
    
    return image


def compute_color_for_id(label):
    """
    Generate unique color for each ID
    
    Args:
        label: track ID
        
    Returns:
        color: BGR color tuple
    """
    palette = (2 ** 11 - 1, 2 ** 15 - 1, 2 ** 20 - 1)
    color = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
    return tuple(color)
