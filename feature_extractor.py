"""
Feature extractor for DeepSORT
"""
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import numpy as np
from PIL import Image


class FeatureExtractor(nn.Module):
    """
    Simple CNN feature extractor for person re-identification
    """
    
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        
        # Simple CNN architecture
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.fc = nn.Sequential(
            nn.Linear(128 * 16 * 8, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 128)
        )
        
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # L2 normalize
        x = nn.functional.normalize(x, p=2, dim=1)
        return x


class Extractor:
    """
    Wrapper for feature extraction
    """
    
    def __init__(self, model_path=None, use_cuda=True):
        self.device = torch.device('cuda' if use_cuda and torch.cuda.is_available() else 'cpu')
        
        self.model = FeatureExtractor()
        self.model.to(self.device)
        self.model.eval()
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((128, 64)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                               std=[0.229, 0.224, 0.225])
        ])
        
    def extract(self, image, bbox_list):
        """
        Extract features for detected bounding boxes
        
        Args:
            image: numpy array (BGR format)
            bbox_list: list of bounding boxes in format [x1, y1, x2, y2]
            
        Returns:
            features: numpy array of shape (N, 128)
        """
        if len(bbox_list) == 0:
            return np.array([])
        
        # Convert BGR to RGB
        image_rgb = image[..., ::-1]
        
        crops = []
        for bbox in bbox_list:
            x1, y1, x2, y2 = map(int, bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(image.shape[1], x2), min(image.shape[0], y2)
            
            crop = image_rgb[y1:y2, x1:x2]
            if crop.size == 0:
                # Invalid crop, use dummy feature
                crops.append(torch.zeros(3, 128, 64))
            else:
                crop_pil = Image.fromarray(crop)
                crop_tensor = self.transform(crop_pil)
                crops.append(crop_tensor)
        
        # Batch process
        crops_batch = torch.stack(crops).to(self.device)
        
        with torch.no_grad():
            features = self.model(crops_batch)
        
        return features.cpu().numpy()
