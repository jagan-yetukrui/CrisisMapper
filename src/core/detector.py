"""
Disaster Detection Module using YOLOv8.

This module provides the main disaster detection functionality using YOLOv8
for object detection and classification of disaster zones from satellite imagery.
"""

import os
import time
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union
import logging

import cv2
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
import yaml

from ..utils.config import load_config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class DisasterDetector:
    """
    Main disaster detection class using YOLOv8.
    
    This class handles the detection and classification of disaster zones
    from satellite and drone imagery using YOLOv8 models.
    """
    
    def __init__(self, config_path: Optional[str] = None, model_path: Optional[str] = None):
        """
        Initialize the DisasterDetector.
        
        Args:
            config_path: Path to configuration file
            model_path: Path to trained YOLOv8 model weights
        """
        self.config = load_config(config_path)
        self.model = None
        self.device = self._setup_device()
        self.classes = self.config['disaster_classes']
        self.class_names = [cls['name'] for cls in self.classes]
        
        # Performance tracking
        self.inference_times = []
        self.total_detections = 0
        
        # Load model
        self._load_model(model_path)
        
        logger.info(f"DisasterDetector initialized with device: {self.device}")
    
    def _setup_device(self) -> str:
        """Setup the computation device (CPU/GPU)."""
        if self.config['model']['device'] == 'auto':
            if torch.cuda.is_available():
                return 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return 'mps'
            else:
                return 'cpu'
        return self.config['model']['device']
    
    def _load_model(self, model_path: Optional[str] = None):
        """Load the YOLOv8 model."""
        if model_path is None:
            model_path = self.config['model']['weights_path']
        
        try:
            if not os.path.exists(model_path):
                logger.warning(f"Model weights not found at {model_path}. Downloading pretrained model...")
                model_name = self.config['model']['name']
                self.model = YOLO(f"{model_name}.pt")
                # Save the downloaded model
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                self.model.save(model_path)
            else:
                self.model = YOLO(model_path)
            
            # Move model to device
            self.model.to(self.device)
            logger.info(f"Model loaded successfully from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise
    
    def detect(self, 
               image: Union[str, np.ndarray, Image.Image], 
               save_results: bool = False,
               output_dir: Optional[str] = None) -> Dict:
        """
        Detect disaster zones in an image.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            save_results: Whether to save detection results
            output_dir: Directory to save results
            
        Returns:
            Dictionary containing detection results
        """
        start_time = time.time()
        
        # Preprocess image
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not load image from {image}")
        elif isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Run inference
        results = self.model(
            image,
            conf=self.config['model']['confidence_threshold'],
            iou=self.config['model']['iou_threshold'],
            max_det=self.config['model']['max_detections'],
            verbose=False
        )
        
        # Process results
        detections = self._process_detections(results[0], image.shape)
        
        # Calculate inference time
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        self.total_detections += len(detections['boxes'])
        
        # Create result dictionary
        result = {
            'image_shape': image.shape,
            'detections': detections,
            'inference_time': inference_time,
            'fps': 1.0 / inference_time if inference_time > 0 else 0,
            'model_info': {
                'name': self.config['model']['name'],
                'device': self.device,
                'confidence_threshold': self.config['model']['confidence_threshold']
            }
        }
        
        # Save results if requested
        if save_results and output_dir:
            self._save_results(image, result, output_dir)
        
        logger.info(f"Detection completed in {inference_time:.3f}s, found {len(detections['boxes'])} objects")
        
        return result
    
    def _process_detections(self, result, image_shape: Tuple[int, int, int]) -> Dict:
        """
        Process YOLOv8 detection results.
        
        Args:
            result: YOLOv8 result object
            image_shape: Shape of the input image (height, width, channels)
            
        Returns:
            Processed detection results
        """
        boxes = []
        confidences = []
        class_ids = []
        class_names = []
        
        if result.boxes is not None:
            for box in result.boxes:
                # Extract box coordinates (x1, y1, x2, y2)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                confidence = box.conf[0].cpu().numpy()
                class_id = int(box.cls[0].cpu().numpy())
                
                # Convert to relative coordinates if needed
                height, width = image_shape[:2]
                x1, y1, x2, y2 = x1/width, y1/height, x2/width, y2/height
                
                boxes.append([x1, y1, x2, y2])
                confidences.append(float(confidence))
                class_ids.append(class_id)
                
                # Get class name
                if class_id < len(self.class_names):
                    class_names.append(self.class_names[class_id])
                else:
                    class_names.append(f"class_{class_id}")
        
        return {
            'boxes': boxes,
            'confidences': confidences,
            'class_ids': class_ids,
            'class_names': class_names
        }
    
    def detect_batch(self, 
                    image_paths: List[str], 
                    output_dir: Optional[str] = None) -> List[Dict]:
        """
        Detect disaster zones in a batch of images.
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save results
            
        Returns:
            List of detection results for each image
        """
        results = []
        
        logger.info(f"Processing batch of {len(image_paths)} images")
        
        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"Processing image {i+1}/{len(image_paths)}: {image_path}")
                result = self.detect(image_path, save_results=True, output_dir=output_dir)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error processing {image_path}: {e}")
                results.append({
                    'error': str(e),
                    'image_path': image_path
                })
        
        return results
    
    def _save_results(self, image: np.ndarray, result: Dict, output_dir: str):
        """Save detection results and annotated image."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Create annotated image
        annotated_image = self._draw_detections(image, result['detections'])
        
        # Save annotated image
        timestamp = int(time.time())
        output_path = os.path.join(output_dir, f"detection_{timestamp}.jpg")
        cv2.imwrite(output_path, annotated_image)
        
        # Save detection data as JSON
        import json
        json_path = os.path.join(output_dir, f"detection_{timestamp}.json")
        with open(json_path, 'w') as f:
            json.dump(result, f, indent=2)
        
        logger.info(f"Results saved to {output_dir}")
    
    def _draw_detections(self, image: np.ndarray, detections: Dict) -> np.ndarray:
        """Draw bounding boxes and labels on image."""
        annotated = image.copy()
        height, width = image.shape[:2]
        
        for i, (box, conf, class_id, class_name) in enumerate(zip(
            detections['boxes'],
            detections['confidences'],
            detections['class_ids'],
            detections['class_names']
        )):
            # Convert relative coordinates to absolute
            x1 = int(box[0] * width)
            y1 = int(box[1] * height)
            x2 = int(box[2] * width)
            y2 = int(box[3] * height)
            
            # Get class color
            if class_id < len(self.classes):
                color = self.classes[class_id]['color']
            else:
                color = [0, 255, 0]  # Default green
            
            # Draw bounding box
            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {conf:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(annotated, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            cv2.putText(annotated, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return annotated
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        if not self.inference_times:
            return {}
        
        return {
            'total_inferences': len(self.inference_times),
            'total_detections': self.total_detections,
            'average_inference_time': np.mean(self.inference_times),
            'average_fps': 1.0 / np.mean(self.inference_times) if np.mean(self.inference_times) > 0 else 0,
            'min_inference_time': np.min(self.inference_times),
            'max_inference_time': np.max(self.inference_times),
            'std_inference_time': np.std(self.inference_times)
        }
    
    def reset_stats(self):
        """Reset performance statistics."""
        self.inference_times = []
        self.total_detections = 0
        logger.info("Performance statistics reset")
