"""
Disaster Classification Module.

This module provides advanced classification capabilities for disaster zones,
including severity assessment and confidence scoring.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
import logging
from sklearn.metrics import classification_report, confusion_matrix
import json

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class DisasterClassifier:
    """
    Advanced disaster classification with severity assessment.
    
    This class provides enhanced classification capabilities beyond basic
    YOLOv8 detection, including severity assessment and confidence scoring.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the DisasterClassifier.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.classes = config['disaster_classes']
        self.class_names = [cls['name'] for cls in self.classes]
        
        # Severity thresholds (can be adjusted based on training data)
        self.severity_thresholds = {
            'flood': {'low': 0.3, 'medium': 0.6, 'high': 0.8},
            'wildfire': {'low': 0.4, 'medium': 0.7, 'high': 0.85},
            'earthquake': {'low': 0.35, 'medium': 0.65, 'high': 0.8},
            'landslide': {'low': 0.3, 'medium': 0.6, 'high': 0.8},
            'hurricane': {'low': 0.4, 'medium': 0.7, 'high': 0.85}
        }
        
        logger.info("DisasterClassifier initialized")
    
    def classify_detection(self, detection: Dict) -> Dict:
        """
        Classify a single detection with severity assessment.
        
        Args:
            detection: Detection result from DisasterDetector
            
        Returns:
            Enhanced classification result
        """
        enhanced_detections = []
        
        for i, (box, conf, class_id, class_name) in enumerate(zip(
            detection['detections']['boxes'],
            detection['detections']['confidences'],
            detection['detections']['class_ids'],
            detection['detections']['class_names']
        )):
            # Calculate severity
            severity = self._calculate_severity(class_name, conf)
            
            # Calculate area and coverage
            area = self._calculate_area(box)
            coverage_percentage = self._calculate_coverage(box, detection['image_shape'])
            
            # Enhanced detection
            enhanced_detection = {
                'box': box,
                'confidence': conf,
                'class_id': class_id,
                'class_name': class_name,
                'severity': severity,
                'area': area,
                'coverage_percentage': coverage_percentage,
                'risk_score': self._calculate_risk_score(class_name, conf, area)
            }
            
            enhanced_detections.append(enhanced_detection)
        
        return {
            'original_detection': detection,
            'enhanced_detections': enhanced_detections,
            'summary': self._generate_summary(enhanced_detections)
        }
    
    def _calculate_severity(self, class_name: str, confidence: float) -> str:
        """
        Calculate severity level based on confidence and class type.
        
        Args:
            class_name: Name of the disaster class
            confidence: Detection confidence score
            
        Returns:
            Severity level: 'low', 'medium', or 'high'
        """
        if class_name not in self.severity_thresholds:
            class_name = 'flood'  # Default to flood thresholds
        
        thresholds = self.severity_thresholds[class_name]
        
        if confidence >= thresholds['high']:
            return 'high'
        elif confidence >= thresholds['medium']:
            return 'medium'
        else:
            return 'low'
    
    def _calculate_area(self, box: List[float]) -> float:
        """
        Calculate the area of a bounding box.
        
        Args:
            box: Bounding box coordinates [x1, y1, x2, y2] (relative coordinates)
            
        Returns:
            Area of the bounding box
        """
        x1, y1, x2, y2 = box
        return (x2 - x1) * (y2 - y1)
    
    def _calculate_coverage(self, box: List[float], image_shape: Tuple[int, int, int]) -> float:
        """
        Calculate the percentage of image covered by the bounding box.
        
        Args:
            box: Bounding box coordinates [x1, y1, x2, y2] (relative coordinates)
            image_shape: Shape of the image (height, width, channels)
            
        Returns:
            Coverage percentage
        """
        area = self._calculate_area(box)
        total_area = image_shape[0] * image_shape[1]
        return (area * total_area) / total_area * 100
    
    def _calculate_risk_score(self, class_name: str, confidence: float, area: float) -> float:
        """
        Calculate a composite risk score for the detection.
        
        Args:
            class_name: Name of the disaster class
            confidence: Detection confidence score
            area: Area of the detection
            
        Returns:
            Risk score between 0 and 1
        """
        # Base risk from confidence
        base_risk = confidence
        
        # Area factor (larger areas are higher risk)
        area_factor = min(area * 2, 1.0)  # Cap at 1.0
        
        # Class-specific risk weights
        class_weights = {
            'flood': 0.9,
            'wildfire': 1.0,
            'earthquake': 0.95,
            'landslide': 0.85,
            'hurricane': 1.0
        }
        
        class_weight = class_weights.get(class_name, 0.8)
        
        # Composite risk score
        risk_score = (base_risk * 0.6 + area_factor * 0.4) * class_weight
        
        return min(risk_score, 1.0)
    
    def _generate_summary(self, enhanced_detections: List[Dict]) -> Dict:
        """
        Generate a summary of all detections.
        
        Args:
            enhanced_detections: List of enhanced detection results
            
        Returns:
            Summary statistics
        """
        if not enhanced_detections:
            return {
                'total_detections': 0,
                'class_counts': {},
                'severity_counts': {'low': 0, 'medium': 0, 'high': 0},
                'average_confidence': 0.0,
                'average_risk_score': 0.0,
                'total_coverage': 0.0
            }
        
        # Count by class
        class_counts = {}
        for detection in enhanced_detections:
            class_name = detection['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        # Count by severity
        severity_counts = {'low': 0, 'medium': 0, 'high': 0}
        for detection in enhanced_detections:
            severity = detection['severity']
            severity_counts[severity] += 1
        
        # Calculate averages
        confidences = [det['confidence'] for det in enhanced_detections]
        risk_scores = [det['risk_score'] for det in enhanced_detections]
        coverages = [det['coverage_percentage'] for det in enhanced_detections]
        
        return {
            'total_detections': len(enhanced_detections),
            'class_counts': class_counts,
            'severity_counts': severity_counts,
            'average_confidence': np.mean(confidences),
            'average_risk_score': np.mean(risk_scores),
            'total_coverage': np.sum(coverages),
            'max_risk_score': np.max(risk_scores),
            'min_risk_score': np.min(risk_scores)
        }
    
    def evaluate_classification(self, 
                              predictions: List[Dict], 
                              ground_truth: List[Dict]) -> Dict:
        """
        Evaluate classification performance against ground truth.
        
        Args:
            predictions: List of prediction results
            ground_truth: List of ground truth annotations
            
        Returns:
            Evaluation metrics
        """
        # Extract predicted and true labels
        y_pred = []
        y_true = []
        
        for pred, gt in zip(predictions, ground_truth):
            pred_classes = pred['detections']['class_names']
            true_classes = gt['detections']['class_names']
            
            y_pred.extend(pred_classes)
            y_true.extend(true_classes)
        
        # Calculate metrics
        report = classification_report(y_true, y_pred, output_dict=True)
        cm = confusion_matrix(y_true, y_pred, labels=self.class_names)
        
        # Calculate additional metrics
        accuracy = report['accuracy']
        macro_avg = report['macro avg']
        weighted_avg = report['weighted avg']
        
        return {
            'accuracy': accuracy,
            'macro_avg': macro_avg,
            'weighted_avg': weighted_avg,
            'confusion_matrix': cm.tolist(),
            'class_names': self.class_names,
            'detailed_report': report
        }
    
    def generate_classification_report(self, evaluation_results: Dict) -> str:
        """
        Generate a human-readable classification report.
        
        Args:
            evaluation_results: Results from evaluate_classification
            
        Returns:
            Formatted report string
        """
        report = "=== Disaster Classification Report ===\n\n"
        
        # Overall accuracy
        report += f"Overall Accuracy: {evaluation_results['accuracy']:.3f}\n\n"
        
        # Per-class metrics
        report += "Per-Class Metrics:\n"
        report += "-" * 50 + "\n"
        
        for class_name in self.class_names:
            if class_name in evaluation_results['detailed_report']:
                metrics = evaluation_results['detailed_report'][class_name]
                report += f"{class_name.upper()}:\n"
                report += f"  Precision: {metrics['precision']:.3f}\n"
                report += f"  Recall: {metrics['recall']:.3f}\n"
                report += f"  F1-Score: {metrics['f1-score']:.3f}\n"
                report += f"  Support: {metrics['support']}\n\n"
        
        # Macro averages
        macro = evaluation_results['macro_avg']
        report += f"Macro Average:\n"
        report += f"  Precision: {macro['precision']:.3f}\n"
        report += f"  Recall: {macro['recall']:.3f}\n"
        report += f"  F1-Score: {macro['f1-score']:.3f}\n\n"
        
        # Weighted averages
        weighted = evaluation_results['weighted_avg']
        report += f"Weighted Average:\n"
        report += f"  Precision: {weighted['precision']:.3f}\n"
        report += f"  Recall: {weighted['recall']:.3f}\n"
        report += f"  F1-Score: {weighted['f1-score']:.3f}\n"
        
        return report
