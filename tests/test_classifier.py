"""
Tests for the DisasterClassifier class.
"""

import pytest
import numpy as np
from src.core.classifier import DisasterClassifier


class TestDisasterClassifier:
    """Test cases for DisasterClassifier."""
    
    @pytest.fixture
    def config(self):
        """Create test configuration."""
        return {
            'disaster_classes': [
                {'name': 'flood', 'id': 0, 'color': [0, 0, 255]},
                {'name': 'wildfire', 'id': 1, 'color': [255, 0, 0]},
                {'name': 'earthquake', 'id': 2, 'color': [255, 255, 0]}
            ]
        }
    
    @pytest.fixture
    def classifier(self, config):
        """Create classifier instance for testing."""
        return DisasterClassifier(config)
    
    @pytest.fixture
    def sample_detection(self):
        """Create sample detection result."""
        return {
            'image_shape': (480, 640, 3),
            'detections': {
                'boxes': [[0.1, 0.1, 0.3, 0.3], [0.5, 0.5, 0.8, 0.8]],
                'confidences': [0.85, 0.72],
                'class_ids': [0, 1],
                'class_names': ['flood', 'wildfire']
            }
        }
    
    def test_classifier_initialization(self, classifier):
        """Test classifier initialization."""
        assert classifier is not None
        assert len(classifier.class_names) > 0
        assert 'flood' in classifier.severity_thresholds
    
    def test_classify_detection(self, classifier, sample_detection):
        """Test classification of detection results."""
        result = classifier.classify_detection(sample_detection)
        
        assert 'enhanced_detections' in result
        assert 'summary' in result
        assert len(result['enhanced_detections']) == 2
        
        # Check enhanced detection properties
        for det in result['enhanced_detections']:
            assert 'severity' in det
            assert 'area' in det
            assert 'coverage_percentage' in det
            assert 'risk_score' in det
            assert det['severity'] in ['low', 'medium', 'high']
    
    def test_calculate_severity(self, classifier):
        """Test severity calculation."""
        # Test high confidence
        severity = classifier._calculate_severity('flood', 0.9)
        assert severity == 'high'
        
        # Test medium confidence
        severity = classifier._calculate_severity('flood', 0.7)
        assert severity == 'medium'
        
        # Test low confidence
        severity = classifier._calculate_severity('flood', 0.2)
        assert severity == 'low'
    
    def test_calculate_area(self, classifier):
        """Test area calculation."""
        box = [0.1, 0.1, 0.3, 0.3]  # 0.2 x 0.2 = 0.04
        area = classifier._calculate_area(box)
        assert area == 0.04
    
    def test_calculate_coverage(self, classifier):
        """Test coverage calculation."""
        box = [0.1, 0.1, 0.3, 0.3]
        image_shape = (100, 100, 3)
        coverage = classifier._calculate_coverage(box, image_shape)
        assert coverage == 4.0  # 0.04 * 10000 / 10000 * 100
    
    def test_calculate_risk_score(self, classifier):
        """Test risk score calculation."""
        risk_score = classifier._calculate_risk_score('flood', 0.8, 0.1)
        assert 0 <= risk_score <= 1
        
        # Test different classes have different risk weights
        flood_risk = classifier._calculate_risk_score('flood', 0.8, 0.1)
        wildfire_risk = classifier._calculate_risk_score('wildfire', 0.8, 0.1)
        assert wildfire_risk > flood_risk  # Wildfire should have higher risk
    
    def test_generate_summary(self, classifier):
        """Test summary generation."""
        enhanced_detections = [
            {
                'class_name': 'flood',
                'confidence': 0.8,
                'risk_score': 0.7,
                'coverage_percentage': 10.0,
                'severity': 'high'
            },
            {
                'class_name': 'wildfire',
                'confidence': 0.6,
                'risk_score': 0.8,
                'coverage_percentage': 15.0,
                'severity': 'medium'
            }
        ]
        
        summary = classifier._generate_summary(enhanced_detections)
        
        assert summary['total_detections'] == 2
        assert summary['class_counts']['flood'] == 1
        assert summary['class_counts']['wildfire'] == 1
        assert summary['severity_counts']['high'] == 1
        assert summary['severity_counts']['medium'] == 1
        assert summary['average_confidence'] == 0.7
        assert summary['average_risk_score'] == 0.75
        assert summary['total_coverage'] == 25.0
    
    def test_empty_detections(self, classifier):
        """Test handling of empty detections."""
        empty_detection = {
            'image_shape': (480, 640, 3),
            'detections': {
                'boxes': [],
                'confidences': [],
                'class_ids': [],
                'class_names': []
            }
        }
        
        result = classifier.classify_detection(empty_detection)
        
        assert result['enhanced_detections'] == []
        assert result['summary']['total_detections'] == 0
    
    def test_evaluate_classification(self, classifier):
        """Test classification evaluation."""
        predictions = [
            {
                'detections': {
                    'class_names': ['flood', 'wildfire']
                }
            }
        ]
        
        ground_truth = [
            {
                'detections': {
                    'class_names': ['flood', 'wildfire']
                }
            }
        ]
        
        evaluation = classifier.evaluate_classification(predictions, ground_truth)
        
        assert 'accuracy' in evaluation
        assert 'confusion_matrix' in evaluation
        assert 'class_names' in evaluation
        assert evaluation['accuracy'] == 1.0  # Perfect match
    
    def test_generate_classification_report(self, classifier):
        """Test classification report generation."""
        evaluation_results = {
            'accuracy': 0.85,
            'macro_avg': {'precision': 0.8, 'recall': 0.9, 'f1-score': 0.85},
            'weighted_avg': {'precision': 0.82, 'recall': 0.85, 'f1-score': 0.83},
            'detailed_report': {
                'flood': {'precision': 0.8, 'recall': 0.9, 'f1-score': 0.85, 'support': 10},
                'wildfire': {'precision': 0.8, 'recall': 0.9, 'f1-score': 0.85, 'support': 10}
            }
        }
        
        report = classifier.generate_classification_report(evaluation_results)
        
        assert isinstance(report, str)
        assert 'Disaster Classification Report' in report
        assert 'Overall Accuracy: 0.850' in report
        assert 'FLOOD:' in report
        assert 'WILDFIRE:' in report
