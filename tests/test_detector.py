#!/usr/bin/env python3
"""
Unit tests for the DisasterDetector class.

Author: CrisisMapper Team
License: MIT
"""

import pytest
import numpy as np
import cv2
from pathlib import Path
import tempfile
import os
import sys
from unittest.mock import Mock, patch

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.core.detector import DisasterDetector

class TestDisasterDetector:
    """Test class for DisasterDetector."""
    
    @pytest.fixture
    def sample_config(self):
        """Create a sample configuration for testing."""
        return {
            'model': {
                'name': 'yolov8m',
                'confidence_threshold': 0.5,
                'iou_threshold': 0.45,
                'max_detections': 1000,
                'device': 'cpu'
            },
            'preprocessing': {
                'resize': True,
                'normalize': True,
                'tile_size': 1024
            },
            'output': {
                'save_results': True,
                'export_formats': ['geojson']
            }
        }
    
    @pytest.fixture
    def sample_image(self):
        """Create a sample image for testing."""
        return np.random.randint(0, 255, (1024, 1024, 3), dtype=np.uint8)
    
    def test_detector_initialization(self, sample_config):
        """Test detector initialization."""
        with patch('src.core.detector.YOLO') as mock_yolo:
            mock_yolo.return_value = Mock()
            
            detector = DisasterDetector(sample_config)
            
            assert detector.config == sample_config
            assert detector.model_name == 'yolov8m'
            assert detector.confidence_threshold == 0.5
    
    def test_detect_with_numpy_array(self, sample_config, sample_image):
        """Test detection with numpy array input."""
        with patch('src.core.detector.YOLO') as mock_yolo:
            mock_model = Mock()
            mock_model.return_value = [
                Mock(
                    boxes=Mock(
                        xyxy=Mock(cpu=lambda: np.array([[100, 100, 200, 200]])),
                        conf=Mock(cpu=lambda: np.array([0.8])),
                        cls=Mock(cpu=lambda: np.array([0]))
                    ),
                    names={0: 'flood'}
                )
            ]
            mock_yolo.return_value = mock_model
            
            detector = DisasterDetector(sample_config)
            results = detector.detect(sample_image)
            
            assert results is not None
            assert 'detections' in results
            assert 'model_info' in results
            assert 'inference_time' in results

if __name__ == "__main__":
    pytest.main([__file__])