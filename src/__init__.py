"""
CrisisMapper: AI-powered geospatial mapping system for disaster detection.

This package provides comprehensive tools for detecting and classifying
disaster zones from satellite and drone imagery using YOLOv8 and OpenCV.
"""

__version__ = "1.0.0"
__author__ = "CrisisMapper Team"
__email__ = "contact@crisismapper.ai"

from .core.detector import DisasterDetector
from .core.classifier import DisasterClassifier
from .geospatial.processor import GeospatialProcessor
from .data.ingestion import DataIngestion
from .api.main import app

__all__ = [
    "DisasterDetector",
    "DisasterClassifier", 
    "GeospatialProcessor",
    "DataIngestion",
    "app"
]
