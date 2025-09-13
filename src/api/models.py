"""
Pydantic models for CrisisMapper API.
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Union
from datetime import datetime
from enum import Enum


class DisasterClass(str, Enum):
    """Disaster class enumeration."""
    FLOOD = "flood"
    WILDFIRE = "wildfire"
    EARTHQUAKE = "earthquake"
    LANDSLIDE = "landslide"
    HURRICANE = "hurricane"


class SeverityLevel(str, Enum):
    """Severity level enumeration."""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class DetectionRequest(BaseModel):
    """Request model for disaster detection."""
    image_path: Optional[str] = Field(None, description="Path to input image")
    image_url: Optional[str] = Field(None, description="URL to input image")
    confidence_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold")
    iou_threshold: Optional[float] = Field(0.45, ge=0.0, le=1.0, description="IoU threshold")
    max_detections: Optional[int] = Field(1000, ge=1, le=10000, description="Maximum number of detections")
    save_results: bool = Field(True, description="Whether to save detection results")
    export_formats: Optional[List[str]] = Field(["geojson"], description="Export formats")
    
    class Config:
        schema_extra = {
            "example": {
                "image_path": "/path/to/image.jpg",
                "confidence_threshold": 0.5,
                "iou_threshold": 0.45,
                "max_detections": 1000,
                "save_results": True,
                "export_formats": ["geojson", "shapefile"]
            }
        }


class BoundingBox(BaseModel):
    """Bounding box model."""
    x1: float = Field(..., description="Left coordinate")
    y1: float = Field(..., description="Top coordinate")
    x2: float = Field(..., description="Right coordinate")
    y2: float = Field(..., description="Bottom coordinate")


class Detection(BaseModel):
    """Individual detection model."""
    detection_id: int = Field(..., description="Unique detection ID")
    class_id: int = Field(..., description="Class ID")
    class_name: str = Field(..., description="Class name")
    confidence: float = Field(..., description="Confidence score")
    bounding_box: BoundingBox = Field(..., description="Bounding box coordinates")
    severity: Optional[SeverityLevel] = Field(None, description="Severity level")
    area_m2: Optional[float] = Field(None, description="Area in square meters")
    coverage_percentage: Optional[float] = Field(None, description="Coverage percentage")
    risk_score: Optional[float] = Field(None, description="Risk score")


class DetectionSummary(BaseModel):
    """Detection summary model."""
    total_detections: int = Field(..., description="Total number of detections")
    class_counts: Dict[str, int] = Field(..., description="Count by class")
    severity_counts: Dict[str, int] = Field(..., description="Count by severity")
    average_confidence: float = Field(..., description="Average confidence")
    average_risk_score: float = Field(..., description="Average risk score")
    total_coverage: float = Field(..., description="Total coverage percentage")


class DetectionResponse(BaseModel):
    """Response model for disaster detection."""
    success: bool = Field(..., description="Whether the detection was successful")
    message: str = Field(..., description="Response message")
    detections: List[Detection] = Field(..., description="List of detections")
    summary: DetectionSummary = Field(..., description="Detection summary")
    inference_time: float = Field(..., description="Inference time in seconds")
    fps: float = Field(..., description="Frames per second")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    exported_files: Optional[Dict[str, str]] = Field(None, description="Exported file paths")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "success": True,
                "message": "Detection completed successfully",
                "detections": [
                    {
                        "detection_id": 0,
                        "class_id": 0,
                        "class_name": "flood",
                        "confidence": 0.85,
                        "bounding_box": {
                            "x1": 0.1,
                            "y1": 0.2,
                            "x2": 0.4,
                            "y2": 0.6
                        },
                        "severity": "high",
                        "area_m2": 1500.0,
                        "coverage_percentage": 12.5,
                        "risk_score": 0.8
                    }
                ],
                "summary": {
                    "total_detections": 1,
                    "class_counts": {"flood": 1},
                    "severity_counts": {"high": 1},
                    "average_confidence": 0.85,
                    "average_risk_score": 0.8,
                    "total_coverage": 12.5
                },
                "inference_time": 0.15,
                "fps": 6.67,
                "model_info": {
                    "name": "yolov8n",
                    "device": "cuda",
                    "confidence_threshold": 0.5
                },
                "exported_files": {
                    "geojson": "/path/to/results.geojson",
                    "shapefile": "/path/to/results.shp"
                }
            }
        }


class BatchDetectionRequest(BaseModel):
    """Request model for batch disaster detection."""
    image_paths: List[str] = Field(..., description="List of image paths")
    confidence_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    iou_threshold: Optional[float] = Field(0.45, ge=0.0, le=1.0)
    max_detections: Optional[int] = Field(1000, ge=1, le=10000)
    save_results: bool = Field(True)
    export_formats: Optional[List[str]] = Field(["geojson"])
    
    class Config:
        schema_extra = {
            "example": {
                "image_paths": ["/path/to/image1.jpg", "/path/to/image2.jpg"],
                "confidence_threshold": 0.5,
                "save_results": True,
                "export_formats": ["geojson"]
            }
        }


class BatchDetectionResponse(BaseModel):
    """Response model for batch disaster detection."""
    success: bool = Field(..., description="Whether the batch detection was successful")
    message: str = Field(..., description="Response message")
    results: List[DetectionResponse] = Field(..., description="List of detection results")
    total_images: int = Field(..., description="Total number of images processed")
    successful_images: int = Field(..., description="Number of successfully processed images")
    failed_images: int = Field(..., description="Number of failed images")
    total_processing_time: float = Field(..., description="Total processing time in seconds")
    average_processing_time: float = Field(..., description="Average processing time per image")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")


class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    version: str = Field(..., description="Service version")
    model_loaded: bool = Field(..., description="Whether the model is loaded")
    device: str = Field(..., description="Computation device")
    memory_usage: Optional[Dict[str, Any]] = Field(None, description="Memory usage information")
    
    class Config:
        schema_extra = {
            "example": {
                "status": "healthy",
                "version": "1.0.0",
                "model_loaded": True,
                "device": "cuda",
                "memory_usage": {
                    "total_mb": 8192,
                    "used_mb": 4096,
                    "free_mb": 4096
                }
            }
        }


class ErrorResponse(BaseModel):
    """Error response model."""
    success: bool = Field(False, description="Always false for errors")
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp")
    
    class Config:
        schema_extra = {
            "example": {
                "success": False,
                "error": "ValidationError",
                "message": "Invalid image path provided",
                "details": {
                    "field": "image_path",
                    "value": "/invalid/path"
                }
            }
        }
