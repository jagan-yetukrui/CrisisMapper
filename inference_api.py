#!/usr/bin/env python3
"""
CrisisMapper Inference API Server

A high-performance FastAPI server for real-time disaster detection from satellite
and drone imagery using YOLOv8 and advanced geospatial processing.

Author: CrisisMapper Team
License: MIT
"""

import os
import sys
import time
import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Optional, Any, Union
from datetime import datetime
import uuid
import json

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# FastAPI and related imports
from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import uvicorn

# Core imports
from src.core.detector import DisasterDetector
from src.core.classifier import DisasterClassifier
from src.geospatial.processor import GeospatialProcessor
from src.geospatial.export import GeospatialExporter
from src.data.ingestion import DataIngestion
from src.utils.config import load_config
from src.utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

# Global application state
app_state = {
    "detector": None,
    "classifier": None,
    "geospatial_processor": None,
    "exporter": None,
    "data_ingestion": None,
    "config": None,
    "startup_time": None,
    "request_count": 0,
    "total_inference_time": 0.0
}

# Pydantic models for API
class DetectionRequest(BaseModel):
    """Request model for disaster detection."""
    image_path: Optional[str] = Field(None, description="Path to input image")
    image_url: Optional[str] = Field(None, description="URL to input image")
    model_type: str = Field("yolov8m", description="Model type (yolov8n/s/m/l/x)")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0, description="Confidence threshold")
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0, description="IoU threshold")
    max_detections: int = Field(1000, ge=1, le=10000, description="Maximum detections")
    save_results: bool = Field(True, description="Whether to save results")
    export_formats: List[str] = Field(["geojson"], description="Export formats")
    georeference: bool = Field(True, description="Whether to georeference results")
    
    class Config:
        schema_extra = {
            "example": {
                "image_path": "/path/to/satellite_image.tif",
                "model_type": "yolov8m",
                "confidence_threshold": 0.7,
                "export_formats": ["geojson", "shapefile"]
            }
        }

class DetectionResponse(BaseModel):
    """Response model for disaster detection."""
    success: bool = Field(..., description="Whether detection was successful")
    message: str = Field(..., description="Response message")
    detections: List[Dict[str, Any]] = Field(..., description="Detection results")
    summary: Dict[str, Any] = Field(..., description="Detection summary")
    inference_time: float = Field(..., description="Inference time in seconds")
    fps: float = Field(..., description="Frames per second")
    model_info: Dict[str, Any] = Field(..., description="Model information")
    exported_files: Optional[Dict[str, str]] = Field(None, description="Exported file paths")
    timestamp: datetime = Field(default_factory=datetime.now, description="Response timestamp")
    request_id: str = Field(default_factory=lambda: str(uuid.uuid4()), description="Request ID")

class BatchDetectionRequest(BaseModel):
    """Request model for batch detection."""
    image_paths: List[str] = Field(..., description="List of image paths")
    model_type: str = Field("yolov8m", description="Model type")
    confidence_threshold: float = Field(0.5, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)
    max_detections: int = Field(1000, ge=1, le=10000)
    save_results: bool = Field(True)
    export_formats: List[str] = Field(["geojson"])
    parallel_processing: bool = Field(True, description="Whether to process in parallel")

class HealthResponse(BaseModel):
    """Health check response model."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(default_factory=datetime.now, description="Check timestamp")
    version: str = Field("2.0.0", description="Service version")
    uptime: float = Field(..., description="Uptime in seconds")
    components: Dict[str, str] = Field(..., description="Component status")
    metrics: Dict[str, Any] = Field(..., description="System metrics")
    request_count: int = Field(..., description="Total requests processed")
    average_inference_time: float = Field(..., description="Average inference time")

class ModelInfo(BaseModel):
    """Model information model."""
    model_type: str = Field(..., description="Model type")
    accuracy: float = Field(..., description="Model accuracy")
    precision: float = Field(..., description="Model precision")
    recall: float = Field(..., description="Model recall")
    f1_score: float = Field(..., description="Model F1 score")
    inference_time: float = Field(..., description="Average inference time")
    memory_usage: float = Field(..., description="Memory usage in GB")
    model_size: float = Field(..., description="Model size in MB")

# Initialize FastAPI app
app = FastAPI(
    title="CrisisMapper Inference API",
    description="AI-powered disaster detection from satellite and drone imagery",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(
    TrustedHostMiddleware,
    allowed_hosts=["*"]  # Configure appropriately for production
)

# Request logging middleware
@app.middleware("http")
async def log_requests(request, call_next):
    """Log all requests and track performance."""
    start_time = time.time()
    
    # Increment request counter
    app_state["request_count"] += 1
    
    # Process request
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    app_state["total_inference_time"] += process_time
    
    # Add response headers
    response.headers["X-Process-Time"] = str(process_time)
    response.headers["X-Request-ID"] = str(uuid.uuid4())
    
    # Log request
    logger.info(f"{request.method} {request.url.path} - {response.status_code} - {process_time:.3f}s")
    
    return response

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting CrisisMapper Inference API...")
    
    try:
        # Load configuration
        config = load_config()
        app_state["config"] = config
        
        # Initialize components
        logger.info("Initializing detection components...")
        app_state["detector"] = DisasterDetector(config)
        app_state["classifier"] = DisasterClassifier(config)
        app_state["geospatial_processor"] = GeospatialProcessor(config)
        app_state["exporter"] = GeospatialExporter(config)
        app_state["data_ingestion"] = DataIngestion(config)
        
        # Set startup time
        app_state["startup_time"] = time.time()
        
        logger.info("CrisisMapper Inference API started successfully")
        logger.info(f"Model: {config['model']['name']}")
        logger.info(f"Device: {app_state['detector'].device}")
        logger.info(f"Confidence threshold: {config['model']['confidence_threshold']}")
        
    except Exception as e:
        logger.error(f"Failed to start CrisisMapper API: {e}")
        raise

# Shutdown event
@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown."""
    logger.info("Shutting down CrisisMapper Inference API...")
    
    # Calculate final metrics
    uptime = time.time() - app_state["startup_time"]
    avg_inference_time = (
        app_state["total_inference_time"] / app_state["request_count"]
        if app_state["request_count"] > 0 else 0
    )
    
    logger.info(f"Total requests processed: {app_state['request_count']}")
    logger.info(f"Total uptime: {uptime:.2f} seconds")
    logger.info(f"Average inference time: {avg_inference_time:.3f} seconds")
    logger.info("CrisisMapper Inference API shutdown complete")

# Root endpoint
@app.get("/", response_model=Dict[str, Any])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "CrisisMapper Inference API",
        "version": "2.0.0",
        "description": "AI-powered disaster detection from satellite and drone imagery",
        "docs": "/docs",
        "health": "/health",
        "endpoints": {
            "detect": "/detect",
            "batch_detect": "/detect/batch",
            "upload": "/upload",
            "models": "/models",
            "health": "/health"
        },
        "features": [
            "Real-time disaster detection",
            "Multi-model support (YOLOv8 n/s/m/l/x)",
            "Geospatial processing",
            "Multi-format export",
            "Batch processing",
            "High-performance inference"
        ]
    }

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Comprehensive health check."""
    try:
        # Check component status
        components = {}
        
        if app_state["detector"]:
            components["detector"] = "healthy"
        else:
            components["detector"] = "unhealthy"
        
        if app_state["classifier"]:
            components["classifier"] = "healthy"
        else:
            components["classifier"] = "unhealthy"
        
        if app_state["geospatial_processor"]:
            components["geospatial_processor"] = "healthy"
        else:
            components["geospatial_processor"] = "unhealthy"
        
        if app_state["exporter"]:
            components["exporter"] = "healthy"
        else:
            components["exporter"] = "unhealthy"
        
        # Calculate uptime
        uptime = time.time() - app_state["startup_time"] if app_state["startup_time"] else 0
        
        # Calculate average inference time
        avg_inference_time = (
            app_state["total_inference_time"] / app_state["request_count"]
            if app_state["request_count"] > 0 else 0
        )
        
        # Get system metrics
        import psutil
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        
        metrics = {
            "memory_usage_percent": memory.percent,
            "memory_available_gb": memory.available / (1024**3),
            "cpu_percent": cpu_percent,
            "disk_usage_percent": psutil.disk_usage('/').percent
        }
        
        # Determine overall status
        overall_status = "healthy" if all(status == "healthy" for status in components.values()) else "degraded"
        
        return HealthResponse(
            status=overall_status,
            timestamp=datetime.now(),
            version="2.0.0",
            uptime=uptime,
            components=components,
            metrics=metrics,
            request_count=app_state["request_count"],
            average_inference_time=avg_inference_time
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            version="2.0.0",
            uptime=0,
            components={},
            metrics={},
            request_count=0,
            average_inference_time=0
        )

# Detection endpoint
@app.post("/detect", response_model=DetectionResponse)
async def detect_disasters(request: DetectionRequest):
    """Detect disasters in a single image."""
    try:
        # Validate input
        if not request.image_path and not request.image_url:
            raise HTTPException(
                status_code=400,
                detail="Either image_path or image_url must be provided"
            )
        
        # Get image path
        image_path = request.image_path
        if request.image_url:
            # In a real implementation, you would download the image
            raise HTTPException(
                status_code=501,
                detail="Image URL download not yet implemented"
            )
        
        # Validate image exists
        if not os.path.exists(image_path):
            raise HTTPException(
                status_code=404,
                detail=f"Image not found: {image_path}"
            )
        
        # Update detector parameters
        detector = app_state["detector"]
        detector.config['model']['confidence_threshold'] = request.confidence_threshold
        detector.config['model']['iou_threshold'] = request.iou_threshold
        detector.config['model']['max_detections'] = request.max_detections
        
        # Run detection
        start_time = time.time()
        detection_result = detector.detect(
            image_path, 
            save_results=request.save_results
        )
        detection_time = time.time() - start_time
        
        # Classify detections
        classifier = app_state["classifier"]
        classification_result = classifier.classify_detection(detection_result)
        
        # Process geospatially if requested
        exported_files = None
        if request.georeference and request.export_formats:
            try:
                gdf = app_state["geospatial_processor"].process_detection_results([detection_result])
                exported_files = app_state["exporter"].export_results(
                    gdf,
                    filename_prefix=f"detection_{int(time.time())}",
                    formats=request.export_formats
                )
            except Exception as e:
                logger.warning(f"Geospatial processing failed: {e}")
        
        # Create response
        response = DetectionResponse(
            success=True,
            message="Detection completed successfully",
            detections=classification_result['enhanced_detections'],
            summary=classification_result['summary'],
            inference_time=detection_time,
            fps=1.0 / detection_time if detection_time > 0 else 0,
            model_info=detection_result['model_info'],
            exported_files=exported_files,
            timestamp=datetime.now(),
            request_id=str(uuid.uuid4())
        )
        
        logger.info(f"Detection completed: {len(classification_result['enhanced_detections'])} detections in {detection_time:.3f}s")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}"
        )

# Batch detection endpoint
@app.post("/detect/batch", response_model=Dict[str, Any])
async def detect_disasters_batch(request: BatchDetectionRequest):
    """Detect disasters in multiple images."""
    try:
        # Validate image paths
        valid_paths = []
        for path in request.image_paths:
            if os.path.exists(path):
                valid_paths.append(path)
            else:
                logger.warning(f"Image not found: {path}")
        
        if not valid_paths:
            raise HTTPException(
                status_code=400,
                detail="No valid image paths provided"
            )
        
        # Process images
        start_time = time.time()
        results = []
        
        if request.parallel_processing:
            # Parallel processing
            import concurrent.futures
            
            def process_single_image(image_path):
                try:
                    detector = app_state["detector"]
                    detector.config['model']['confidence_threshold'] = request.confidence_threshold
                    detector.config['model']['iou_threshold'] = request.iou_threshold
                    detector.config['model']['max_detections'] = request.max_detections
                    
                    result = detector.detect(image_path, save_results=request.save_results)
                    classification = app_state["classifier"].classify_detection(result)
                    
                    return {
                        "image_path": image_path,
                        "success": True,
                        "detections": classification['enhanced_detections'],
                        "summary": classification['summary']
                    }
                except Exception as e:
                    return {
                        "image_path": image_path,
                        "success": False,
                        "error": str(e)
                    }
            
            with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
                future_to_path = {
                    executor.submit(process_single_image, path): path 
                    for path in valid_paths
                }
                
                for future in concurrent.futures.as_completed(future_to_path):
                    result = future.result()
                    results.append(result)
        else:
            # Sequential processing
            for image_path in valid_paths:
                try:
                    detector = app_state["detector"]
                    detector.config['model']['confidence_threshold'] = request.confidence_threshold
                    detector.config['model']['iou_threshold'] = request.iou_threshold
                    detector.config['model']['max_detections'] = request.max_detections
                    
                    result = detector.detect(image_path, save_results=request.save_results)
                    classification = app_state["classifier"].classify_detection(result)
                    
                    results.append({
                        "image_path": image_path,
                        "success": True,
                        "detections": classification['enhanced_detections'],
                        "summary": classification['summary']
                    })
                except Exception as e:
                    results.append({
                        "image_path": image_path,
                        "success": False,
                        "error": str(e)
                    })
        
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r.get("success", False))
        failed = len(results) - successful
        
        # Calculate aggregate statistics
        all_detections = []
        for result in results:
            if result.get("success", False):
                all_detections.extend(result.get("detections", []))
        
        aggregate_summary = {
            "total_images": len(valid_paths),
            "successful": successful,
            "failed": failed,
            "total_detections": len(all_detections),
            "processing_time": total_time,
            "average_time_per_image": total_time / len(valid_paths) if valid_paths else 0
        }
        
        return {
            "success": True,
            "message": f"Batch detection completed: {successful} successful, {failed} failed",
            "results": results,
            "aggregate_summary": aggregate_summary,
            "timestamp": datetime.now().isoformat(),
            "request_id": str(uuid.uuid4())
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch detection failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Batch detection failed: {str(e)}"
        )

# File upload endpoint
@app.post("/upload")
async def upload_image(file: UploadFile = File(...)):
    """Upload an image file for processing."""
    try:
        # Create uploads directory
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)
        
        # Generate unique filename
        file_extension = Path(file.filename).suffix
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        file_path = uploads_dir / unique_filename
        
        # Save uploaded file
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Get file info
        file_size = len(content)
        
        logger.info(f"File uploaded: {file.filename} -> {file_path} ({file_size} bytes)")
        
        return {
            "success": True,
            "message": "File uploaded successfully",
            "file_path": str(file_path),
            "file_size": file_size,
            "filename": file.filename,
            "upload_id": str(uuid.uuid4())
        }
        
    except Exception as e:
        logger.error(f"File upload failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"File upload failed: {str(e)}"
        )

# Models endpoint
@app.get("/models", response_model=List[ModelInfo])
async def get_models():
    """Get information about available models."""
    try:
        models = []
        
        # Model performance data (in a real implementation, this would be dynamic)
        model_data = {
            "yolov8n": {"accuracy": 0.892, "precision": 0.873, "recall": 0.911, "f1_score": 0.891, "inference_time": 0.006, "memory_usage": 1.2, "model_size": 6.2},
            "yolov8s": {"accuracy": 0.921, "precision": 0.908, "recall": 0.934, "f1_score": 0.921, "inference_time": 0.011, "memory_usage": 2.1, "model_size": 21.5},
            "yolov8m": {"accuracy": 0.947, "precision": 0.932, "recall": 0.961, "f1_score": 0.946, "inference_time": 0.013, "memory_usage": 3.8, "model_size": 49.7},
            "yolov8l": {"accuracy": 0.958, "precision": 0.941, "recall": 0.973, "f1_score": 0.957, "inference_time": 0.018, "memory_usage": 5.2, "model_size": 83.7},
            "yolov8x": {"accuracy": 0.963, "precision": 0.948, "recall": 0.978, "f1_score": 0.963, "inference_time": 0.029, "memory_usage": 7.1, "model_size": 136.7}
        }
        
        for model_type, data in model_data.items():
            models.append(ModelInfo(
                model_type=model_type,
                accuracy=data["accuracy"],
                precision=data["precision"],
                recall=data["recall"],
                f1_score=data["f1_score"],
                inference_time=data["inference_time"],
                memory_usage=data["memory_usage"],
                model_size=data["model_size"]
            ))
        
        return models
        
    except Exception as e:
        logger.error(f"Failed to get models: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get models: {str(e)}"
        )

# Results endpoint
@app.get("/results/{filename}")
async def get_result_file(filename: str):
    """Download a result file."""
    try:
        results_dir = Path("results")
        file_path = results_dir / filename
        
        if not file_path.exists():
            raise HTTPException(
                status_code=404,
                detail="File not found"
            )
        
        return FileResponse(
            path=str(file_path),
            filename=filename,
            media_type="application/octet-stream"
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File download failed: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"File download failed: {str(e)}"
        )

# Performance metrics endpoint
@app.get("/metrics")
async def get_metrics():
    """Get performance metrics."""
    try:
        uptime = time.time() - app_state["startup_time"] if app_state["startup_time"] else 0
        avg_inference_time = (
            app_state["total_inference_time"] / app_state["request_count"]
            if app_state["request_count"] > 0 else 0
        )
        
        # Get system metrics
        import psutil
        memory = psutil.virtual_memory()
        cpu_percent = psutil.cpu_percent()
        disk = psutil.disk_usage('/')
        
        return {
            "uptime_seconds": uptime,
            "total_requests": app_state["request_count"],
            "total_inference_time": app_state["total_inference_time"],
            "average_inference_time": avg_inference_time,
            "requests_per_second": app_state["request_count"] / uptime if uptime > 0 else 0,
            "system_metrics": {
                "memory_usage_percent": memory.percent,
                "memory_available_gb": memory.available / (1024**3),
                "cpu_percent": cpu_percent,
                "disk_usage_percent": disk.percent,
                "disk_free_gb": disk.free / (1024**3)
            },
            "model_metrics": app_state["detector"].get_performance_stats() if app_state["detector"] else {}
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get metrics: {str(e)}"
        )

# Main execution
if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    # Run the server
    uvicorn.run(
        "inference_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
