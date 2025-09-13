"""
FastAPI main application for CrisisMapper.

This module provides the REST API interface for disaster detection,
data management, and result visualization.
"""

import os
import time
import asyncio
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

from fastapi import FastAPI, HTTPException, UploadFile, File, Form, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from .models import (
    DetectionRequest, DetectionResponse, BatchDetectionRequest, BatchDetectionResponse,
    HealthResponse, ErrorResponse, Detection, DetectionSummary, BoundingBox
)
from ..core.detector import DisasterDetector
from ..core.classifier import DisasterClassifier
from ..geospatial.processor import GeospatialProcessor
from ..geospatial.export import GeospatialExporter
from ..data.ingestion import DataIngestion
from ..utils.config import load_config
from ..utils.logger import setup_logger

# Setup logging
logger = setup_logger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="CrisisMapper API",
    description="AI-powered geospatial mapping system for disaster detection and classification",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model instances
detector = None
classifier = None
geospatial_processor = None
exporter = None
data_ingestion = None
config = None


@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup."""
    global detector, classifier, geospatial_processor, exporter, data_ingestion, config
    
    try:
        # Load configuration
        config = load_config()
        
        # Initialize detector
        detector = DisasterDetector(config)
        
        # Initialize classifier
        classifier = DisasterClassifier(config)
        
        # Initialize geospatial processor
        geospatial_processor = GeospatialProcessor(config)
        
        # Initialize exporter
        exporter = GeospatialExporter(config)
        
        # Initialize data ingestion
        data_ingestion = DataIngestion(config)
        
        logger.info("CrisisMapper API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to initialize CrisisMapper API: {e}")
        raise


@app.get("/", response_model=Dict[str, str])
async def root():
    """Root endpoint with API information."""
    return {
        "message": "CrisisMapper API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    try:
        # Check if models are loaded
        model_loaded = detector is not None and classifier is not None
        
        # Get device information
        device = detector.device if detector else "unknown"
        
        # Get memory usage (simplified)
        memory_usage = None
        try:
            import psutil
            memory = psutil.virtual_memory()
            memory_usage = {
                "total_mb": round(memory.total / (1024 * 1024)),
                "used_mb": round(memory.used / (1024 * 1024)),
                "free_mb": round(memory.available / (1024 * 1024))
            }
        except ImportError:
            pass
        
        return HealthResponse(
            status="healthy" if model_loaded else "unhealthy",
            version="1.0.0",
            model_loaded=model_loaded,
            device=device,
            memory_usage=memory_usage
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            version="1.0.0",
            model_loaded=False,
            device="unknown"
        )


@app.post("/detect", response_model=DetectionResponse)
async def detect_disasters(request: DetectionRequest):
    """Detect disasters in a single image."""
    try:
        # Validate input
        if not request.image_path and not request.image_url:
            raise HTTPException(status_code=400, detail="Either image_path or image_url must be provided")
        
        # Get image path
        image_path = request.image_path
        if request.image_url:
            # Download image from URL (placeholder)
            # In a real implementation, you would download the image here
            raise HTTPException(status_code=501, detail="Image URL download not yet implemented")
        
        # Validate image exists
        if not os.path.exists(image_path):
            raise HTTPException(status_code=404, detail=f"Image not found: {image_path}")
        
        # Update detector parameters
        if request.confidence_threshold is not None:
            detector.config['model']['confidence_threshold'] = request.confidence_threshold
        if request.iou_threshold is not None:
            detector.config['model']['iou_threshold'] = request.iou_threshold
        if request.max_detections is not None:
            detector.config['model']['max_detections'] = request.max_detections
        
        # Run detection
        start_time = time.time()
        detection_result = detector.detect(image_path, save_results=request.save_results)
        detection_time = time.time() - start_time
        
        # Classify detections
        classification_result = classifier.classify_detection(detection_result)
        
        # Convert to API response format
        detections = []
        for i, det in enumerate(classification_result['enhanced_detections']):
            detection = Detection(
                detection_id=i,
                class_id=det['class_id'],
                class_name=det['class_name'],
                confidence=det['confidence'],
                bounding_box=BoundingBox(
                    x1=det['box'][0],
                    y1=det['box'][1],
                    x2=det['box'][2],
                    y2=det['box'][3]
                ),
                severity=det.get('severity'),
                area_m2=det.get('area'),
                coverage_percentage=det.get('coverage_percentage'),
                risk_score=det.get('risk_score')
            )
            detections.append(detection)
        
        # Create summary
        summary = DetectionSummary(
            total_detections=classification_result['summary']['total_detections'],
            class_counts=classification_result['summary']['class_counts'],
            severity_counts=classification_result['summary']['severity_counts'],
            average_confidence=classification_result['summary']['average_confidence'],
            average_risk_score=classification_result['summary']['average_risk_score'],
            total_coverage=classification_result['summary']['total_coverage']
        )
        
        # Export results if requested
        exported_files = None
        if request.save_results and request.export_formats:
            try:
                # Process with geospatial processor
                gdf = geospatial_processor.process_detection_results([detection_result])
                
                # Export to requested formats
                exported_files = exporter.export_results(
                    gdf, 
                    filename_prefix=f"detection_{int(time.time())}",
                    formats=request.export_formats
                )
            except Exception as e:
                logger.warning(f"Export failed: {e}")
        
        return DetectionResponse(
            success=True,
            message="Detection completed successfully",
            detections=detections,
            summary=summary,
            inference_time=detection_time,
            fps=1.0 / detection_time if detection_time > 0 else 0,
            model_info=detection_result['model_info'],
            exported_files=exported_files
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")


@app.post("/detect/batch", response_model=BatchDetectionResponse)
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
            raise HTTPException(status_code=400, detail="No valid image paths provided")
        
        # Process images
        start_time = time.time()
        results = []
        
        for i, image_path in enumerate(valid_paths):
            try:
                # Create single detection request
                single_request = DetectionRequest(
                    image_path=image_path,
                    confidence_threshold=request.confidence_threshold,
                    iou_threshold=request.iou_threshold,
                    max_detections=request.max_detections,
                    save_results=request.save_results,
                    export_formats=request.export_formats
                )
                
                # Run detection
                result = await detect_disasters(single_request)
                results.append(result)
                
            except Exception as e:
                logger.error(f"Failed to process {image_path}: {e}")
                # Create error response
                error_result = DetectionResponse(
                    success=False,
                    message=f"Failed to process image: {str(e)}",
                    detections=[],
                    summary=DetectionSummary(
                        total_detections=0,
                        class_counts={},
                        severity_counts={},
                        average_confidence=0.0,
                        average_risk_score=0.0,
                        total_coverage=0.0
                    ),
                    inference_time=0.0,
                    fps=0.0,
                    model_info={}
                )
                results.append(error_result)
        
        total_time = time.time() - start_time
        successful = sum(1 for r in results if r.success)
        failed = len(results) - successful
        
        return BatchDetectionResponse(
            success=True,
            message=f"Batch detection completed: {successful} successful, {failed} failed",
            results=results,
            total_images=len(request.image_paths),
            successful_images=successful,
            failed_images=failed,
            total_processing_time=total_time,
            average_processing_time=total_time / len(valid_paths) if valid_paths else 0
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch detection failed: {e}")
        raise HTTPException(status_code=500, detail=f"Batch detection failed: {str(e)}")


@app.post("/upload", response_model=Dict[str, str])
async def upload_image(file: UploadFile = File(...)):
    """Upload an image file."""
    try:
        # Create uploads directory
        uploads_dir = Path("uploads")
        uploads_dir.mkdir(exist_ok=True)
        
        # Save uploaded file
        file_path = uploads_dir / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        return {
            "message": "File uploaded successfully",
            "file_path": str(file_path),
            "file_size": len(content)
        }
        
    except Exception as e:
        logger.error(f"Upload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")


@app.get("/results/{filename}")
async def get_result_file(filename: str):
    """Download a result file."""
    try:
        results_dir = Path(config['output']['base_dir']) / "results"
        file_path = results_dir / filename
        
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found")
        
        return FileResponse(file_path)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"File download failed: {e}")
        raise HTTPException(status_code=500, detail=f"File download failed: {str(e)}")


@app.get("/data/summary", response_model=Dict[str, Any])
async def get_data_summary():
    """Get summary of available data."""
    try:
        summary = data_ingestion.get_data_summary()
        return summary
        
    except Exception as e:
        logger.error(f"Data summary failed: {e}")
        raise HTTPException(status_code=500, detail=f"Data summary failed: {str(e)}")


@app.get("/performance", response_model=Dict[str, Any])
async def get_performance_stats():
    """Get performance statistics."""
    try:
        if detector:
            stats = detector.get_performance_stats()
            return stats
        else:
            return {"error": "Detector not initialized"}
            
    except Exception as e:
        logger.error(f"Performance stats failed: {e}")
        raise HTTPException(status_code=500, detail=f"Performance stats failed: {str(e)}")


if __name__ == "__main__":
    # Run the API server
    uvicorn.run(
        "src.api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
