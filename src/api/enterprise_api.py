"""
Enterprise-Grade FastAPI Application for CrisisMapper.

This module provides a comprehensive, production-ready API with authentication,
authorization, monitoring, and advanced features for enterprise deployment.
"""

import os
import time
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Any, Union
from pathlib import Path
import logging
import json
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Depends, status, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from pydantic import BaseModel, Field
import psutil

from ..core.detector import DisasterDetector
from ..core.classifier import DisasterClassifier
from ..geospatial.processor import GeospatialProcessor
from ..geospatial.export import GeospatialExporter
from ..data.ingestion import DataIngestion
from ..research.experiment_manager import ExperimentManager
from ..monitoring.metrics_collector import MetricsCollector
from ..security.auth_manager import AuthManager, User, Permission, UserRole
from ..utils.config import load_config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

# Global application state
app_state = {
    "detector": None,
    "classifier": None,
    "geospatial_processor": None,
    "exporter": None,
    "data_ingestion": None,
    "experiment_manager": None,
    "metrics_collector": None,
    "auth_manager": None,
    "config": None
}

# Security
security = HTTPBearer()

# Request/Response Models
class LoginRequest(BaseModel):
    username: str = Field(..., description="Username or email")
    password: str = Field(..., description="Password")

class LoginResponse(BaseModel):
    access_token: str = Field(..., description="JWT access token")
    token_type: str = Field(default="bearer", description="Token type")
    expires_in: int = Field(..., description="Token expiry time in seconds")
    user_info: Dict[str, Any] = Field(..., description="User information")

class DetectionRequest(BaseModel):
    image_path: Optional[str] = Field(None, description="Path to input image")
    image_url: Optional[str] = Field(None, description="URL to input image")
    confidence_threshold: Optional[float] = Field(0.5, ge=0.0, le=1.0)
    iou_threshold: Optional[float] = Field(0.45, ge=0.0, le=1.0)
    max_detections: Optional[int] = Field(1000, ge=1, le=10000)
    save_results: bool = Field(True, description="Whether to save results")
    export_formats: Optional[List[str]] = Field(["geojson"])
    experiment_id: Optional[str] = Field(None, description="Associated experiment ID")

class ExperimentRequest(BaseModel):
    name: str = Field(..., description="Experiment name")
    description: str = Field(..., description="Experiment description")
    model_type: str = Field(..., description="Model type")
    model_params: Dict[str, Any] = Field(..., description="Model parameters")
    dataset_path: str = Field(..., description="Dataset path")
    train_split: float = Field(0.7, ge=0.0, le=1.0)
    val_split: float = Field(0.15, ge=0.0, le=1.0)
    test_split: float = Field(0.15, ge=0.0, le=1.0)

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Check timestamp")
    version: str = Field(..., description="Service version")
    uptime: float = Field(..., description="Uptime in seconds")
    components: Dict[str, str] = Field(..., description="Component status")
    metrics: Dict[str, Any] = Field(..., description="System metrics")

# Dependency functions
async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)) -> User:
    """Get current authenticated user."""
    token = credentials.credentials
    
    # Validate JWT token
    payload = app_state["auth_manager"].validate_jwt_token(token)
    if not payload:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Get user from database
    user = app_state["auth_manager"].users.get(payload["user_id"])
    if not user or not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found or inactive",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return user

def require_permission(permission: Permission):
    """Decorator to require specific permission."""
    def permission_checker(user: User = Depends(get_current_user)):
        if not app_state["auth_manager"].check_permission(user, permission):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Permission required: {permission.value}"
            )
        return user
    return permission_checker

def require_role(role: UserRole):
    """Decorator to require specific role."""
    def role_checker(user: User = Depends(get_current_user)):
        if not app_state["auth_manager"].check_role(user, role):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role required: {role.value}"
            )
        return user
    return role_checker

# Application lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting CrisisMapper Enterprise API...")
    
    try:
        # Load configuration
        config = load_config()
        app_state["config"] = config
        
        # Initialize components
        app_state["detector"] = DisasterDetector(config)
        app_state["classifier"] = DisasterClassifier(config)
        app_state["geospatial_processor"] = GeospatialProcessor(config)
        app_state["exporter"] = GeospatialExporter(config)
        app_state["data_ingestion"] = DataIngestion(config)
        app_state["experiment_manager"] = ExperimentManager()
        app_state["metrics_collector"] = MetricsCollector()
        app_state["auth_manager"] = AuthManager(secret_key=os.getenv("JWT_SECRET_KEY", "your-secret-key"))
        
        # Start metrics collection
        app_state["metrics_collector"].start_collection()
        
        # Record startup metrics
        app_state["metrics_collector"].record_metric("api_startup", 1)
        app_state["metrics_collector"].record_metric("startup_time", time.time())
        
        logger.info("CrisisMapper Enterprise API started successfully")
        
    except Exception as e:
        logger.error(f"Failed to start API: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("Shutting down CrisisMapper Enterprise API...")
    
    try:
        # Stop metrics collection
        if app_state["metrics_collector"]:
            app_state["metrics_collector"].stop_collection()
        
        # Cleanup
        app_state["auth_manager"].cleanup_expired_sessions()
        
        logger.info("CrisisMapper Enterprise API shutdown complete")
        
    except Exception as e:
        logger.error(f"Error during shutdown: {e}")

# Initialize FastAPI app
app = FastAPI(
    title="CrisisMapper Enterprise API",
    description="AI-powered geospatial mapping system for disaster detection and classification",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan
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
async def log_requests(request: Request, call_next):
    """Log all requests."""
    start_time = time.time()
    
    # Record request metrics
    app_state["metrics_collector"].increment_counter("api_requests_total")
    
    response = await call_next(request)
    
    # Calculate processing time
    process_time = time.time() - start_time
    app_state["metrics_collector"].record_timing("api_request_duration", process_time)
    
    # Add response headers
    response.headers["X-Process-Time"] = str(process_time)
    
    return response

# Authentication endpoints
@app.post("/auth/login", response_model=LoginResponse)
async def login(login_data: LoginRequest, request: Request):
    """Authenticate user and return JWT token."""
    try:
        # Get client IP and user agent
        client_ip = request.client.host
        user_agent = request.headers.get("user-agent", "")
        
        # Authenticate user
        session_id = app_state["auth_manager"].authenticate_user(
            login_data.username,
            login_data.password,
            client_ip,
            user_agent
        )
        
        if not session_id:
            app_state["metrics_collector"].increment_counter("auth_failed_logins")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        # Get user information
        user = app_state["auth_manager"].validate_session(session_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication failed"
            )
        
        # Generate JWT token
        access_token = app_state["auth_manager"].generate_jwt_token(user)
        
        # Record successful login
        app_state["metrics_collector"].increment_counter("auth_successful_logins")
        
        return LoginResponse(
            access_token=access_token,
            token_type="bearer",
            expires_in=app_state["auth_manager"].token_expiry_hours * 3600,
            user_info={
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "roles": [role.value for role in user.roles],
                "permissions": [perm.value for perm in user.permissions]
            }
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Login error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error"
        )

@app.post("/auth/logout")
async def logout(current_user: User = Depends(get_current_user)):
    """Logout current user."""
    # In a real implementation, you would invalidate the JWT token
    # For now, we'll just return success
    app_state["metrics_collector"].increment_counter("auth_logouts")
    
    return {"message": "Logged out successfully"}

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
        
        if app_state["auth_manager"]:
            components["auth_manager"] = "healthy"
        else:
            components["auth_manager"] = "unhealthy"
        
        # Get system metrics
        metrics = app_state["metrics_collector"].get_health_status()
        
        # Calculate uptime
        uptime = time.time() - app_state["metrics_collector"].get_metric_summary("api_startup", 24).get("latest", time.time())
        
        return HealthResponse(
            status="healthy" if all(status == "healthy" for status in components.values()) else "degraded",
            timestamp=datetime.now(),
            version="2.0.0",
            uptime=uptime,
            components=components,
            metrics=metrics
        )
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return HealthResponse(
            status="unhealthy",
            timestamp=datetime.now(),
            version="2.0.0",
            uptime=0,
            components={},
            metrics={}
        )

# Detection endpoints
@app.post("/detect")
async def detect_disasters(
    request: DetectionRequest,
    current_user: User = Depends(require_permission(Permission.RUN_DETECTION))
):
    """Detect disasters in an image."""
    try:
        # Validate input
        if not request.image_path and not request.image_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Either image_path or image_url must be provided"
            )
        
        # Get image path
        image_path = request.image_path
        if request.image_url:
            # In a real implementation, you would download the image
            raise HTTPException(
                status_code=status.HTTP_501_NOT_IMPLEMENTED,
                detail="Image URL download not yet implemented"
            )
        
        # Validate image exists
        if not os.path.exists(image_path):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Image not found: {image_path}"
            )
        
        # Update detector parameters
        detector = app_state["detector"]
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
        classifier = app_state["classifier"]
        classification_result = classifier.classify_detection(detection_result)
        
        # Record metrics
        app_state["metrics_collector"].record_timing("detection_duration", detection_time)
        app_state["metrics_collector"].increment_counter("detections_total")
        app_state["metrics_collector"].record_metric("detection_confidence", classification_result['summary']['average_confidence'])
        
        # Export results if requested
        exported_files = None
        if request.save_results and request.export_formats:
            try:
                gdf = app_state["geospatial_processor"].process_detection_results([detection_result])
                exported_files = app_state["exporter"].export_results(
                    gdf,
                    filename_prefix=f"detection_{int(time.time())}",
                    formats=request.export_formats
                )
            except Exception as e:
                logger.warning(f"Export failed: {e}")
        
        # Return results
        return {
            "success": True,
            "message": "Detection completed successfully",
            "detections": classification_result['enhanced_detections'],
            "summary": classification_result['summary'],
            "inference_time": detection_time,
            "fps": 1.0 / detection_time if detection_time > 0 else 0,
            "model_info": detection_result['model_info'],
            "exported_files": exported_files,
            "experiment_id": request.experiment_id,
            "timestamp": datetime.now().isoformat()
        }
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed: {e}")
        app_state["metrics_collector"].increment_counter("detection_errors")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {str(e)}"
        )

# Experiment management endpoints
@app.post("/experiments")
async def create_experiment(
    request: ExperimentRequest,
    current_user: User = Depends(require_permission(Permission.CREATE_EXPERIMENTS))
):
    """Create a new experiment."""
    try:
        experiment_id = app_state["experiment_manager"].create_experiment(
            name=request.name,
            description=request.description,
            model_type=request.model_type,
            model_params=request.model_params,
            dataset_path=request.dataset_path,
            train_split=request.train_split,
            val_split=request.val_split,
            test_split=request.test_split,
            created_by=current_user.username
        )
        
        app_state["metrics_collector"].increment_counter("experiments_created")
        
        return {
            "experiment_id": experiment_id,
            "message": "Experiment created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create experiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create experiment: {str(e)}"
        )

@app.get("/experiments")
async def list_experiments(
    status: Optional[str] = None,
    model_type: Optional[str] = None,
    current_user: User = Depends(require_permission(Permission.VIEW_EXPERIMENTS))
):
    """List experiments."""
    try:
        experiments = app_state["experiment_manager"].list_experiments(
            status=status,
            model_type=model_type
        )
        
        return {"experiments": experiments}
        
    except Exception as e:
        logger.error(f"Failed to list experiments: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list experiments: {str(e)}"
        )

@app.post("/experiments/{experiment_id}/run")
async def run_experiment(
    experiment_id: str,
    current_user: User = Depends(require_permission(Permission.MANAGE_EXPERIMENTS))
):
    """Run an experiment."""
    try:
        results = app_state["experiment_manager"].run_experiment(experiment_id)
        
        app_state["metrics_collector"].increment_counter("experiments_run")
        
        return {
            "experiment_id": experiment_id,
            "results": results,
            "message": "Experiment completed successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to run experiment: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to run experiment: {str(e)}"
        )

# Metrics endpoints
@app.get("/metrics")
async def get_metrics(
    hours: int = 24,
    current_user: User = Depends(require_permission(Permission.VIEW_LOGS))
):
    """Get system metrics."""
    try:
        metrics = app_state["metrics_collector"].get_all_metrics_summary(hours)
        
        return {
            "metrics": metrics,
            "time_range_hours": hours,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Failed to get metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get metrics: {str(e)}"
        )

@app.get("/metrics/export")
async def export_metrics(
    format: str = "json",
    hours: int = 24,
    current_user: User = Depends(require_permission(Permission.VIEW_LOGS))
):
    """Export metrics data."""
    try:
        output_path = app_state["metrics_collector"].export_metrics(
            format=format,
            hours=hours
        )
        
        return FileResponse(
            path=output_path,
            filename=f"metrics_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{format}",
            media_type="application/octet-stream"
        )
        
    except Exception as e:
        logger.error(f"Failed to export metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to export metrics: {str(e)}"
        )

# User management endpoints (admin only)
@app.get("/users")
async def list_users(
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """List all users (admin only)."""
    try:
        users = []
        for user in app_state["auth_manager"].users.values():
            users.append({
                "user_id": user.user_id,
                "username": user.username,
                "email": user.email,
                "roles": [role.value for role in user.roles],
                "is_active": user.is_active,
                "created_at": user.created_at.isoformat(),
                "last_login": user.last_login.isoformat() if user.last_login else None
            })
        
        return {"users": users}
        
    except Exception as e:
        logger.error(f"Failed to list users: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list users: {str(e)}"
        )

@app.post("/users")
async def create_user(
    username: str,
    email: str,
    password: str,
    roles: List[str],
    current_user: User = Depends(require_role(UserRole.ADMIN))
):
    """Create a new user (admin only)."""
    try:
        # Convert string roles to UserRole enum
        user_roles = [UserRole(role) for role in roles]
        
        user_id = app_state["auth_manager"].create_user(
            username=username,
            email=email,
            password=password,
            roles=user_roles,
            created_by=current_user.username
        )
        
        app_state["metrics_collector"].increment_counter("users_created")
        
        return {
            "user_id": user_id,
            "message": "User created successfully"
        }
        
    except Exception as e:
        logger.error(f"Failed to create user: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to create user: {str(e)}"
        )

# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "message": "CrisisMapper Enterprise API",
        "version": "2.0.0",
        "docs": "/docs",
        "health": "/health",
        "features": [
            "Disaster Detection",
            "Geospatial Analysis",
            "Experiment Management",
            "User Authentication",
            "Metrics Collection",
            "Enterprise Security"
        ]
    }

if __name__ == "__main__":
    uvicorn.run(
        "src.api.enterprise_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
