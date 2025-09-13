"""
Streamlit Dashboard for CrisisMapper.

This module provides an interactive web dashboard for disaster detection,
visualization, and result analysis.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import logging
from datetime import datetime, timedelta
import asyncio
import concurrent.futures
from dataclasses import dataclass
import base64
import io

from ..core.detector import DisasterDetector
from ..core.classifier import DisasterClassifier
from ..geospatial.processor import GeospatialProcessor
from ..geospatial.export import GeospatialExporter
from ..utils.config import load_config
from ..utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class DashboardMetrics:
    """Dashboard metrics data class."""
    total_detections: int
    accuracy_score: float
    processing_time: float
    model_performance: Dict[str, float]
    disaster_distribution: Dict[str, int]
    severity_breakdown: Dict[str, int]
    geographic_coverage: float
    confidence_distribution: List[float]
    risk_assessment: Dict[str, float]


def create_dashboard():
    """Create and configure the Streamlit dashboard."""
    
    # Page configuration
    st.set_page_config(
        page_title="CrisisMapper Enterprise",
        page_icon="🌍",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # Import and use enterprise dashboard
    from .enterprise_dashboard import create_dashboard as create_enterprise_dashboard
    create_enterprise_dashboard()
    
    # Custom CSS
    st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .disaster-card {
        border-left: 4px solid #ff6b6b;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #fff5f5;
    }
    </style>
    """, unsafe_allow_html=True)
    
    # Main header
    st.markdown('<h1 class="main-header">🌍 CrisisMapper Dashboard</h1>', unsafe_allow_html=True)
    st.markdown("AI-powered geospatial mapping system for disaster detection and classification")
    
    # Sidebar
    with st.sidebar:
        st.header("Navigation")
        page = st.selectbox(
            "Select Page",
            ["Detection", "Data Management", "Analytics", "Settings"]
        )
    
    # Load configuration
    try:
        config = load_config()
    except Exception as e:
        st.error(f"Failed to load configuration: {e}")
        return
    
    # Initialize components
    if 'detector' not in st.session_state:
        try:
            st.session_state.detector = DisasterDetector(config)
            st.session_state.classifier = DisasterClassifier(config)
            st.session_state.geospatial_processor = GeospatialProcessor(config)
            st.session_state.exporter = GeospatialExporter(config)
        except Exception as e:
            st.error(f"Failed to initialize components: {e}")
            return
    
    # Route to appropriate page
    if page == "Detection":
        detection_page(st.session_state.detector, st.session_state.classifier, config)
    elif page == "Data Management":
        data_management_page(config)
    elif page == "Analytics":
        analytics_page(st.session_state.detector, config)
    elif page == "Settings":
        settings_page(config)


def detection_page(detector: DisasterDetector, classifier: DisasterClassifier, config: Dict):
    """Detection page for running disaster detection."""
    
    st.header("🔍 Disaster Detection")
    
    # Detection parameters
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Detection Parameters")
        confidence_threshold = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=config['model']['confidence_threshold'],
            step=0.05
        )
        
        iou_threshold = st.slider(
            "IoU Threshold",
            min_value=0.0,
            max_value=1.0,
            value=config['model']['iou_threshold'],
            step=0.05
        )
        
        max_detections = st.number_input(
            "Max Detections",
            min_value=1,
            max_value=10000,
            value=config['model']['max_detections']
        )
    
    with col2:
        st.subheader("Input Options")
        input_type = st.radio(
            "Input Type",
            ["Upload Image", "Image Path", "Batch Processing"]
        )
        
        if input_type == "Upload Image":
            uploaded_file = st.file_uploader(
                "Choose an image file",
                type=['jpg', 'jpeg', 'png', 'tiff', 'tif']
            )
            image_path = None
            batch_paths = None
            
        elif input_type == "Image Path":
            image_path = st.text_input("Enter image path")
            uploaded_file = None
            batch_paths = None
            
        else:  # Batch Processing
            batch_dir = st.text_input("Enter directory path for batch processing")
            if batch_dir and Path(batch_dir).exists():
                image_extensions = config['data_processing']['input_formats']
                batch_paths = list(Path(batch_dir).glob(f"*{image_extensions[0]}"))
                for ext in image_extensions[1:]:
                    batch_paths.extend(Path(batch_dir).glob(f"*{ext}"))
                st.info(f"Found {len(batch_paths)} images")
            else:
                batch_paths = None
            uploaded_file = None
            image_path = None
    
    # Export options
    st.subheader("Export Options")
    col1, col2 = st.columns(2)
    
    with col1:
        save_results = st.checkbox("Save Results", value=True)
        export_geojson = st.checkbox("Export GeoJSON", value=True)
        export_shapefile = st.checkbox("Export Shapefile", value=False)
        export_kml = st.checkbox("Export KML", value=False)
    
    with col2:
        show_confidence = st.checkbox("Show Confidence Scores", value=True)
        show_severity = st.checkbox("Show Severity Levels", value=True)
        show_risk_scores = st.checkbox("Show Risk Scores", value=True)
    
    # Run detection
    if st.button("🚀 Run Detection", type="primary"):
        if not any([uploaded_file, image_path, batch_paths]):
            st.error("Please provide an image input")
            return
        
        # Update detector parameters
        detector.config['model']['confidence_threshold'] = confidence_threshold
        detector.config['model']['iou_threshold'] = iou_threshold
        detector.config['model']['max_detections'] = max_detections
        
        # Prepare export formats
        export_formats = []
        if export_geojson:
            export_formats.append("geojson")
        if export_shapefile:
            export_formats.append("shapefile")
        if export_kml:
            export_formats.append("kml")
        
        # Run detection
        with st.spinner("Running detection..."):
            if uploaded_file:
                # Save uploaded file
                temp_path = Path("temp") / uploaded_file.name
                temp_path.parent.mkdir(exist_ok=True)
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                try:
                    result = detector.detect(str(temp_path), save_results=save_results)
                    display_detection_results(result, classifier, show_confidence, show_severity, show_risk_scores)
                finally:
                    # Clean up temp file
                    temp_path.unlink()
                    
            elif image_path:
                if not Path(image_path).exists():
                    st.error(f"Image not found: {image_path}")
                    return
                
                result = detector.detect(image_path, save_results=save_results)
                display_detection_results(result, classifier, show_confidence, show_severity, show_risk_scores)
                
            elif batch_paths:
                results = detector.detect_batch([str(p) for p in batch_paths])
                display_batch_results(results, classifier, show_confidence, show_severity, show_risk_scores)


def display_detection_results(result: Dict, classifier: DisasterClassifier, 
                            show_confidence: bool, show_severity: bool, show_risk_scores: bool):
    """Display detection results."""
    
    st.header("📊 Detection Results")
    
    # Performance metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Inference Time", f"{result['inference_time']:.3f}s")
    with col2:
        st.metric("FPS", f"{result['fps']:.2f}")
    with col3:
        st.metric("Detections", len(result['detections']['boxes']))
    with col4:
        st.metric("Model", result['model_info']['name'])
    
    # Classification results
    if result['detections']['boxes']:
        classification_result = classifier.classify_detection(result)
        
        # Summary statistics
        st.subheader("📈 Summary Statistics")
        summary = classification_result['summary']
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Detections", summary['total_detections'])
            st.metric("Average Confidence", f"{summary['average_confidence']:.3f}")
            st.metric("Total Coverage", f"{summary['total_coverage']:.1f}%")
        
        with col2:
            st.metric("Average Risk Score", f"{summary['average_risk_score']:.3f}")
            st.metric("Max Risk Score", f"{summary['max_risk_score']:.3f}")
            st.metric("Min Risk Score", f"{summary['min_risk_score']:.3f}")
        
        # Class distribution
        st.subheader("🏷️ Class Distribution")
        class_counts = summary['class_counts']
        
        if class_counts:
            fig = px.pie(
                values=list(class_counts.values()),
                names=list(class_counts.keys()),
                title="Detection Count by Class"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Severity distribution
        if show_severity and summary['severity_counts']:
            st.subheader("⚠️ Severity Distribution")
            severity_counts = summary['severity_counts']
            
            fig = px.bar(
                x=list(severity_counts.keys()),
                y=list(severity_counts.values()),
                title="Detection Count by Severity"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Detailed detections table
        st.subheader("🔍 Detailed Detections")
        
        # Create detections dataframe
        detections_data = []
        for i, det in enumerate(classification_result['enhanced_detections']):
            detection_data = {
                'ID': i,
                'Class': det['class_name'],
                'Confidence': f"{det['confidence']:.3f}",
                'Area (m²)': f"{det['area']:.1f}",
                'Coverage (%)': f"{det['coverage_percentage']:.1f}"
            }
            
            if show_severity:
                detection_data['Severity'] = det['severity']
            if show_risk_scores:
                detection_data['Risk Score'] = f"{det['risk_score']:.3f}"
            
            detections_data.append(detection_data)
        
        df = pd.DataFrame(detections_data)
        st.dataframe(df, use_container_width=True)
        
        # Export options
        if st.button("📥 Export Results"):
            st.info("Export functionality would be implemented here")
    
    else:
        st.info("No disasters detected in the image")


def display_batch_results(results: List[Dict], classifier: DisasterClassifier,
                         show_confidence: bool, show_severity: bool, show_risk_scores: bool):
    """Display batch detection results."""
    
    st.header("📊 Batch Detection Results")
    
    # Overall statistics
    total_images = len(results)
    successful = sum(1 for r in results if 'error' not in r)
    failed = total_images - successful
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Images", total_images)
    with col2:
        st.metric("Successful", successful)
    with col3:
        st.metric("Failed", failed)
    
    # Process each result
    for i, result in enumerate(results):
        with st.expander(f"Image {i+1}: {Path(result.get('image_path', 'Unknown')).name}"):
            if 'error' in result:
                st.error(f"Error: {result['error']}")
            else:
                display_detection_results(result, classifier, show_confidence, show_severity, show_risk_scores)


def data_management_page(config: Dict):
    """Data management page."""
    
    st.header("📁 Data Management")
    
    # Data summary
    st.subheader("Data Summary")
    
    # Placeholder for data summary
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Files", "0")
    with col2:
        st.metric("Total Size", "0 MB")
    with col3:
        st.metric("Raw Data", "0 files")
    with col4:
        st.metric("Processed Data", "0 files")
    
    # Data upload
    st.subheader("Upload Data")
    
    uploaded_files = st.file_uploader(
        "Choose files to upload",
        accept_multiple_files=True,
        type=['jpg', 'jpeg', 'png', 'tiff', 'tif', 'geojson', 'shp']
    )
    
    if uploaded_files:
        st.write(f"Uploaded {len(uploaded_files)} files")
        
        if st.button("Process Uploaded Files"):
            st.info("File processing would be implemented here")
    
    # Data sources
    st.subheader("Data Sources")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**NASA EarthData**")
        st.write("Status: Not configured")
        
    with col2:
        st.write("**Sentinel-2**")
        st.write("Status: Not configured")


def analytics_page(detector: DisasterDetector, config: Dict):
    """Analytics page for performance and trend analysis."""
    
    st.header("📈 Analytics")
    
    # Performance metrics
    st.subheader("Performance Metrics")
    
    if detector:
        stats = detector.get_performance_stats()
        
        if stats:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Inferences", stats.get('total_inferences', 0))
            with col2:
                st.metric("Total Detections", stats.get('total_detections', 0))
            with col3:
                st.metric("Avg Inference Time", f"{stats.get('average_inference_time', 0):.3f}s")
            with col4:
                st.metric("Avg FPS", f"{stats.get('average_fps', 0):.2f}")
            
            # Performance chart
            if stats.get('total_inferences', 0) > 0:
                st.subheader("Performance Over Time")
                
                # Create dummy data for demonstration
                np.random.seed(42)
                n_points = min(20, stats['total_inferences'])
                times = np.random.normal(stats['average_inference_time'], 
                                       stats['std_inference_time'], n_points)
                
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=times,
                    mode='lines+markers',
                    name='Inference Time',
                    line=dict(color='blue')
                ))
                
                fig.update_layout(
                    title="Inference Time Over Time",
                    xaxis_title="Inference Number",
                    yaxis_title="Time (seconds)"
                )
                
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No performance data available. Run some detections to see metrics.")
    else:
        st.error("Detector not initialized")
    
    # Model information
    st.subheader("Model Information")
    
    model_info = {
        "Model Name": config['model']['name'],
        "Device": config['model']['device'],
        "Confidence Threshold": config['model']['confidence_threshold'],
        "IoU Threshold": config['model']['iou_threshold'],
        "Max Detections": config['model']['max_detections']
    }
    
    for key, value in model_info.items():
        st.write(f"**{key}**: {value}")


def settings_page(config: Dict):
    """Settings page for configuration management."""
    
    st.header("⚙️ Settings")
    
    # Model settings
    st.subheader("Model Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        model_name = st.selectbox(
            "Model Name",
            ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
            index=["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"].index(config['model']['name'])
        )
        
        device = st.selectbox(
            "Device",
            ["auto", "cpu", "cuda", "mps"],
            index=["auto", "cpu", "cuda", "mps"].index(config['model']['device'])
        )
    
    with col2:
        confidence = st.slider(
            "Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=config['model']['confidence_threshold'],
            step=0.05
        )
        
        iou = st.slider(
            "IoU Threshold",
            min_value=0.0,
            max_value=1.0,
            value=config['model']['iou_threshold'],
            step=0.05
        )
    
    # Data processing settings
    st.subheader("Data Processing Settings")
    
    col1, col2 = st.columns(2)
    
    with col1:
        tile_size = st.number_input(
            "Tile Size",
            min_value=256,
            max_value=2048,
            value=config['data_processing']['tile_size'],
            step=256
        )
        
        overlap = st.slider(
            "Overlap",
            min_value=0.0,
            max_value=0.5,
            value=config['data_processing']['overlap'],
            step=0.05
        )
    
    with col2:
        batch_size = st.number_input(
            "Batch Size",
            min_value=1,
            max_value=32,
            value=config['data_processing']['batch_size']
        )
        
        max_image_size = st.number_input(
            "Max Image Size",
            min_value=512,
            max_value=8192,
            value=config['data_processing']['max_image_size'],
            step=512
        )
    
    # Save settings
    if st.button("💾 Save Settings"):
        # Update configuration
        config['model']['name'] = model_name
        config['model']['device'] = device
        config['model']['confidence_threshold'] = confidence
        config['model']['iou_threshold'] = iou
        config['data_processing']['tile_size'] = tile_size
        config['data_processing']['overlap'] = overlap
        config['data_processing']['batch_size'] = batch_size
        config['data_processing']['max_image_size'] = max_image_size
        
        # Save to file
        import yaml
        with open("config/settings.yaml", "w") as f:
            yaml.dump(config, f, default_flow_style=False, indent=2)
        
        st.success("Settings saved successfully!")
        st.rerun()


if __name__ == "__main__":
    create_dashboard()
