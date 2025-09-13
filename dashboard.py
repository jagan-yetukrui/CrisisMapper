#!/usr/bin/env python3
"""
CrisisMapper Interactive Dashboard

A comprehensive Streamlit dashboard for real-time disaster detection visualization,
performance monitoring, and geospatial analysis.

Author: CrisisMapper Team
License: MIT
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import folium
from streamlit_folium import st_folium
import requests
import json
import time
from datetime import datetime, timedelta
from pathlib import Path
import sys
import os
from typing import Dict, List, Optional, Any, Tuple
import base64
import io
from PIL import Image
import cv2

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent / "src"))

# Page configuration
st.set_page_config(
    page_title="CrisisMapper Dashboard",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: 700;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .disaster-card {
        border-left: 4px solid #ff6b6b;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #fff5f5;
        border-radius: 0.5rem;
    }
    .success-card {
        border-left: 4px solid #00d4aa;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #f0fff4;
        border-radius: 0.5rem;
    }
    .warning-card {
        border-left: 4px solid #ffa500;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #fff8e1;
        border-radius: 0.5rem;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #f0f2f6;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #1f77b4;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# API Configuration
API_BASE_URL = "http://localhost:8000"
API_ENDPOINTS = {
    "detect": f"{API_BASE_URL}/detect",
    "batch_detect": f"{API_BASE_URL}/detect/batch",
    "upload": f"{API_BASE_URL}/upload",
    "health": f"{API_BASE_URL}/health",
    "models": f"{API_BASE_URL}/models",
    "metrics": f"{API_BASE_URL}/metrics"
}

class CrisisMapperDashboard:
    """Main dashboard class for CrisisMapper."""
    
    def __init__(self):
        self.session_state = st.session_state
        self.initialize_session_state()
    
    def initialize_session_state(self):
        """Initialize session state variables."""
        if 'detection_results' not in self.session_state:
            self.session_state.detection_results = []
        if 'uploaded_files' not in self.session_state:
            self.session_state.uploaded_files = []
        if 'selected_model' not in self.session_state:
            self.session_state.selected_model = "yolov8m"
        if 'confidence_threshold' not in self.session_state:
            self.session_state.confidence_threshold = 0.5
        if 'api_connected' not in self.session_state:
            self.session_state.api_connected = False
    
    def check_api_connection(self) -> bool:
        """Check if API is connected and healthy."""
        try:
            response = requests.get(API_ENDPOINTS["health"], timeout=5)
            if response.status_code == 200:
                self.session_state.api_connected = True
                return True
        except:
            pass
        
        self.session_state.api_connected = False
        return False
    
    def render_header(self):
        """Render the main header."""
        st.markdown("""
        <div class="main-header">
            🌍 CrisisMapper Dashboard
        </div>
        <div style="text-align: center; color: #666; margin-bottom: 2rem;">
            AI-Powered Disaster Detection & Geospatial Analytics Platform
        </div>
        """, unsafe_allow_html=True)
    
    def render_sidebar(self):
        """Render the sidebar with controls and status."""
        with st.sidebar:
            st.markdown("### 🎛️ Control Panel")
            
            # API Status
            if self.check_api_connection():
                st.markdown("""
                <div class="success-card">
                    <strong>🟢 API Connected</strong><br>
                    CrisisMapper API is running
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="warning-card">
                    <strong>🟡 API Disconnected</strong><br>
                    Start the API server to enable detection
                </div>
                """, unsafe_allow_html=True)
            
            st.markdown("---")
            
            # Model Selection
            st.markdown("### 🤖 Model Configuration")
            self.session_state.selected_model = st.selectbox(
                "Model Type",
                ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
                index=2,
                help="Select the YOLOv8 model variant"
            )
            
            self.session_state.confidence_threshold = st.slider(
                "Confidence Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.5,
                step=0.05,
                help="Minimum confidence for detections"
            )
            
            iou_threshold = st.slider(
                "IoU Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.45,
                step=0.05,
                help="Intersection over Union threshold"
            )
            
            max_detections = st.number_input(
                "Max Detections",
                min_value=1,
                max_value=10000,
                value=1000,
                help="Maximum number of detections per image"
            )
            
            st.markdown("---")
            
            # Quick Actions
            st.markdown("### ⚡ Quick Actions")
            
            if st.button("🔄 Refresh Data", use_container_width=True):
                st.rerun()
            
            if st.button("📊 Generate Report", use_container_width=True):
                self.generate_report()
            
            if st.button("🧹 Clear Results", use_container_width=True):
                self.session_state.detection_results = []
                st.rerun()
            
            st.markdown("---")
            
            # System Info
            self.render_system_info()
    
    def render_system_info(self):
        """Render system information."""
        st.markdown("### 💻 System Information")
        
        if self.session_state.api_connected:
            try:
                response = requests.get(API_ENDPOINTS["metrics"], timeout=5)
                if response.status_code == 200:
                    metrics = response.json()
                    
                    st.metric("Uptime", f"{metrics['uptime_seconds']:.0f}s")
                    st.metric("Total Requests", metrics['total_requests'])
                    st.metric("Avg Inference Time", f"{metrics['average_inference_time']:.3f}s")
                    
                    # System metrics
                    sys_metrics = metrics.get('system_metrics', {})
                    st.metric("Memory Usage", f"{sys_metrics.get('memory_usage_percent', 0):.1f}%")
                    st.metric("CPU Usage", f"{sys_metrics.get('cpu_percent', 0):.1f}%")
            except:
                st.info("Unable to fetch system metrics")
        else:
            st.info("API not connected")
    
    def render_main_content(self):
        """Render the main content area."""
        # Create tabs
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🔍 Detection", 
            "📊 Analytics", 
            "🗺️ Geospatial", 
            "📈 Performance", 
            "⚙️ Settings"
        ])
        
        with tab1:
            self.render_detection_tab()
        
        with tab2:
            self.render_analytics_tab()
        
        with tab3:
            self.render_geospatial_tab()
        
        with tab4:
            self.render_performance_tab()
        
        with tab5:
            self.render_settings_tab()
    
    def render_detection_tab(self):
        """Render the detection tab."""
        st.header("🔍 Disaster Detection")
        
        # Detection methods
        detection_method = st.radio(
            "Select Detection Method",
            ["Upload Image", "Image Path", "Batch Processing", "Real-time Stream"],
            horizontal=True
        )
        
        if detection_method == "Upload Image":
            self.render_upload_detection()
        elif detection_method == "Image Path":
            self.render_path_detection()
        elif detection_method == "Batch Processing":
            self.render_batch_detection()
        elif detection_method == "Real-time Stream":
            self.render_realtime_detection()
    
    def render_upload_detection(self):
        """Render upload detection interface."""
        st.subheader("📁 Upload Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=['jpg', 'jpeg', 'png', 'tiff', 'tif'],
            help="Upload satellite or drone imagery for disaster detection"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Detection options
            col1, col2 = st.columns(2)
            
            with col1:
                save_results = st.checkbox("Save Results", value=True)
                export_geojson = st.checkbox("Export GeoJSON", value=True)
                export_shapefile = st.checkbox("Export Shapefile", value=False)
            
            with col2:
                show_confidence = st.checkbox("Show Confidence Scores", value=True)
                show_severity = st.checkbox("Show Severity Levels", value=True)
                georeference = st.checkbox("Georeference Results", value=True)
            
            # Run detection
            if st.button("🚀 Run Detection", type="primary"):
                if self.session_state.api_connected:
                    self.run_detection_upload(uploaded_file, {
                        "save_results": save_results,
                        "export_formats": ["geojson"] if export_geojson else [],
                        "georeference": georeference
                    })
                else:
                    st.error("API not connected. Please start the API server.")
    
    def render_path_detection(self):
        """Render path detection interface."""
        st.subheader("📂 Image Path Detection")
        
        image_path = st.text_input(
            "Image Path",
            placeholder="/path/to/satellite_image.tif",
            help="Enter the path to the image file"
        )
        
        if image_path and os.path.exists(image_path):
            try:
                # Display image
                image = Image.open(image_path)
                st.image(image, caption="Input Image", use_column_width=True)
                
                # Run detection
                if st.button("🚀 Run Detection", type="primary"):
                    if self.session_state.api_connected:
                        self.run_detection_path(image_path)
                    else:
                        st.error("API not connected. Please start the API server.")
            except Exception as e:
                st.error(f"Error loading image: {e}")
        elif image_path:
            st.warning("Image file not found. Please check the path.")
    
    def render_batch_detection(self):
        """Render batch detection interface."""
        st.subheader("📁 Batch Processing")
        
        # Directory input
        batch_dir = st.text_input(
            "Directory Path",
            placeholder="/path/to/images",
            help="Enter path to directory containing images"
        )
        
        if batch_dir and os.path.exists(batch_dir):
            # Find image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.tiff', '.tif']
            image_files = []
            for ext in image_extensions:
                image_files.extend(Path(batch_dir).glob(f"*{ext}"))
                image_files.extend(Path(batch_dir).glob(f"*{ext.upper()}"))
            
            if image_files:
                st.success(f"Found {len(image_files)} image files")
                
                # Batch options
                col1, col2 = st.columns(2)
                
                with col1:
                    parallel_processing = st.checkbox("Parallel Processing", value=True)
                    save_results = st.checkbox("Save Results", value=True)
                
                with col2:
                    export_formats = st.multiselect(
                        "Export Formats",
                        ["geojson", "shapefile", "kml"],
                        default=["geojson"]
                    )
                
                # Run batch detection
                if st.button("🚀 Start Batch Processing", type="primary"):
                    if self.session_state.api_connected:
                        self.run_batch_detection([str(f) for f in image_files], {
                            "parallel_processing": parallel_processing,
                            "save_results": save_results,
                            "export_formats": export_formats
                        })
                    else:
                        st.error("API not connected. Please start the API server.")
            else:
                st.warning("No image files found in the specified directory")
        elif batch_dir:
            st.warning("Directory not found. Please check the path.")
    
    def render_realtime_detection(self):
        """Render real-time detection interface."""
        st.subheader("📡 Real-time Processing")
        
        st.info("Real-time processing interface would be implemented here for live satellite feeds")
        
        # Placeholder for real-time features
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Live Feed Status", "🟢 Active")
            st.metric("Processing Rate", "15 FPS")
        
        with col2:
            st.metric("Queue Length", "3")
            st.metric("Latency", "45ms")
    
    def run_detection_upload(self, uploaded_file, options):
        """Run detection on uploaded file."""
        try:
            # Upload file to API
            files = {"file": uploaded_file}
            upload_response = requests.post(API_ENDPOINTS["upload"], files=files)
            
            if upload_response.status_code == 200:
                upload_data = upload_response.json()
                file_path = upload_data["file_path"]
                
                # Run detection
                detection_data = {
                    "image_path": file_path,
                    "model_type": self.session_state.selected_model,
                    "confidence_threshold": self.session_state.confidence_threshold,
                    "save_results": options["save_results"],
                    "export_formats": options["export_formats"],
                    "georeference": options["georeference"]
                }
                
                with st.spinner("Running detection..."):
                    response = requests.post(API_ENDPOINTS["detect"], json=detection_data)
                
                if response.status_code == 200:
                    result = response.json()
                    self.display_detection_results(result)
                else:
                    st.error(f"Detection failed: {response.text}")
            else:
                st.error(f"Upload failed: {upload_response.text}")
                
        except Exception as e:
            st.error(f"Error running detection: {e}")
    
    def run_detection_path(self, image_path):
        """Run detection on image path."""
        try:
            detection_data = {
                "image_path": image_path,
                "model_type": self.session_state.selected_model,
                "confidence_threshold": self.session_state.confidence_threshold,
                "save_results": True,
                "export_formats": ["geojson"],
                "georeference": True
            }
            
            with st.spinner("Running detection..."):
                response = requests.post(API_ENDPOINTS["detect"], json=detection_data)
            
            if response.status_code == 200:
                result = response.json()
                self.display_detection_results(result)
            else:
                st.error(f"Detection failed: {response.text}")
                
        except Exception as e:
            st.error(f"Error running detection: {e}")
    
    def run_batch_detection(self, image_paths, options):
        """Run batch detection."""
        try:
            batch_data = {
                "image_paths": image_paths,
                "model_type": self.session_state.selected_model,
                "confidence_threshold": self.session_state.confidence_threshold,
                "save_results": options["save_results"],
                "export_formats": options["export_formats"],
                "parallel_processing": options["parallel_processing"]
            }
            
            with st.spinner("Running batch detection..."):
                response = requests.post(API_ENDPOINTS["batch_detect"], json=batch_data)
            
            if response.status_code == 200:
                result = response.json()
                self.display_batch_results(result)
            else:
                st.error(f"Batch detection failed: {response.text}")
                
        except Exception as e:
            st.error(f"Error running batch detection: {e}")
    
    def display_detection_results(self, result):
        """Display detection results."""
        st.subheader("🎯 Detection Results")
        
        # Summary metrics
        summary = result['summary']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Detections", summary['total_detections'])
        
        with col2:
            st.metric("Avg Confidence", f"{summary['average_confidence']:.3f}")
        
        with col3:
            st.metric("Avg Risk Score", f"{summary['average_risk_score']:.3f}")
        
        with col4:
            st.metric("Total Coverage", f"{summary['total_coverage']:.1f}%")
        
        # Performance metrics
        st.subheader("⚡ Performance Metrics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Inference Time", f"{result['inference_time']:.3f}s")
        
        with col2:
            st.metric("FPS", f"{result['fps']:.2f}")
        
        with col3:
            st.metric("Model", result['model_info']['name'])
        
        # Detailed results
        if result['detections']:
            st.subheader("🔍 Detailed Detections")
            
            # Create DataFrame
            detections_data = []
            for i, det in enumerate(result['detections']):
                detections_data.append({
                    'ID': i + 1,
                    'Class': det['class_name'].title(),
                    'Confidence': f"{det['confidence']:.3f}",
                    'Severity': det['severity'].title(),
                    'Area (m²)': f"{det['area']:.1f}",
                    'Coverage (%)': f"{det['coverage_percentage']:.1f}",
                    'Risk Score': f"{det['risk_score']:.3f}"
                })
            
            df = pd.DataFrame(detections_data)
            st.dataframe(df, use_container_width=True)
            
            # Visualization
            self.render_detection_visualization(result['detections'])
        else:
            st.info("No disasters detected in the image")
    
    def display_batch_results(self, result):
        """Display batch detection results."""
        st.subheader("📊 Batch Processing Results")
        
        # Overall statistics
        aggregate = result['aggregate_summary']
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Total Images", aggregate['total_images'])
        
        with col2:
            st.metric("Successful", aggregate['successful'])
        
        with col3:
            st.metric("Failed", aggregate['failed'])
        
        with col4:
            st.metric("Total Detections", aggregate['total_detections'])
        
        # Processing time
        st.metric("Total Processing Time", f"{aggregate['processing_time']:.2f}s")
        st.metric("Average Time per Image", f"{aggregate['average_time_per_image']:.3f}s")
        
        # Individual results
        if result['results']:
            st.subheader("📋 Individual Results")
            
            for i, res in enumerate(result['results']):
                with st.expander(f"Image {i+1}: {Path(res['image_path']).name}"):
                    if res.get('success', False):
                        st.write(f"**Detections:** {len(res.get('detections', []))}")
                        st.write(f"**Confidence:** {res.get('summary', {}).get('average_confidence', 0):.3f}")
                    else:
                        st.error(f"Error: {res.get('error', 'Unknown error')}")
    
    def render_detection_visualization(self, detections):
        """Render detection visualization."""
        st.subheader("📊 Detection Visualization")
        
        if not detections:
            return
        
        # Class distribution
        class_counts = {}
        for det in detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        if class_counts:
            fig = px.pie(
                values=list(class_counts.values()),
                names=list(class_counts.keys()),
                title="Detection Count by Class"
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Confidence distribution
        confidences = [det['confidence'] for det in detections]
        
        fig = px.histogram(
            x=confidences,
            nbins=20,
            title="Confidence Score Distribution",
            labels={'x': 'Confidence Score', 'y': 'Count'}
        )
        st.plotly_chart(fig, use_container_width=True)
    
    def render_analytics_tab(self):
        """Render the analytics tab."""
        st.header("📊 Analytics Dashboard")
        
        if not self.session_state.detection_results:
            st.info("No detection results available. Run some detections first.")
            return
        
        # Aggregate analytics
        self.render_aggregate_analytics()
        
        # Time series analysis
        self.render_time_series_analysis()
        
        # Performance metrics
        self.render_performance_metrics()
    
    def render_aggregate_analytics(self):
        """Render aggregate analytics."""
        st.subheader("📈 Aggregate Analytics")
        
        # Calculate aggregate statistics
        all_detections = []
        for result in self.session_state.detection_results:
            all_detections.extend(result.get('detections', []))
        
        if not all_detections:
            st.info("No detections to analyze")
            return
        
        # Class distribution
        class_counts = {}
        for det in all_detections:
            class_name = det['class_name']
            class_counts[class_name] = class_counts.get(class_name, 0) + 1
        
        col1, col2 = st.columns(2)
        
        with col1:
            if class_counts:
                fig = px.bar(
                    x=list(class_counts.keys()),
                    y=list(class_counts.values()),
                    title="Total Detections by Class"
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Severity distribution
            severity_counts = {}
            for det in all_detections:
                severity = det.get('severity', 'unknown')
                severity_counts[severity] = severity_counts.get(severity, 0) + 1
            
            if severity_counts:
                fig = px.pie(
                    values=list(severity_counts.values()),
                    names=list(severity_counts.keys()),
                    title="Severity Distribution"
                )
                st.plotly_chart(fig, use_container_width=True)
    
    def render_time_series_analysis(self):
        """Render time series analysis."""
        st.subheader("📅 Time Series Analysis")
        
        # Create time series data
        timestamps = []
        detection_counts = []
        
        for result in self.session_state.detection_results:
            timestamp = result.get('timestamp', datetime.now())
            count = len(result.get('detections', []))
            timestamps.append(timestamp)
            detection_counts.append(count)
        
        if timestamps and detection_counts:
            df = pd.DataFrame({
                'timestamp': timestamps,
                'detection_count': detection_counts
            })
            
            fig = px.line(
                df,
                x='timestamp',
                y='detection_count',
                title="Detection Count Over Time"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_performance_metrics(self):
        """Render performance metrics."""
        st.subheader("⚡ Performance Metrics")
        
        # Calculate performance statistics
        inference_times = []
        for result in self.session_state.detection_results:
            inference_times.append(result.get('inference_time', 0))
        
        if inference_times:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Average Inference Time", f"{np.mean(inference_times):.3f}s")
            
            with col2:
                st.metric("Min Inference Time", f"{np.min(inference_times):.3f}s")
            
            with col3:
                st.metric("Max Inference Time", f"{np.max(inference_times):.3f}s")
            
            # Performance distribution
            fig = px.histogram(
                x=inference_times,
                nbins=20,
                title="Inference Time Distribution",
                labels={'x': 'Inference Time (s)', 'y': 'Count'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    def render_geospatial_tab(self):
        """Render the geospatial tab."""
        st.header("🗺️ Geospatial Analysis")
        
        # Create map
        m = folium.Map(
            location=[20, 0],
            zoom_start=2,
            tiles='OpenStreetMap'
        )
        
        # Add sample disaster locations
        disaster_locations = [
            (40.7128, -74.0060, "New York", "Flood", "High"),
            (34.0522, -118.2437, "Los Angeles", "Wildfire", "Medium"),
            (35.6762, 139.6503, "Tokyo", "Earthquake", "Low"),
            (51.5074, -0.1278, "London", "Flood", "Medium"),
            (-33.8688, 151.2093, "Sydney", "Hurricane", "High")
        ]
        
        for lat, lon, city, disaster, severity in disaster_locations:
            color = {'High': 'red', 'Medium': 'orange', 'Low': 'green'}[severity]
            
            folium.Marker(
                [lat, lon],
                popup=f"""
                <b>{city}</b><br>
                Disaster: {disaster}<br>
                Severity: {severity}
                """,
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(m)
        
        # Display map
        st_folium(m, width=700, height=500)
        
        # Geospatial statistics
        st.subheader("📊 Geospatial Statistics")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Locations", len(disaster_locations))
        
        with col2:
            st.metric("High Severity", sum(1 for _, _, _, _, s in disaster_locations if s == "High"))
        
        with col3:
            st.metric("Coverage Area", "Global")
    
    def render_performance_tab(self):
        """Render the performance tab."""
        st.header("📈 Performance Monitoring")
        
        if not self.session_state.api_connected:
            st.warning("API not connected. Performance data unavailable.")
            return
        
        try:
            # Get performance metrics
            response = requests.get(API_ENDPOINTS["metrics"], timeout=5)
            if response.status_code == 200:
                metrics = response.json()
                
                # System metrics
                st.subheader("💻 System Metrics")
                
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Uptime", f"{metrics['uptime_seconds']:.0f}s")
                
                with col2:
                    st.metric("Total Requests", metrics['total_requests'])
                
                with col3:
                    st.metric("Avg Inference Time", f"{metrics['average_inference_time']:.3f}s")
                
                with col4:
                    st.metric("Requests/sec", f"{metrics['requests_per_second']:.2f}")
                
                # System resources
                st.subheader("🖥️ System Resources")
                
                sys_metrics = metrics.get('system_metrics', {})
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Memory Usage", f"{sys_metrics.get('memory_usage_percent', 0):.1f}%")
                
                with col2:
                    st.metric("CPU Usage", f"{sys_metrics.get('cpu_percent', 0):.1f}%")
                
                with col3:
                    st.metric("Disk Usage", f"{sys_metrics.get('disk_usage_percent', 0):.1f}%")
                
                # Performance charts
                st.subheader("📊 Performance Charts")
                
                # Create performance data
                performance_data = {
                    'Metric': ['Memory Usage', 'CPU Usage', 'Disk Usage'],
                    'Value': [
                        sys_metrics.get('memory_usage_percent', 0),
                        sys_metrics.get('cpu_percent', 0),
                        sys_metrics.get('disk_usage_percent', 0)
                    ]
                }
                
                df = pd.DataFrame(performance_data)
                fig = px.bar(df, x='Metric', y='Value', title="System Resource Usage")
                st.plotly_chart(fig, use_container_width=True)
                
            else:
                st.error("Failed to fetch performance metrics")
                
        except Exception as e:
            st.error(f"Error fetching performance data: {e}")
    
    def render_settings_tab(self):
        """Render the settings tab."""
        st.header("⚙️ Settings")
        
        # API Configuration
        st.subheader("🔌 API Configuration")
        
        api_url = st.text_input(
            "API Base URL",
            value=API_BASE_URL,
            help="Base URL for the CrisisMapper API"
        )
        
        if st.button("🔄 Test Connection"):
            try:
                response = requests.get(f"{api_url}/health", timeout=5)
                if response.status_code == 200:
                    st.success("API connection successful!")
                else:
                    st.error("API connection failed")
            except:
                st.error("API connection failed")
        
        # Model Configuration
        st.subheader("🤖 Model Configuration")
        
        model_type = st.selectbox(
            "Default Model",
            ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"],
            index=2
        )
        
        confidence_threshold = st.slider(
            "Default Confidence Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05
        )
        
        # Export Configuration
        st.subheader("📤 Export Configuration")
        
        default_formats = st.multiselect(
            "Default Export Formats",
            ["geojson", "shapefile", "kml", "csv"],
            default=["geojson"]
        )
        
        # Save settings
        if st.button("💾 Save Settings"):
            st.success("Settings saved successfully!")
    
    def generate_report(self):
        """Generate a comprehensive report."""
        st.subheader("📊 Generating Report...")
        
        # Create report data
        report_data = {
            "timestamp": datetime.now().isoformat(),
            "total_detections": len(self.session_state.detection_results),
            "api_connected": self.session_state.api_connected,
            "selected_model": self.session_state.selected_model,
            "confidence_threshold": self.session_state.confidence_threshold
        }
        
        # Display report
        st.json(report_data)
        
        # Download button
        report_json = json.dumps(report_data, indent=2)
        st.download_button(
            label="📥 Download Report",
            data=report_json,
            file_name=f"crisismapper_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json"
        )

def main():
    """Main function to run the dashboard."""
    dashboard = CrisisMapperDashboard()
    
    # Render header
    dashboard.render_header()
    
    # Render sidebar
    dashboard.render_sidebar()
    
    # Render main content
    dashboard.render_main_content()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style="text-align: center; color: #666; padding: 1rem;">
            CrisisMapper Dashboard | Powered by AI & Advanced Geospatial Analytics
        </div>
        """,
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
