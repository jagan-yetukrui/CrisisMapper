"""
Data Ingestion Module for CrisisMapper.

This module handles data ingestion from various sources including
NASA EarthData, Sentinel-2, and drone data.
"""

import os
import asyncio
import aiohttp
import requests
from pathlib import Path
from typing import List, Dict, Optional, Union, Tuple
import logging
from datetime import datetime, timedelta
import json

from ..utils.logger import setup_logger
from ..utils.file_utils import ensure_dir, get_file_info

logger = setup_logger(__name__)


class DataIngestion:
    """
    Main data ingestion class for CrisisMapper.
    
    Handles downloading and processing data from various sources
    including satellite imagery, drone data, and reference datasets.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the DataIngestion.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.data_dir = Path(config['output']['base_dir']) / "data"
        self.raw_dir = self.data_dir / "raw"
        self.processed_dir = self.data_dir / "processed"
        
        # Ensure directories exist
        ensure_dir(self.raw_dir)
        ensure_dir(self.processed_dir)
        
        # Initialize data sources
        self.sources = self._initialize_sources()
        
        logger.info(f"DataIngestion initialized with data dir: {self.data_dir}")
    
    def _initialize_sources(self) -> Dict:
        """Initialize data source handlers."""
        sources = {}
        
        # NASA EarthData
        if self.config['data_sources']['nasa_earthdata']['username']:
            sources['nasa'] = NASADataSource(self.config)
        
        # Sentinel-2
        if self.config['data_sources']['sentinel2']['username']:
            sources['sentinel2'] = Sentinel2DataSource(self.config)
        
        # OpenStreetMap
        sources['osm'] = OpenStreetMapSource(self.config)
        
        return sources
    
    async def download_satellite_data(self, 
                                    bbox: Tuple[float, float, float, float],
                                    start_date: str,
                                    end_date: str,
                                    sources: List[str] = None) -> Dict[str, List[str]]:
        """
        Download satellite data for a given bounding box and date range.
        
        Args:
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            start_date: Start date in YYYY-MM-DD format
            end_date: End date in YYYY-MM-DD format
            sources: List of sources to use (default: all available)
            
        Returns:
            Dictionary mapping source names to lists of downloaded files
        """
        if sources is None:
            sources = list(self.sources.keys())
        
        results = {}
        
        # Download from each source
        for source_name in sources:
            if source_name not in self.sources:
                logger.warning(f"Source {source_name} not available")
                continue
            
            try:
                logger.info(f"Downloading data from {source_name}")
                source = self.sources[source_name]
                
                if hasattr(source, 'download_data'):
                    files = await source.download_data(bbox, start_date, end_date)
                    results[source_name] = files
                else:
                    logger.warning(f"Source {source_name} does not support download_data")
                    
            except Exception as e:
                logger.error(f"Error downloading from {source_name}: {e}")
                results[source_name] = []
        
        return results
    
    def process_downloaded_data(self, 
                              source_files: Dict[str, List[str]],
                              output_format: str = "processed") -> Dict[str, List[str]]:
        """
        Process downloaded data for use in disaster detection.
        
        Args:
            source_files: Dictionary of source files to process
            output_format: Output format for processed data
            
        Returns:
            Dictionary mapping sources to processed file paths
        """
        processed_files = {}
        
        for source_name, files in source_files.items():
            if not files:
                continue
            
            logger.info(f"Processing {len(files)} files from {source_name}")
            
            source_processed = []
            for file_path in files:
                try:
                    processed_path = self._process_single_file(file_path, output_format)
                    source_processed.append(processed_path)
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
            
            processed_files[source_name] = source_processed
        
        return processed_files
    
    def _process_single_file(self, file_path: str, output_format: str) -> str:
        """
        Process a single file.
        
        Args:
            file_path: Path to input file
            output_format: Output format
            
        Returns:
            Path to processed file
        """
        file_path = Path(file_path)
        
        # Determine file type and processing method
        if file_path.suffix.lower() in ['.tif', '.tiff']:
            return self._process_geotiff(file_path, output_format)
        elif file_path.suffix.lower() in ['.jpg', '.jpeg', '.png']:
            return self._process_image(file_path, output_format)
        else:
            logger.warning(f"Unknown file type: {file_path.suffix}")
            return str(file_path)
    
    def _process_geotiff(self, file_path: Path, output_format: str) -> str:
        """Process GeoTIFF file."""
        # For now, just copy to processed directory
        # In a real implementation, this would include:
        # - Reprojection
        # - Resampling
        # - Band selection
        # - Format conversion
        
        output_path = self.processed_dir / f"{file_path.stem}_{output_format}.tif"
        
        # Copy file
        import shutil
        shutil.copy2(file_path, output_path)
        
        return str(output_path)
    
    def _process_image(self, file_path: Path, output_format: str) -> str:
        """Process image file."""
        # For now, just copy to processed directory
        # In a real implementation, this would include:
        # - Resizing
        # - Format conversion
        # - Quality optimization
        
        output_path = self.processed_dir / f"{file_path.stem}_{output_format}{file_path.suffix}"
        
        # Copy file
        import shutil
        shutil.copy2(file_path, output_path)
        
        return str(output_path)
    
    def get_data_summary(self, data_dir: Optional[Path] = None) -> Dict:
        """
        Get summary of available data.
        
        Args:
            data_dir: Directory to summarize (default: data directory)
            
        Returns:
            Summary dictionary
        """
        if data_dir is None:
            data_dir = self.data_dir
        
        summary = {
            'total_files': 0,
            'total_size_mb': 0,
            'sources': {},
            'file_types': {},
            'date_range': {'earliest': None, 'latest': None}
        }
        
        # Walk through data directory
        for root, dirs, files in os.walk(data_dir):
            for file in files:
                file_path = Path(root) / file
                file_info = get_file_info(file_path)
                
                summary['total_files'] += 1
                summary['total_size_mb'] += file_info.get('size_mb', 0)
                
                # Count by file type
                file_type = file_info.get('suffix', '').lower()
                summary['file_types'][file_type] = summary['file_types'].get(file_type, 0) + 1
                
                # Track date range
                modified_date = file_info.get('modified', '')
                if modified_date:
                    if summary['date_range']['earliest'] is None or modified_date < summary['date_range']['earliest']:
                        summary['date_range']['earliest'] = modified_date
                    if summary['date_range']['latest'] is None or modified_date > summary['date_range']['latest']:
                        summary['date_range']['latest'] = modified_date
        
        # Convert size to GB
        summary['total_size_gb'] = round(summary['total_size_mb'] / 1024, 2)
        
        return summary


class NASADataSource:
    """NASA EarthData data source handler."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.base_url = config['data_sources']['nasa_earthdata']['base_url']
        self.username = config['data_sources']['nasa_earthdata']['username']
        self.password = config['data_sources']['nasa_earthdata']['password']
    
    async def download_data(self, bbox: Tuple[float, float, float, float], 
                          start_date: str, end_date: str) -> List[str]:
        """Download data from NASA EarthData."""
        # Placeholder implementation
        # In a real implementation, this would:
        # 1. Query NASA EarthData API
        # 2. Download relevant datasets
        # 3. Return list of downloaded files
        
        logger.info(f"NASA EarthData download requested for bbox {bbox}, dates {start_date} to {end_date}")
        return []


class Sentinel2DataSource:
    """Sentinel-2 data source handler."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.base_url = config['data_sources']['sentinel2']['base_url']
        self.username = config['data_sources']['sentinel2']['username']
        self.password = config['data_sources']['sentinel2']['password']
    
    async def download_data(self, bbox: Tuple[float, float, float, float], 
                          start_date: str, end_date: str) -> List[str]:
        """Download data from Sentinel-2."""
        # Placeholder implementation
        # In a real implementation, this would:
        # 1. Query Sentinel-2 API
        # 2. Download relevant scenes
        # 3. Return list of downloaded files
        
        logger.info(f"Sentinel-2 download requested for bbox {bbox}, dates {start_date} to {end_date}")
        return []


class OpenStreetMapSource:
    """OpenStreetMap data source handler."""
    
    def __init__(self, config: Dict):
        self.config = config
        self.base_url = config['data_sources']['openstreetmap']['base_url']
        self.user_agent = config['data_sources']['openstreetmap']['user_agent']
    
    async def download_data(self, bbox: Tuple[float, float, float, float], 
                          start_date: str, end_date: str) -> List[str]:
        """Download reference data from OpenStreetMap."""
        # Placeholder implementation
        # In a real implementation, this would:
        # 1. Query OpenStreetMap Overpass API
        # 2. Download relevant features
        # 3. Return list of downloaded files
        
        logger.info(f"OpenStreetMap download requested for bbox {bbox}")
        return []
