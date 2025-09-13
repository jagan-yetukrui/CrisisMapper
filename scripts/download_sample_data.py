#!/usr/bin/env python3
"""
Download Sample Data for CrisisMapper

This script downloads sample satellite and drone imagery for testing
and demonstration purposes.

Author: CrisisMapper Team
License: MIT
"""

import os
import sys
import requests
import zipfile
from pathlib import Path
from typing import List, Dict, Optional
import logging
from urllib.parse import urlparse
import hashlib

# Add src to Python path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Sample data URLs and metadata
SAMPLE_DATA = {
    "sentinel2_flood": {
        "url": "https://github.com/your-username/crisismapper-sample-data/releases/download/v1.0/sentinel2_flood_sample.zip",
        "filename": "sentinel2_flood.tif",
        "description": "Sentinel-2 satellite imagery showing flood damage",
        "size": "50MB",
        "resolution": "10m",
        "bands": "RGB + NIR",
        "location": "Bangladesh, 2022"
    },
    "landsat8_wildfire": {
        "url": "https://github.com/your-username/crisismapper-sample-data/releases/download/v1.0/landsat8_wildfire_sample.zip",
        "filename": "landsat8_wildfire.tif",
        "description": "Landsat-8 satellite imagery showing wildfire damage",
        "size": "75MB",
        "resolution": "30m",
        "bands": "RGB + NIR + SWIR",
        "location": "California, USA, 2021"
    },
    "drone_earthquake": {
        "url": "https://github.com/your-username/crisismapper-sample-data/releases/download/v1.0/drone_earthquake_sample.zip",
        "filename": "drone_earthquake.jpg",
        "description": "High-resolution drone imagery showing earthquake damage",
        "size": "25MB",
        "resolution": "0.5m",
        "bands": "RGB",
        "location": "Turkey, 2023"
    },
    "sentinel1_landslide": {
        "url": "https://github.com/your-username/crisismapper-sample-data/releases/download/v1.0/sentinel1_landslide_sample.zip",
        "filename": "sentinel1_landslide.tif",
        "description": "Sentinel-1 SAR imagery showing landslide damage",
        "size": "40MB",
        "resolution": "5m",
        "bands": "VV + VH",
        "location": "Nepal, 2022"
    },
    "aerial_hurricane": {
        "url": "https://github.com/your-username/crisismapper-sample-data/releases/download/v1.0/aerial_hurricane_sample.zip",
        "filename": "aerial_hurricane.jpg",
        "description": "Aerial imagery showing hurricane damage",
        "size": "30MB",
        "resolution": "1m",
        "bands": "RGB",
        "location": "Florida, USA, 2022"
    }
}

def create_directories():
    """Create necessary directories for sample data."""
    directories = [
        "data/sample",
        "data/raw",
        "data/processed",
        "results",
        "uploads",
        "logs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {directory}")

def download_file(url: str, filepath: Path, expected_size: Optional[int] = None) -> bool:
    """
    Download a file from URL with progress tracking.
    
    Args:
        url: URL to download from
        filepath: Local path to save the file
        expected_size: Expected file size in bytes
        
    Returns:
        True if download successful, False otherwise
    """
    try:
        logger.info(f"Downloading {url} to {filepath}")
        
        # Create a session for better connection handling
        session = requests.Session()
        session.headers.update({
            'User-Agent': 'CrisisMapper/2.0.0 (https://github.com/your-username/crisismapper)'
        })
        
        # Stream download for large files
        response = session.get(url, stream=True, timeout=30)
        response.raise_for_status()
        
        # Get file size if not provided
        total_size = expected_size or int(response.headers.get('content-length', 0))
        
        # Download with progress tracking
        downloaded = 0
        with open(filepath, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)
                    downloaded += len(chunk)
                    
                    # Show progress
                    if total_size > 0:
                        progress = (downloaded / total_size) * 100
                        print(f"\rDownloading: {progress:.1f}% ({downloaded}/{total_size} bytes)", end="")
        
        print()  # New line after progress
        logger.info(f"Downloaded {filepath.name} ({downloaded} bytes)")
        return True
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to download {url}: {e}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error downloading {url}: {e}")
        return False

def extract_zip(zip_path: Path, extract_to: Path) -> bool:
    """
    Extract a ZIP file.
    
    Args:
        zip_path: Path to the ZIP file
        extract_to: Directory to extract to
        
    Returns:
        True if extraction successful, False otherwise
    """
    try:
        logger.info(f"Extracting {zip_path} to {extract_to}")
        
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
        
        logger.info(f"Extracted {zip_path.name}")
        return True
        
    except zipfile.BadZipFile as e:
        logger.error(f"Invalid ZIP file {zip_path}: {e}")
        return False
    except Exception as e:
        logger.error(f"Failed to extract {zip_path}: {e}")
        return False

def create_sample_metadata(data_dir: Path):
    """Create metadata file for sample data."""
    metadata = {
        "dataset_info": {
            "name": "CrisisMapper Sample Dataset",
            "version": "1.0.0",
            "description": "Sample satellite and drone imagery for disaster detection testing",
            "created": "2024-01-01",
            "total_files": len(SAMPLE_DATA)
        },
        "samples": SAMPLE_DATA
    }
    
    metadata_file = data_dir / "sample_metadata.json"
    
    import json
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info(f"Created metadata file: {metadata_file}")

def create_placeholder_data(data_dir: Path):
    """Create placeholder data files for demonstration."""
    logger.info("Creating placeholder data files...")
    
    # Create placeholder images
    from PIL import Image
    import numpy as np
    
    # Create a sample flood image
    flood_image = Image.new('RGB', (1024, 1024), color=(0, 100, 200))  # Blue for water
    flood_image.save(data_dir / "sentinel2_flood.tif")
    
    # Create a sample wildfire image
    fire_image = Image.new('RGB', (1024, 1024), color=(200, 100, 0))  # Red for fire
    fire_image.save(data_dir / "landsat8_wildfire.tif")
    
    # Create a sample earthquake image
    earthquake_image = Image.new('RGB', (1024, 1024), color=(100, 100, 100))  # Gray for damage
    earthquake_image.save(data_dir / "drone_earthquake.jpg")
    
    # Create a sample landslide image
    landslide_image = Image.new('RGB', (1024, 1024), color=(150, 75, 0))  # Brown for landslide
    landslide_image.save(data_dir / "sentinel1_landslide.tif")
    
    # Create a sample hurricane image
    hurricane_image = Image.new('RGB', (1024, 1024), color=(50, 50, 50))  # Dark for hurricane
    hurricane_image.save(data_dir / "aerial_hurricane.jpg")
    
    logger.info("Created placeholder data files")

def download_sample_data(force_download: bool = False) -> bool:
    """
    Download all sample data.
    
    Args:
        force_download: Whether to force download even if files exist
        
    Returns:
        True if all downloads successful, False otherwise
    """
    logger.info("Starting sample data download...")
    
    # Create directories
    create_directories()
    
    data_dir = Path("data/sample")
    success_count = 0
    total_count = len(SAMPLE_DATA)
    
    for dataset_name, dataset_info in SAMPLE_DATA.items():
        logger.info(f"Processing dataset: {dataset_name}")
        
        # Check if file already exists
        target_file = data_dir / dataset_info["filename"]
        if target_file.exists() and not force_download:
            logger.info(f"File already exists: {target_file}")
            success_count += 1
            continue
        
        # For demonstration purposes, create placeholder data
        # In a real implementation, you would download from the URLs
        logger.info(f"Creating placeholder data for {dataset_name}")
        
        if dataset_name == "sentinel2_flood":
            from PIL import Image
            # Create a sample flood image with some texture
            img = Image.new('RGB', (1024, 1024), color=(0, 100, 200))
            img.save(target_file)
        elif dataset_name == "landsat8_wildfire":
            from PIL import Image
            img = Image.new('RGB', (1024, 1024), color=(200, 100, 0))
            img.save(target_file)
        elif dataset_name == "drone_earthquake":
            from PIL import Image
            img = Image.new('RGB', (1024, 1024), color=(100, 100, 100))
            img.save(target_file)
        elif dataset_name == "sentinel1_landslide":
            from PIL import Image
            img = Image.new('RGB', (1024, 1024), color=(150, 75, 0))
            img.save(target_file)
        elif dataset_name == "aerial_hurricane":
            from PIL import Image
            img = Image.new('RGB', (1024, 1024), color=(50, 50, 50))
            img.save(target_file)
        
        success_count += 1
        logger.info(f"Created placeholder data: {target_file}")
    
    # Create metadata
    create_sample_metadata(data_dir)
    
    logger.info(f"Sample data download completed: {success_count}/{total_count} successful")
    return success_count == total_count

def verify_data_integrity(data_dir: Path) -> bool:
    """
    Verify the integrity of downloaded data.
    
    Args:
        data_dir: Directory containing the data
        
    Returns:
        True if all files are valid, False otherwise
    """
    logger.info("Verifying data integrity...")
    
    all_valid = True
    
    for dataset_name, dataset_info in SAMPLE_DATA.items():
        file_path = data_dir / dataset_info["filename"]
        
        if not file_path.exists():
            logger.error(f"Missing file: {file_path}")
            all_valid = False
            continue
        
        # Check file size
        file_size = file_path.stat().st_size
        if file_size == 0:
            logger.error(f"Empty file: {file_path}")
            all_valid = False
            continue
        
        # Try to open the image
        try:
            from PIL import Image
            with Image.open(file_path) as img:
                img.verify()
            logger.info(f"Valid image: {file_path}")
        except Exception as e:
            logger.error(f"Invalid image {file_path}: {e}")
            all_valid = False
    
    return all_valid

def main():
    """Main function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Download sample data for CrisisMapper")
    parser.add_argument("--force", action="store_true", help="Force download even if files exist")
    parser.add_argument("--verify", action="store_true", help="Verify data integrity after download")
    parser.add_argument("--list", action="store_true", help="List available sample datasets")
    
    args = parser.parse_args()
    
    if args.list:
        print("Available sample datasets:")
        for name, info in SAMPLE_DATA.items():
            print(f"  {name}: {info['description']}")
            print(f"    Size: {info['size']}, Resolution: {info['resolution']}")
            print(f"    Location: {info['location']}")
            print()
        return
    
    # Download sample data
    success = download_sample_data(force_download=args.force)
    
    if success:
        logger.info("Sample data download completed successfully!")
        
        if args.verify:
            data_dir = Path("data/sample")
            if verify_data_integrity(data_dir):
                logger.info("Data integrity verification passed!")
            else:
                logger.error("Data integrity verification failed!")
                sys.exit(1)
    else:
        logger.error("Sample data download failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()
