"""
Image Preprocessing Module for CrisisMapper.

This module handles image preprocessing tasks including tiling, augmentation,
and format conversion for optimal disaster detection performance.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import List, Tuple, Optional, Dict, Union
import albumentations as A
from pathlib import Path
import logging

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class ImagePreprocessor:
    """
    Image preprocessing class for disaster detection.
    
    Handles tiling, augmentation, and format conversion for optimal
    YOLOv8 performance on satellite and drone imagery.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the ImagePreprocessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.tile_size = config['data_processing']['tile_size']
        self.overlap = config['data_processing']['overlap']
        self.max_image_size = config['data_processing']['max_image_size']
        
        # Setup augmentation pipeline
        self.augmentation_pipeline = self._setup_augmentation()
        
        logger.info(f"ImagePreprocessor initialized with tile_size={self.tile_size}")
    
    def _setup_augmentation(self) -> A.Compose:
        """Setup data augmentation pipeline."""
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2,
                p=0.5
            ),
            A.HueSaturationValue(
                hue_shift_limit=20,
                sat_shift_limit=30,
                val_shift_limit=20,
                p=0.5
            ),
            A.GaussNoise(var_limit=(10.0, 50.0), p=0.3),
            A.Blur(blur_limit=3, p=0.3),
        ])
    
    def preprocess_image(self, 
                        image: Union[str, np.ndarray, Image.Image],
                        apply_augmentation: bool = False) -> np.ndarray:
        """
        Preprocess a single image for detection.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            apply_augmentation: Whether to apply data augmentation
            
        Returns:
            Preprocessed image as numpy array
        """
        # Load image
        if isinstance(image, str):
            image = cv2.imread(image)
            if image is None:
                raise ValueError(f"Could not load image from {image}")
        elif isinstance(image, Image.Image):
            image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Resize if too large
        image = self._resize_if_needed(image)
        
        # Apply augmentation if requested
        if apply_augmentation:
            image = self._apply_augmentation(image)
        
        # Normalize
        image = self._normalize_image(image)
        
        return image
    
    def _resize_if_needed(self, image: np.ndarray) -> np.ndarray:
        """Resize image if it exceeds maximum size."""
        height, width = image.shape[:2]
        
        if height > self.max_image_size or width > self.max_image_size:
            # Calculate scale factor
            scale = min(self.max_image_size / height, self.max_image_size / width)
            new_height = int(height * scale)
            new_width = int(width * scale)
            
            image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)
            logger.info(f"Resized image from {height}x{width} to {new_height}x{new_width}")
        
        return image
    
    def _apply_augmentation(self, image: np.ndarray) -> np.ndarray:
        """Apply data augmentation to image."""
        try:
            augmented = self.augmentation_pipeline(image=image)
            return augmented['image']
        except Exception as e:
            logger.warning(f"Augmentation failed: {e}. Returning original image.")
            return image
    
    def _normalize_image(self, image: np.ndarray) -> np.ndarray:
        """Normalize image to [0, 1] range."""
        return image.astype(np.float32) / 255.0
    
    def tile_image(self, 
                   image: Union[str, np.ndarray, Image.Image],
                   save_tiles: bool = False,
                   output_dir: Optional[str] = None) -> List[Dict]:
        """
        Tile a large image into smaller patches for processing.
        
        Args:
            image: Input image (path, numpy array, or PIL Image)
            save_tiles: Whether to save tiles to disk
            output_dir: Directory to save tiles
            
        Returns:
            List of tile information dictionaries
        """
        # Load and preprocess image
        image = self.preprocess_image(image)
        height, width = image.shape[:2]
        
        # Calculate tile parameters
        tile_height = self.tile_size
        tile_width = self.tile_size
        overlap_pixels = int(self.tile_size * self.overlap)
        
        # Calculate step sizes
        step_y = tile_height - overlap_pixels
        step_x = tile_width - overlap_pixels
        
        tiles = []
        tile_count = 0
        
        # Generate tiles
        for y in range(0, height - overlap_pixels, step_y):
            for x in range(0, width - overlap_pixels, step_x):
                # Ensure tile doesn't exceed image boundaries
                y_end = min(y + tile_height, height)
                x_end = min(x + tile_width, width)
                
                # Extract tile
                tile = image[y:y_end, x:x_end]
                
                # Skip if tile is too small
                if tile.shape[0] < tile_height * 0.5 or tile.shape[1] < tile_width * 0.5:
                    continue
                
                # Pad tile if necessary
                if tile.shape[0] != tile_height or tile.shape[1] != tile_width:
                    tile = self._pad_tile(tile, tile_height, tile_width)
                
                # Create tile info
                tile_info = {
                    'tile_id': tile_count,
                    'coordinates': {
                        'x': x,
                        'y': y,
                        'x_end': x_end,
                        'y_end': y_end
                    },
                    'relative_coordinates': {
                        'x': x / width,
                        'y': y / height,
                        'x_end': x_end / width,
                        'y_end': y_end / height
                    },
                    'tile': tile,
                    'original_size': (height, width)
                }
                
                tiles.append(tile_info)
                tile_count += 1
                
                # Save tile if requested
                if save_tiles and output_dir:
                    self._save_tile(tile, tile_info, output_dir)
        
        logger.info(f"Generated {len(tiles)} tiles from image {height}x{width}")
        return tiles
    
    def _pad_tile(self, tile: np.ndarray, target_height: int, target_width: int) -> np.ndarray:
        """Pad tile to target dimensions."""
        height, width = tile.shape[:2]
        
        if height < target_height or width < target_width:
            # Calculate padding
            pad_height = max(0, target_height - height)
            pad_width = max(0, target_width - width)
            
            # Pad with zeros (black)
            padded = np.pad(tile, 
                           ((0, pad_height), (0, pad_width), (0, 0)),
                           mode='constant',
                           constant_values=0)
            
            return padded
        
        return tile
    
    def _save_tile(self, tile: np.ndarray, tile_info: Dict, output_dir: str):
        """Save tile to disk."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Convert back to uint8 for saving
        tile_uint8 = (tile * 255).astype(np.uint8)
        
        # Save tile
        filename = f"tile_{tile_info['tile_id']:04d}.jpg"
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, tile_uint8)
        
        # Save tile metadata
        metadata_file = os.path.join(output_dir, f"tile_{tile_info['tile_id']:04d}.json")
        import json
        with open(metadata_file, 'w') as f:
            json.dump({
                'tile_id': tile_info['tile_id'],
                'coordinates': tile_info['coordinates'],
                'relative_coordinates': tile_info['relative_coordinates'],
                'original_size': tile_info['original_size']
            }, f, indent=2)
    
    def reconstruct_from_tiles(self, 
                              tiles: List[Dict], 
                              original_shape: Tuple[int, int, int]) -> np.ndarray:
        """
        Reconstruct original image from tiles.
        
        Args:
            tiles: List of tile information dictionaries
            original_shape: Original image shape (height, width, channels)
            
        Returns:
            Reconstructed image
        """
        height, width, channels = original_shape
        reconstructed = np.zeros((height, width, channels), dtype=np.float32)
        count_map = np.zeros((height, width), dtype=np.float32)
        
        for tile_info in tiles:
            coords = tile_info['coordinates']
            tile = tile_info['tile']
            
            y_start, y_end = coords['y'], coords['y_end']
            x_start, x_end = coords['x'], coords['x_end']
            
            # Add tile to reconstructed image
            reconstructed[y_start:y_end, x_start:x_end] += tile
            count_map[y_start:y_end, x_start:x_end] += 1
        
        # Average overlapping regions
        count_map[count_map == 0] = 1  # Avoid division by zero
        reconstructed = reconstructed / count_map[:, :, np.newaxis]
        
        return reconstructed
    
    def batch_preprocess(self, 
                        image_paths: List[str],
                        output_dir: Optional[str] = None,
                        tile_images: bool = False) -> List[Dict]:
        """
        Preprocess a batch of images.
        
        Args:
            image_paths: List of image file paths
            output_dir: Directory to save processed images
            tile_images: Whether to tile large images
            
        Returns:
            List of preprocessing results
        """
        results = []
        
        for i, image_path in enumerate(image_paths):
            try:
                logger.info(f"Preprocessing image {i+1}/{len(image_paths)}: {image_path}")
                
                # Preprocess image
                processed_image = self.preprocess_image(image_path)
                
                result = {
                    'image_path': image_path,
                    'processed_image': processed_image,
                    'original_shape': processed_image.shape,
                    'tiles': []
                }
                
                # Tile if requested
                if tile_images:
                    tiles = self.tile_image(processed_image, save_tiles=True, output_dir=output_dir)
                    result['tiles'] = tiles
                
                results.append(result)
                
            except Exception as e:
                logger.error(f"Error preprocessing {image_path}: {e}")
                results.append({
                    'image_path': image_path,
                    'error': str(e)
                })
        
        return results
    
    def get_preprocessing_stats(self, results: List[Dict]) -> Dict:
        """Get preprocessing statistics."""
        successful = [r for r in results if 'error' not in r]
        failed = [r for r in results if 'error' in r]
        
        total_tiles = sum(len(r.get('tiles', [])) for r in successful)
        
        return {
            'total_images': len(results),
            'successful': len(successful),
            'failed': len(failed),
            'total_tiles': total_tiles,
            'average_tiles_per_image': total_tiles / len(successful) if successful else 0
        }
