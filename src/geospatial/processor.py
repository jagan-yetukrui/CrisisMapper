"""
Geospatial Processing Module for CrisisMapper.

This module handles geospatial operations including coordinate transformation,
projection, and georeferencing of disaster detection results.
"""

import os
import numpy as np
import geopandas as gpd
from shapely.geometry import Point, Polygon, box
from shapely.ops import unary_union
import rasterio
from rasterio.transform import from_bounds
from rasterio.crs import CRS
import pyproj
from typing import List, Dict, Tuple, Optional, Union
import logging

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class GeospatialProcessor:
    """
    Main geospatial processing class for disaster detection results.
    
    Handles coordinate transformation, georeferencing, and spatial analysis
    of disaster detection results from satellite and drone imagery.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the GeospatialProcessor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.default_crs = CRS.from_string(config['geospatial']['default_crs'])
        self.output_crs = CRS.from_string(config['geospatial']['output_crs'])
        self.buffer_distance = config['geospatial']['buffer_distance']
        self.simplify_tolerance = config['geospatial']['simplify_tolerance']
        
        # Initialize coordinate transformer
        self.transformer = pyproj.Transformer.from_crs(
            self.default_crs, self.output_crs, always_xy=True
        )
        
        logger.info(f"GeospatialProcessor initialized with CRS: {self.default_crs} -> {self.output_crs}")
    
    def process_detection_results(self, 
                                 detection_results: List[Dict],
                                 image_metadata: Optional[Dict] = None) -> gpd.GeoDataFrame:
        """
        Process detection results into geospatial format.
        
        Args:
            detection_results: List of detection results from DisasterDetector
            image_metadata: Optional metadata about the source image
            
        Returns:
            GeoDataFrame with disaster polygons
        """
        geometries = []
        properties = []
        
        for result in detection_results:
            if 'error' in result:
                continue
            
            detections = result.get('detections', {})
            image_shape = result.get('image_shape', (0, 0, 0))
            
            for i, (box, conf, class_id, class_name) in enumerate(zip(
                detections.get('boxes', []),
                detections.get('confidences', []),
                detections.get('class_ids', []),
                detections.get('class_names', [])
            )):
                # Convert relative coordinates to absolute
                height, width = image_shape[:2]
                x1 = box[0] * width
                y1 = box[1] * height
                x2 = box[2] * width
                y2 = box[3] * height
                
                # Create polygon from bounding box
                polygon = self._create_polygon_from_box(x1, y1, x2, y2, image_metadata)
                
                if polygon is not None:
                    geometries.append(polygon)
                    properties.append({
                        'detection_id': i,
                        'class_id': class_id,
                        'class_name': class_name,
                        'confidence': conf,
                        'area_m2': self._calculate_area_m2(polygon),
                        'image_shape': image_shape,
                        'bbox': [x1, y1, x2, y2]
                    })
        
        # Create GeoDataFrame
        gdf = gpd.GeoDataFrame(properties, geometry=geometries, crs=self.default_crs)
        
        # Transform to output CRS
        if gdf.crs != self.output_crs:
            gdf = gdf.to_crs(self.output_crs)
        
        # Add buffer and simplify
        gdf = self._add_buffer_and_simplify(gdf)
        
        logger.info(f"Processed {len(gdf)} disaster polygons")
        return gdf
    
    def _create_polygon_from_box(self, 
                                x1: float, y1: float, x2: float, y2: float,
                                image_metadata: Optional[Dict] = None) -> Optional[Polygon]:
        """
        Create a polygon from bounding box coordinates.
        
        Args:
            x1, y1, x2, y2: Bounding box coordinates
            image_metadata: Optional image metadata with georeferencing info
            
        Returns:
            Shapely Polygon or None
        """
        # For now, create a simple polygon in image coordinates
        # In a real implementation, this would use georeferencing information
        # from the image metadata to convert to geographic coordinates
        
        if image_metadata and 'geotransform' in image_metadata:
            # Use geotransform to convert pixel coordinates to geographic
            gt = image_metadata['geotransform']
            lon1 = gt[0] + x1 * gt[1] + y1 * gt[2]
            lat1 = gt[3] + x1 * gt[4] + y1 * gt[5]
            lon2 = gt[0] + x2 * gt[1] + y2 * gt[2]
            lat2 = gt[3] + x2 * gt[4] + y2 * gt[5]
            
            # Create polygon from geographic coordinates
            polygon = box(min(lon1, lon2), min(lat1, lat2), 
                         max(lon1, lon2), max(lat1, lat2))
        else:
            # Default: create polygon in image coordinates (0-1 range)
            # This is a placeholder - in practice, you'd need proper georeferencing
            polygon = box(x1, y1, x2, y2)
        
        return polygon
    
    def _calculate_area_m2(self, polygon: Polygon) -> float:
        """
        Calculate polygon area in square meters.
        
        Args:
            polygon: Shapely polygon
            
        Returns:
            Area in square meters
        """
        # Transform to a projected CRS for accurate area calculation
        if polygon.crs is None or polygon.crs.is_geographic:
            # Use UTM projection for area calculation
            utm_crs = self._get_utm_crs(polygon.centroid)
            polygon_utm = polygon.to_crs(utm_crs)
            return polygon_utm.area
        else:
            return polygon.area
    
    def _get_utm_crs(self, point: Point) -> CRS:
        """
        Get appropriate UTM CRS for a point.
        
        Args:
            point: Shapely point
            
        Returns:
            UTM CRS
        """
        lon, lat = point.x, point.y
        utm_zone = int((lon + 180) / 6) + 1
        utm_hemisphere = 'north' if lat >= 0 else 'south'
        
        utm_crs = CRS.from_string(f"+proj=utm +zone={utm_zone} +{utm_hemisphere} +datum=WGS84")
        return utm_crs
    
    def _add_buffer_and_simplify(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Add buffer and simplify geometries.
        
        Args:
            gdf: Input GeoDataFrame
            
        Returns:
            Processed GeoDataFrame
        """
        # Add buffer
        if self.buffer_distance > 0:
            gdf['geometry'] = gdf.geometry.buffer(self.buffer_distance)
        
        # Simplify geometries
        if self.simplify_tolerance > 0:
            gdf['geometry'] = gdf.geometry.simplify(self.simplify_tolerance)
        
        return gdf
    
    def merge_disaster_areas(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Merge overlapping disaster areas by class.
        
        Args:
            gdf: Input GeoDataFrame with disaster polygons
            
        Returns:
            Merged GeoDataFrame
        """
        merged_areas = []
        
        # Group by disaster class
        for class_name in gdf['class_name'].unique():
            class_gdf = gdf[gdf['class_name'] == class_name].copy()
            
            if len(class_gdf) == 0:
                continue
            
            # Merge overlapping polygons
            merged_geom = unary_union(class_gdf.geometry.tolist())
            
            # Create new GeoDataFrame for merged areas
            if merged_geom.geom_type == 'Polygon':
                merged_geom = [merged_geom]
            elif merged_geom.geom_type == 'MultiPolygon':
                merged_geom = list(merged_geom.geoms)
            
            for geom in merged_geom:
                merged_areas.append({
                    'class_name': class_name,
                    'geometry': geom,
                    'area_m2': self._calculate_area_m2(geom),
                    'merged_count': len(class_gdf)
                })
        
        if merged_areas:
            merged_gdf = gpd.GeoDataFrame(merged_areas, crs=gdf.crs)
            logger.info(f"Merged {len(gdf)} polygons into {len(merged_gdf)} areas")
            return merged_gdf
        else:
            return gdf
    
    def create_disaster_summary(self, gdf: gpd.GeoDataFrame) -> Dict:
        """
        Create summary statistics for disaster areas.
        
        Args:
            gdf: GeoDataFrame with disaster polygons
            
        Returns:
            Summary statistics dictionary
        """
        if len(gdf) == 0:
            return {
                'total_areas': 0,
                'total_area_m2': 0,
                'class_summary': {},
                'severity_distribution': {}
            }
        
        # Calculate total area
        total_area = gdf['area_m2'].sum()
        
        # Summary by class
        class_summary = {}
        for class_name in gdf['class_name'].unique():
            class_gdf = gdf[gdf['class_name'] == class_name]
            class_summary[class_name] = {
                'count': len(class_gdf),
                'total_area_m2': class_gdf['area_m2'].sum(),
                'average_area_m2': class_gdf['area_m2'].mean(),
                'max_area_m2': class_gdf['area_m2'].max(),
                'min_area_m2': class_gdf['area_m2'].min()
            }
        
        # Severity distribution (if available)
        severity_distribution = {}
        if 'severity' in gdf.columns:
            for severity in gdf['severity'].unique():
                severity_gdf = gdf[gdf['severity'] == severity]
                severity_distribution[severity] = {
                    'count': len(severity_gdf),
                    'total_area_m2': severity_gdf['area_m2'].sum()
                }
        
        return {
            'total_areas': len(gdf),
            'total_area_m2': total_area,
            'total_area_km2': total_area / 1_000_000,
            'class_summary': class_summary,
            'severity_distribution': severity_distribution,
            'bounds': gdf.total_bounds.tolist()
        }
    
    def filter_by_area(self, gdf: gpd.GeoDataFrame, 
                      min_area_m2: float = 0,
                      max_area_m2: Optional[float] = None) -> gpd.GeoDataFrame:
        """
        Filter disaster areas by area size.
        
        Args:
            gdf: Input GeoDataFrame
            min_area_m2: Minimum area in square meters
            max_area_m2: Maximum area in square meters
            
        Returns:
            Filtered GeoDataFrame
        """
        mask = gdf['area_m2'] >= min_area_m2
        
        if max_area_m2 is not None:
            mask = mask & (gdf['area_m2'] <= max_area_m2)
        
        filtered_gdf = gdf[mask].copy()
        logger.info(f"Filtered from {len(gdf)} to {len(filtered_gdf)} areas")
        
        return filtered_gdf
    
    def filter_by_confidence(self, gdf: gpd.GeoDataFrame, 
                           min_confidence: float = 0.0) -> gpd.GeoDataFrame:
        """
        Filter disaster areas by confidence score.
        
        Args:
            gdf: Input GeoDataFrame
            min_confidence: Minimum confidence threshold
            
        Returns:
            Filtered GeoDataFrame
        """
        if 'confidence' not in gdf.columns:
            logger.warning("No confidence column found, returning original GeoDataFrame")
            return gdf
        
        filtered_gdf = gdf[gdf['confidence'] >= min_confidence].copy()
        logger.info(f"Filtered from {len(gdf)} to {len(filtered_gdf)} areas by confidence")
        
        return filtered_gdf
