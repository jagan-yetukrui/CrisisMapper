"""
Geospatial Export Module for CrisisMapper.

This module handles exporting disaster detection results to various
geospatial formats including GeoJSON, Shapefile, and KML.
"""

import os
import json
from pathlib import Path
from typing import Dict, List, Optional, Union
import geopandas as gpd
from shapely.geometry import mapping
import logging

from ..utils.logger import setup_logger

logger = setup_logger(__name__)


class GeospatialExporter:
    """
    Export disaster detection results to various geospatial formats.
    
    Supports GeoJSON, Shapefile, KML, and other common geospatial formats
    for integration with GIS tools and mapping applications.
    """
    
    def __init__(self, config: Dict):
        """
        Initialize the GeospatialExporter.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.output_dir = Path(config['output']['base_dir'])
        self.export_formats = config['output']['export_formats']
        self.include_metadata = config['output']['include_metadata']
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"GeospatialExporter initialized with output dir: {self.output_dir}")
    
    def export_results(self, 
                      gdf: gpd.GeoDataFrame,
                      filename_prefix: str = "disaster_detection",
                      formats: Optional[List[str]] = None) -> Dict[str, str]:
        """
        Export GeoDataFrame to multiple formats.
        
        Args:
            gdf: GeoDataFrame to export
            filename_prefix: Prefix for output files
            formats: List of formats to export (defaults to config)
            
        Returns:
            Dictionary mapping format names to file paths
        """
        if formats is None:
            formats = self.export_formats
        
        exported_files = {}
        
        for format_name in formats:
            try:
                file_path = self._export_single_format(gdf, filename_prefix, format_name)
                exported_files[format_name] = str(file_path)
                logger.info(f"Exported {format_name} to {file_path}")
            except Exception as e:
                logger.error(f"Failed to export {format_name}: {e}")
                exported_files[format_name] = f"Error: {e}"
        
        return exported_files
    
    def _export_single_format(self, 
                             gdf: gpd.GeoDataFrame,
                             filename_prefix: str,
                             format_name: str) -> Path:
        """
        Export GeoDataFrame to a single format.
        
        Args:
            gdf: GeoDataFrame to export
            filename_prefix: Prefix for output files
            format_name: Format to export to
            
        Returns:
            Path to exported file
        """
        if format_name.lower() == 'geojson':
            return self._export_geojson(gdf, filename_prefix)
        elif format_name.lower() == 'shapefile':
            return self._export_shapefile(gdf, filename_prefix)
        elif format_name.lower() == 'kml':
            return self._export_kml(gdf, filename_prefix)
        elif format_name.lower() == 'csv':
            return self._export_csv(gdf, filename_prefix)
        else:
            raise ValueError(f"Unsupported format: {format_name}")
    
    def _export_geojson(self, gdf: gpd.GeoDataFrame, filename_prefix: str) -> Path:
        """Export to GeoJSON format."""
        filename = f"{filename_prefix}.geojson"
        file_path = self.output_dir / filename
        
        # Prepare GeoDataFrame for export
        export_gdf = self._prepare_for_export(gdf)
        
        # Export to GeoJSON
        export_gdf.to_file(file_path, driver='GeoJSON')
        
        return file_path
    
    def _export_shapefile(self, gdf: gpd.GeoDataFrame, filename_prefix: str) -> Path:
        """Export to Shapefile format."""
        filename = f"{filename_prefix}.shp"
        file_path = self.output_dir / filename
        
        # Prepare GeoDataFrame for export
        export_gdf = self._prepare_for_export(gdf)
        
        # Export to Shapefile
        export_gdf.to_file(file_path, driver='ESRI Shapefile')
        
        return file_path
    
    def _export_kml(self, gdf: gpd.GeoDataFrame, filename_prefix: str) -> Path:
        """Export to KML format."""
        filename = f"{filename_prefix}.kml"
        file_path = self.output_dir / filename
        
        # Prepare GeoDataFrame for export
        export_gdf = self._prepare_for_export(gdf)
        
        # Export to KML
        export_gdf.to_file(file_path, driver='KML')
        
        return file_path
    
    def _export_csv(self, gdf: gpd.GeoDataFrame, filename_prefix: str) -> Path:
        """Export to CSV format (geometry as WKT)."""
        filename = f"{filename_prefix}.csv"
        file_path = self.output_dir / filename
        
        # Prepare GeoDataFrame for export
        export_gdf = self._prepare_for_export(gdf)
        
        # Convert geometry to WKT
        export_gdf['geometry_wkt'] = export_gdf.geometry.apply(lambda x: x.wkt)
        
        # Drop geometry column and export
        export_gdf.drop(columns=['geometry']).to_csv(file_path, index=False)
        
        return file_path
    
    def _prepare_for_export(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """
        Prepare GeoDataFrame for export by cleaning and formatting data.
        
        Args:
            gdf: Input GeoDataFrame
            
        Returns:
            Prepared GeoDataFrame
        """
        export_gdf = gdf.copy()
        
        # Ensure we have a valid CRS
        if export_gdf.crs is None:
            export_gdf.crs = 'EPSG:4326'
        
        # Convert to WGS84 for better compatibility
        if export_gdf.crs != 'EPSG:4326':
            export_gdf = export_gdf.to_crs('EPSG:4326')
        
        # Clean column names for shapefile compatibility
        export_gdf.columns = [self._clean_column_name(col) for col in export_gdf.columns]
        
        # Add metadata if requested
        if self.include_metadata:
            export_gdf = self._add_metadata(export_gdf)
        
        return export_gdf
    
    def _clean_column_name(self, name: str) -> str:
        """Clean column name for shapefile compatibility."""
        # Shapefile column names are limited to 10 characters
        cleaned = name.replace(' ', '_').replace('-', '_')
        return cleaned[:10]
    
    def _add_metadata(self, gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
        """Add metadata columns to GeoDataFrame."""
        # Add timestamp
        from datetime import datetime
        gdf['timestamp'] = datetime.now().isoformat()
        
        # Add source information
        gdf['source'] = 'CrisisMapper'
        gdf['version'] = '1.0.0'
        
        return gdf
    
    def export_summary_report(self, 
                             gdf: gpd.GeoDataFrame,
                             summary_stats: Dict,
                             filename_prefix: str = "disaster_summary") -> Path:
        """
        Export a summary report of disaster detection results.
        
        Args:
            gdf: GeoDataFrame with disaster areas
            summary_stats: Summary statistics dictionary
            filename_prefix: Prefix for output file
            
        Returns:
            Path to exported report
        """
        filename = f"{filename_prefix}.json"
        file_path = self.output_dir / filename
        
        # Create comprehensive report
        report = {
            'metadata': {
                'timestamp': summary_stats.get('timestamp', ''),
                'total_areas': summary_stats.get('total_areas', 0),
                'total_area_m2': summary_stats.get('total_area_m2', 0),
                'total_area_km2': summary_stats.get('total_area_km2', 0),
                'bounds': summary_stats.get('bounds', [])
            },
            'class_summary': summary_stats.get('class_summary', {}),
            'severity_distribution': summary_stats.get('severity_distribution', {}),
            'geometry_info': {
                'crs': str(gdf.crs) if gdf.crs else 'Unknown',
                'feature_count': len(gdf),
                'geometry_types': gdf.geometry.geom_type.value_counts().to_dict()
            }
        }
        
        # Add individual feature information
        if len(gdf) > 0:
            report['features'] = []
            for idx, row in gdf.iterrows():
                feature_info = {
                    'id': idx,
                    'class_name': row.get('class_name', 'Unknown'),
                    'confidence': row.get('confidence', 0.0),
                    'area_m2': row.get('area_m2', 0.0),
                    'geometry': mapping(row.geometry) if hasattr(row, 'geometry') else None
                }
                report['features'].append(feature_info)
        
        # Export report
        with open(file_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        logger.info(f"Summary report exported to {file_path}")
        return file_path
    
    def create_visualization_data(self, gdf: gpd.GeoDataFrame) -> Dict:
        """
        Create data structure optimized for web visualization.
        
        Args:
            gdf: GeoDataFrame with disaster areas
            
        Returns:
            Dictionary with visualization data
        """
        # Convert to GeoJSON format for web mapping
        geojson_data = json.loads(gdf.to_json())
        
        # Add additional properties for visualization
        for feature in geojson_data['features']:
            props = feature['properties']
            
            # Add color based on class
            class_name = props.get('class_name', 'unknown')
            props['color'] = self._get_class_color(class_name)
            
            # Add size category based on area
            area_m2 = props.get('area_m2', 0)
            props['size_category'] = self._get_size_category(area_m2)
            
            # Add popup text
            props['popup_text'] = self._create_popup_text(props)
        
        return geojson_data
    
    def _get_class_color(self, class_name: str) -> str:
        """Get color for disaster class."""
        color_map = {
            'flood': '#0066CC',
            'wildfire': '#FF0000',
            'earthquake': '#FFD700',
            'landslide': '#800080',
            'hurricane': '#00FFFF'
        }
        return color_map.get(class_name.lower(), '#666666')
    
    def _get_size_category(self, area_m2: float) -> str:
        """Get size category based on area."""
        if area_m2 < 1000:  # < 0.001 km²
            return 'small'
        elif area_m2 < 100000:  # < 0.1 km²
            return 'medium'
        else:  # >= 0.1 km²
            return 'large'
    
    def _create_popup_text(self, properties: Dict) -> str:
        """Create popup text for map visualization."""
        class_name = properties.get('class_name', 'Unknown')
        confidence = properties.get('confidence', 0.0)
        area_m2 = properties.get('area_m2', 0.0)
        area_km2 = area_m2 / 1_000_000
        
        return f"""
        <b>{class_name.title()}</b><br>
        Confidence: {confidence:.2f}<br>
        Area: {area_km2:.3f} km²<br>
        Severity: {properties.get('severity', 'Unknown')}
        """
