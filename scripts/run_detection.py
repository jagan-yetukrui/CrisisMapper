#!/usr/bin/env python3
"""
Command-line script for running disaster detection.

This script provides a command-line interface for running disaster detection
on single images or batches of images.
"""

import argparse
import sys
from pathlib import Path
import json
from datetime import datetime

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from core.detector import DisasterDetector
from core.classifier import DisasterClassifier
from geospatial.processor import GeospatialProcessor
from geospatial.export import GeospatialExporter
from utils.config import load_config
from utils.logger import setup_logger

logger = setup_logger(__name__)


def main():
    """Main function for command-line interface."""
    parser = argparse.ArgumentParser(
        description="CrisisMapper Disaster Detection CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Detect disasters in a single image
  python scripts/run_detection.py --input image.jpg --output results/

  # Batch detection on a directory
  python scripts/run_detection.py --input data/images/ --output results/ --batch

  # Custom confidence threshold
  python scripts/run_detection.py --input image.jpg --confidence 0.7

  # Export multiple formats
  python scripts/run_detection.py --input image.jpg --export geojson shapefile kml
        """
    )
    
    # Input arguments
    parser.add_argument(
        "--input", "-i",
        required=True,
        help="Input image file or directory path"
    )
    
    parser.add_argument(
        "--output", "-o",
        default="results",
        help="Output directory for results (default: results)"
    )
    
    parser.add_argument(
        "--config", "-c",
        help="Path to configuration file"
    )
    
    # Detection parameters
    parser.add_argument(
        "--confidence",
        type=float,
        default=0.5,
        help="Confidence threshold (default: 0.5)"
    )
    
    parser.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="IoU threshold (default: 0.45)"
    )
    
    parser.add_argument(
        "--max-detections",
        type=int,
        default=1000,
        help="Maximum number of detections (default: 1000)"
    )
    
    # Processing options
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Process directory in batch mode"
    )
    
    parser.add_argument(
        "--tile",
        action="store_true",
        help="Tile large images for processing"
    )
    
    parser.add_argument(
        "--tile-size",
        type=int,
        default=1024,
        help="Tile size for large images (default: 1024)"
    )
    
    # Export options
    parser.add_argument(
        "--export",
        nargs="+",
        choices=["geojson", "shapefile", "kml", "csv"],
        default=["geojson"],
        help="Export formats (default: geojson)"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save detection results"
    )
    
    # Output options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Verbose output"
    )
    
    parser.add_argument(
        "--quiet", "-q",
        action="store_true",
        help="Quiet output"
    )
    
    args = parser.parse_args()
    
    # Set up logging level
    if args.verbose:
        log_level = "DEBUG"
    elif args.quiet:
        log_level = "WARNING"
    else:
        log_level = "INFO"
    
    # Load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        logger.error(f"Failed to load configuration: {e}")
        return 1
    
    # Update configuration with command line arguments
    config['model']['confidence_threshold'] = args.confidence
    config['model']['iou_threshold'] = args.iou
    config['model']['max_detections'] = args.max_detections
    config['data_processing']['tile_size'] = args.tile_size
    
    # Initialize components
    try:
        detector = DisasterDetector(config)
        classifier = DisasterClassifier(config)
        geospatial_processor = GeospatialProcessor(config)
        exporter = GeospatialExporter(config)
    except Exception as e:
        logger.error(f"Failed to initialize components: {e}")
        return 1
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Process input
    input_path = Path(args.input)
    
    if not input_path.exists():
        logger.error(f"Input path does not exist: {input_path}")
        return 1
    
    try:
        if args.batch or input_path.is_dir():
            # Batch processing
            logger.info(f"Processing directory: {input_path}")
            
            # Find image files
            image_extensions = config['data_processing']['input_formats']
            image_files = []
            for ext in image_extensions:
                image_files.extend(input_path.glob(f"*{ext}"))
                image_files.extend(input_path.glob(f"*{ext.upper()}"))
            
            if not image_files:
                logger.error(f"No image files found in {input_path}")
                return 1
            
            logger.info(f"Found {len(image_files)} image files")
            
            # Process each image
            results = []
            for i, image_file in enumerate(image_files):
                logger.info(f"Processing {i+1}/{len(image_files)}: {image_file.name}")
                
                try:
                    result = detector.detect(str(image_file), save_results=not args.no_save)
                    classification_result = classifier.classify_detection(result)
                    results.append(classification_result)
                    
                except Exception as e:
                    logger.error(f"Failed to process {image_file}: {e}")
                    results.append({'error': str(e), 'image_path': str(image_file)})
            
            # Export batch results
            if not args.no_save and results:
                export_batch_results(results, exporter, output_dir, args.export)
        
        else:
            # Single image processing
            logger.info(f"Processing image: {input_path}")
            
            try:
                result = detector.detect(str(input_path), save_results=not args.no_save)
                classification_result = classifier.classify_detection(result)
                
                # Export results
                if not args.no_save:
                    export_single_result(classification_result, exporter, output_dir, args.export)
                
                # Print summary
                print_detection_summary(classification_result)
                
            except Exception as e:
                logger.error(f"Failed to process {input_path}: {e}")
                return 1
        
        logger.info("Processing completed successfully")
        return 0
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Processing failed: {e}")
        return 1


def export_single_result(result, exporter, output_dir, export_formats):
    """Export single detection result."""
    try:
        # Process with geospatial processor
        gdf = exporter.geospatial_processor.process_detection_results([result['original_detection']])
        
        # Export to requested formats
        exported_files = exporter.export_results(
            gdf,
            filename_prefix=f"detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            formats=export_formats
        )
        
        logger.info(f"Exported results to: {list(exported_files.values())}")
        
    except Exception as e:
        logger.error(f"Export failed: {e}")


def export_batch_results(results, exporter, output_dir, export_formats):
    """Export batch detection results."""
    try:
        # Filter successful results
        successful_results = [r for r in results if 'error' not in r]
        
        if not successful_results:
            logger.warning("No successful results to export")
            return
        
        # Process with geospatial processor
        gdf = exporter.geospatial_processor.process_detection_results(
            [r['original_detection'] for r in successful_results]
        )
        
        # Export to requested formats
        exported_files = exporter.export_results(
            gdf,
            filename_prefix=f"batch_detection_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            formats=export_formats
        )
        
        logger.info(f"Exported batch results to: {list(exported_files.values())}")
        
    except Exception as e:
        logger.error(f"Batch export failed: {e}")


def print_detection_summary(result):
    """Print detection summary to console."""
    summary = result['summary']
    
    print("\n" + "="*50)
    print("DETECTION SUMMARY")
    print("="*50)
    print(f"Total Detections: {summary['total_detections']}")
    print(f"Average Confidence: {summary['average_confidence']:.3f}")
    print(f"Average Risk Score: {summary['average_risk_score']:.3f}")
    print(f"Total Coverage: {summary['total_coverage']:.1f}%")
    
    if summary['class_counts']:
        print("\nClass Distribution:")
        for class_name, count in summary['class_counts'].items():
            print(f"  {class_name}: {count}")
    
    if summary['severity_counts']:
        print("\nSeverity Distribution:")
        for severity, count in summary['severity_counts'].items():
            print(f"  {severity}: {count}")
    
    print("="*50)


if __name__ == "__main__":
    sys.exit(main())
