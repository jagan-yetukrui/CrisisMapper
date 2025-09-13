#!/usr/bin/env python3
"""
Basic usage example for CrisisMapper.

This script demonstrates how to use CrisisMapper for disaster detection
on satellite and drone imagery.
"""

import sys
from pathlib import Path

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
    """Main example function."""
    print("🌍 CrisisMapper Basic Usage Example")
    print("="*50)
    
    # Load configuration
    try:
        config = load_config()
        print("✅ Configuration loaded")
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return 1
    
    # Initialize components
    try:
        print("🔄 Initializing components...")
        detector = DisasterDetector(config)
        classifier = DisasterClassifier(config)
        geospatial_processor = GeospatialProcessor(config)
        exporter = GeospatialExporter(config)
        print("✅ Components initialized")
    except Exception as e:
        print(f"❌ Failed to initialize components: {e}")
        return 1
    
    # Example 1: Single image detection
    print("\n📸 Example 1: Single Image Detection")
    print("-" * 40)
    
    # Check if sample image exists
    sample_image = Path("data/sample/test_image.jpg")
    if not sample_image.exists():
        print(f"⚠️  Sample image not found: {sample_image}")
        print("   Please run setup.py first to create sample data")
        return 1
    
    try:
        # Run detection
        print(f"🔍 Detecting disasters in {sample_image}")
        detection_result = detector.detect(str(sample_image))
        
        # Classify results
        classification_result = classifier.classify_detection(detection_result)
        
        # Print results
        print_detection_results(classification_result)
        
        # Process geospatially
        print("\n🗺️  Processing geospatial data...")
        gdf = geospatial_processor.process_detection_results([detection_result])
        
        # Export results
        print("📤 Exporting results...")
        exported_files = exporter.export_results(
            gdf,
            filename_prefix="example_detection",
            formats=["geojson", "shapefile"]
        )
        
        print("✅ Export completed:")
        for format_name, file_path in exported_files.items():
            print(f"   {format_name}: {file_path}")
        
    except Exception as e:
        print(f"❌ Detection failed: {e}")
        return 1
    
    # Example 2: Batch processing
    print("\n📁 Example 2: Batch Processing")
    print("-" * 40)
    
    # Create multiple sample images
    try:
        import cv2
        import numpy as np
        
        batch_dir = Path("data/sample/batch")
        batch_dir.mkdir(parents=True, exist_ok=True)
        
        # Create sample images
        for i in range(3):
            image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
            
            # Add some colored rectangles
            cv2.rectangle(image, (100 + i*50, 100), (200 + i*50, 200), (0, 0, 255), -1)
            cv2.rectangle(image, (300 + i*50, 150), (400 + i*50, 250), (255, 0, 0), -1)
            
            image_path = batch_dir / f"sample_{i+1}.jpg"
            cv2.imwrite(str(image_path), image)
        
        print(f"✅ Created {len(list(batch_dir.glob('*.jpg')))} sample images")
        
        # Run batch detection
        print("🔍 Running batch detection...")
        batch_results = detector.detect_batch([str(p) for p in batch_dir.glob("*.jpg")])
        
        # Process results
        successful_results = [r for r in batch_results if 'error' not in r]
        print(f"✅ Processed {len(successful_results)}/{len(batch_results)} images successfully")
        
        # Export batch results
        if successful_results:
            batch_gdf = geospatial_processor.process_detection_results(successful_results)
            batch_exported = exporter.export_results(
                batch_gdf,
                filename_prefix="batch_example",
                formats=["geojson"]
            )
            print("✅ Batch export completed")
        
    except ImportError:
        print("⚠️  OpenCV not available, skipping batch example")
    except Exception as e:
        print(f"❌ Batch processing failed: {e}")
    
    # Example 3: Performance analysis
    print("\n📊 Example 3: Performance Analysis")
    print("-" * 40)
    
    stats = detector.get_performance_stats()
    if stats:
        print("Performance Statistics:")
        print(f"  Total Inferences: {stats.get('total_inferences', 0)}")
        print(f"  Total Detections: {stats.get('total_detections', 0)}")
        print(f"  Average Inference Time: {stats.get('average_inference_time', 0):.3f}s")
        print(f"  Average FPS: {stats.get('average_fps', 0):.2f}")
        print(f"  Min Inference Time: {stats.get('min_inference_time', 0):.3f}s")
        print(f"  Max Inference Time: {stats.get('max_inference_time', 0):.3f}s")
    else:
        print("No performance data available")
    
    print("\n🎉 Example completed successfully!")
    print("\nNext steps:")
    print("1. Explore the exported files in the results/ directory")
    print("2. Try the Streamlit dashboard: python scripts/run_dashboard.py")
    print("3. Run the API server: python scripts/run_api.py")
    print("4. Check the documentation in README.md")
    
    return 0


def print_detection_results(result):
    """Print detection results in a formatted way."""
    summary = result['summary']
    
    print(f"\n📊 Detection Summary:")
    print(f"  Total Detections: {summary['total_detections']}")
    print(f"  Average Confidence: {summary['average_confidence']:.3f}")
    print(f"  Average Risk Score: {summary['average_risk_score']:.3f}")
    print(f"  Total Coverage: {summary['total_coverage']:.1f}%")
    
    if summary['class_counts']:
        print(f"\n🏷️  Class Distribution:")
        for class_name, count in summary['class_counts'].items():
            print(f"    {class_name}: {count}")
    
    if summary['severity_counts']:
        print(f"\n⚠️  Severity Distribution:")
        for severity, count in summary['severity_counts'].items():
            print(f"    {severity}: {count}")
    
    print(f"\n🔍 Detailed Detections:")
    for i, det in enumerate(result['enhanced_detections']):
        print(f"  Detection {i+1}:")
        print(f"    Class: {det['class_name']}")
        print(f"    Confidence: {det['confidence']:.3f}")
        print(f"    Severity: {det['severity']}")
        print(f"    Area: {det['area']:.1f} m²")
        print(f"    Coverage: {det['coverage_percentage']:.1f}%")
        print(f"    Risk Score: {det['risk_score']:.3f}")


if __name__ == "__main__":
    sys.exit(main())
