"""
Comprehensive Metrics Collection System for CrisisMapper.

This module provides enterprise-grade metrics collection, aggregation,
and analysis for monitoring system performance and health.
"""

import time
import psutil
import threading
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from collections import defaultdict, deque
import json
import logging
from pathlib import Path
import numpy as np
import pandas as pd

from ..utils.logger import setup_logger

logger = setup_logger(__name__)

@dataclass
class MetricPoint:
    """A single metric data point."""
    timestamp: datetime
    value: float
    tags: Dict[str, str] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class MetricSeries:
    """A series of metric data points."""
    name: str
    points: deque = field(default_factory=lambda: deque(maxlen=10000))
    aggregation_interval: int = 60  # seconds
    last_aggregation: Optional[datetime] = None

class MetricsCollector:
    """
    Comprehensive metrics collection system for CrisisMapper.
    
    Collects system metrics, application metrics, and custom metrics
    with real-time aggregation and analysis capabilities.
    """
    
    def __init__(self, 
                 collection_interval: int = 10,
                 retention_days: int = 30,
                 storage_path: str = "metrics"):
        """
        Initialize the MetricsCollector.
        
        Args:
            collection_interval: Interval for collecting metrics (seconds)
            retention_days: Number of days to retain metrics data
            storage_path: Path to store metrics data
        """
        self.collection_interval = collection_interval
        self.retention_days = retention_days
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Metrics storage
        self.metrics: Dict[str, MetricSeries] = {}
        self.custom_metrics: Dict[str, Callable] = {}
        
        # Collection control
        self.collection_active = False
        self.collection_thread = None
        
        # Performance tracking
        self.performance_counters = defaultdict(int)
        self.timing_data = defaultdict(list)
        
        # Initialize system metrics
        self._initialize_system_metrics()
        
        logger.info(f"MetricsCollector initialized with {collection_interval}s interval")
    
    def start_collection(self):
        """Start metrics collection."""
        if self.collection_active:
            logger.warning("Metrics collection already active")
            return
        
        self.collection_active = True
        self.collection_thread = threading.Thread(target=self._collection_loop, daemon=True)
        self.collection_thread.start()
        
        logger.info("Started metrics collection")
    
    def stop_collection(self):
        """Stop metrics collection."""
        self.collection_active = False
        if self.collection_thread:
            self.collection_thread.join(timeout=5)
        
        logger.info("Stopped metrics collection")
    
    def add_custom_metric(self, name: str, collector_func: Callable[[], float]):
        """
        Add a custom metric collector.
        
        Args:
            name: Metric name
            collector_func: Function that returns metric value
        """
        self.custom_metrics[name] = collector_func
        logger.info(f"Added custom metric: {name}")
    
    def record_metric(self, 
                     name: str, 
                     value: float, 
                     tags: Optional[Dict[str, str]] = None,
                     metadata: Optional[Dict[str, Any]] = None):
        """
        Record a metric value.
        
        Args:
            name: Metric name
            value: Metric value
            tags: Optional tags for the metric
            metadata: Optional metadata
        """
        if name not in self.metrics:
            self.metrics[name] = MetricSeries(name=name)
        
        point = MetricPoint(
            timestamp=datetime.now(),
            value=value,
            tags=tags or {},
            metadata=metadata or {}
        )
        
        self.metrics[name].points.append(point)
    
    def record_timing(self, operation: str, duration: float):
        """
        Record timing information for an operation.
        
        Args:
            operation: Operation name
            duration: Duration in seconds
        """
        self.timing_data[operation].append(duration)
        self.record_metric(f"timing.{operation}", duration)
    
    def increment_counter(self, name: str, value: int = 1, tags: Optional[Dict[str, str]] = None):
        """
        Increment a counter metric.
        
        Args:
            name: Counter name
            value: Value to increment by
            tags: Optional tags
        """
        self.performance_counters[name] += value
        self.record_metric(f"counter.{name}", self.performance_counters[name], tags)
    
    def get_metric_summary(self, name: str, hours: int = 24) -> Dict[str, Any]:
        """
        Get summary statistics for a metric.
        
        Args:
            name: Metric name
            hours: Number of hours to look back
            
        Returns:
            Summary statistics
        """
        if name not in self.metrics:
            return {}
        
        series = self.metrics[name]
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter points within time range
        recent_points = [
            point for point in series.points 
            if point.timestamp >= cutoff_time
        ]
        
        if not recent_points:
            return {}
        
        values = [point.value for point in recent_points]
        
        return {
            "count": len(values),
            "mean": np.mean(values),
            "median": np.median(values),
            "std": np.std(values),
            "min": np.min(values),
            "max": np.max(values),
            "p95": np.percentile(values, 95),
            "p99": np.percentile(values, 99),
            "latest": values[-1] if values else None,
            "time_range": {
                "start": recent_points[0].timestamp.isoformat(),
                "end": recent_points[-1].timestamp.isoformat()
            }
        }
    
    def get_all_metrics_summary(self, hours: int = 24) -> Dict[str, Any]:
        """
        Get summary for all metrics.
        
        Args:
            hours: Number of hours to look back
            
        Returns:
            Summary of all metrics
        """
        summary = {}
        
        for name in self.metrics:
            summary[name] = self.get_metric_summary(name, hours)
        
        return summary
    
    def export_metrics(self, 
                      output_path: Optional[str] = None,
                      format: str = "json",
                      hours: int = 24) -> str:
        """
        Export metrics data.
        
        Args:
            output_path: Output file path
            format: Export format (json, csv, parquet)
            hours: Number of hours to export
            
        Returns:
            Path to exported file
        """
        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = self.storage_path / f"metrics_export_{timestamp}.{format}"
        
        output_path = Path(output_path)
        
        # Collect all metrics data
        all_data = []
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        for name, series in self.metrics.items():
            for point in series.points:
                if point.timestamp >= cutoff_time:
                    all_data.append({
                        "timestamp": point.timestamp.isoformat(),
                        "metric_name": name,
                        "value": point.value,
                        "tags": point.tags,
                        "metadata": point.metadata
                    })
        
        # Export based on format
        if format == "json":
            with open(output_path, 'w') as f:
                json.dump(all_data, f, indent=2, default=str)
        
        elif format == "csv":
            df = pd.DataFrame(all_data)
            df.to_csv(output_path, index=False)
        
        elif format == "parquet":
            df = pd.DataFrame(all_data)
            df.to_parquet(output_path, index=False)
        
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Exported metrics to {output_path}")
        return str(output_path)
    
    def _collection_loop(self):
        """Main collection loop."""
        while self.collection_active:
            try:
                self._collect_system_metrics()
                self._collect_application_metrics()
                self._collect_custom_metrics()
                self._aggregate_metrics()
                
                time.sleep(self.collection_interval)
                
            except Exception as e:
                logger.error(f"Error in metrics collection: {e}")
                time.sleep(self.collection_interval)
    
    def _initialize_system_metrics(self):
        """Initialize system metrics collection."""
        # CPU metrics
        self.metrics["cpu_percent"] = MetricSeries("cpu_percent")
        self.metrics["cpu_count"] = MetricSeries("cpu_count")
        self.metrics["load_average"] = MetricSeries("load_average")
        
        # Memory metrics
        self.metrics["memory_total"] = MetricSeries("memory_total")
        self.metrics["memory_available"] = MetricSeries("memory_available")
        self.metrics["memory_percent"] = MetricSeries("memory_percent")
        self.metrics["memory_used"] = MetricSeries("memory_used")
        
        # Disk metrics
        self.metrics["disk_total"] = MetricSeries("disk_total")
        self.metrics["disk_used"] = MetricSeries("disk_used")
        self.metrics["disk_free"] = MetricSeries("disk_free")
        self.metrics["disk_percent"] = MetricSeries("disk_percent")
        
        # Network metrics
        self.metrics["network_bytes_sent"] = MetricSeries("network_bytes_sent")
        self.metrics["network_bytes_recv"] = MetricSeries("network_bytes_recv")
        self.metrics["network_packets_sent"] = MetricSeries("network_packets_sent")
        self.metrics["network_packets_recv"] = MetricSeries("network_packets_recv")
    
    def _collect_system_metrics(self):
        """Collect system-level metrics."""
        try:
            # CPU metrics
            cpu_percent = psutil.cpu_percent(interval=1)
            cpu_count = psutil.cpu_count()
            
            self.record_metric("cpu_percent", cpu_percent)
            self.record_metric("cpu_count", cpu_count)
            
            # Load average (Unix systems)
            try:
                load_avg = psutil.getloadavg()[0]
                self.record_metric("load_average", load_avg)
            except AttributeError:
                pass  # Not available on Windows
            
            # Memory metrics
            memory = psutil.virtual_memory()
            self.record_metric("memory_total", memory.total)
            self.record_metric("memory_available", memory.available)
            self.record_metric("memory_percent", memory.percent)
            self.record_metric("memory_used", memory.used)
            
            # Disk metrics
            disk = psutil.disk_usage('/')
            self.record_metric("disk_total", disk.total)
            self.record_metric("disk_used", disk.used)
            self.record_metric("disk_free", disk.free)
            self.record_metric("disk_percent", (disk.used / disk.total) * 100)
            
            # Network metrics
            network = psutil.net_io_counters()
            self.record_metric("network_bytes_sent", network.bytes_sent)
            self.record_metric("network_bytes_recv", network.bytes_recv)
            self.record_metric("network_packets_sent", network.packets_sent)
            self.record_metric("network_packets_recv", network.packets_recv)
            
        except Exception as e:
            logger.error(f"Error collecting system metrics: {e}")
    
    def _collect_application_metrics(self):
        """Collect application-specific metrics."""
        try:
            # Process metrics
            process = psutil.Process()
            
            # Process CPU and memory
            process_cpu = process.cpu_percent()
            process_memory = process.memory_info()
            
            self.record_metric("process_cpu_percent", process_cpu)
            self.record_metric("process_memory_rss", process_memory.rss)
            self.record_metric("process_memory_vms", process_memory.vms)
            
            # Process file descriptors
            try:
                num_fds = process.num_fds()
                self.record_metric("process_file_descriptors", num_fds)
            except AttributeError:
                pass  # Not available on Windows
            
            # Process threads
            num_threads = process.num_threads()
            self.record_metric("process_threads", num_threads)
            
        except Exception as e:
            logger.error(f"Error collecting application metrics: {e}")
    
    def _collect_custom_metrics(self):
        """Collect custom metrics."""
        for name, collector_func in self.custom_metrics.items():
            try:
                value = collector_func()
                self.record_metric(name, value)
            except Exception as e:
                logger.error(f"Error collecting custom metric {name}: {e}")
    
    def _aggregate_metrics(self):
        """Aggregate metrics based on their aggregation intervals."""
        current_time = datetime.now()
        
        for name, series in self.metrics.items():
            if (series.last_aggregation is None or 
                (current_time - series.last_aggregation).seconds >= series.aggregation_interval):
                
                # Perform aggregation
                self._aggregate_series(series)
                series.last_aggregation = current_time
    
    def _aggregate_series(self, series: MetricSeries):
        """Aggregate a metric series."""
        if len(series.points) < 2:
            return
        
        # Calculate aggregated values
        values = [point.value for point in series.points]
        
        aggregated = {
            "mean": np.mean(values),
            "min": np.min(values),
            "max": np.max(values),
            "count": len(values)
        }
        
        # Store aggregated values
        for agg_name, agg_value in aggregated.items():
            self.record_metric(
                f"{series.name}.{agg_name}",
                agg_value,
                tags={"aggregation": "1min"}
            )
    
    def cleanup_old_metrics(self):
        """Clean up old metrics data."""
        cutoff_time = datetime.now() - timedelta(days=self.retention_days)
        
        for series in self.metrics.values():
            # Remove old points
            while series.points and series.points[0].timestamp < cutoff_time:
                series.points.popleft()
        
        logger.info("Cleaned up old metrics data")
    
    def get_health_status(self) -> Dict[str, Any]:
        """
        Get overall system health status.
        
        Returns:
            Health status dictionary
        """
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "metrics": {}
        }
        
        # Check critical metrics
        critical_metrics = {
            "cpu_percent": {"threshold": 90, "current": None},
            "memory_percent": {"threshold": 90, "current": None},
            "disk_percent": {"threshold": 90, "current": None}
        }
        
        for metric_name, config in critical_metrics.items():
            summary = self.get_metric_summary(metric_name, hours=1)
            if summary and "latest" in summary:
                current_value = summary["latest"]
                config["current"] = current_value
                
                if current_value > config["threshold"]:
                    health["status"] = "warning"
                    health["alerts"] = health.get("alerts", [])
                    health["alerts"].append(
                        f"{metric_name} is {current_value:.1f}% (threshold: {config['threshold']}%)"
                    )
        
        health["metrics"] = critical_metrics
        return health
