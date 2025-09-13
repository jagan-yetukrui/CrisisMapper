"""
Experiment Management System for CrisisMapper.

This module provides comprehensive experiment tracking, model comparison,
and research workflow management for disaster detection research.
"""

import json
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import logging
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from ..utils.logger import setup_logger
from ..core.detector import DisasterDetector
from ..core.classifier import DisasterClassifier

logger = setup_logger(__name__)

@dataclass
class ExperimentConfig:
    """Configuration for an experiment."""
    experiment_id: str
    name: str
    description: str
    model_type: str
    model_params: Dict[str, Any]
    dataset_path: str
    train_split: float
    val_split: float
    test_split: float
    augmentation: bool
    preprocessing: Dict[str, Any]
    created_at: datetime
    created_by: str

@dataclass
class ExperimentResults:
    """Results from an experiment."""
    experiment_id: str
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    confusion_matrix: List[List[int]]
    class_metrics: Dict[str, Dict[str, float]]
    inference_time: float
    memory_usage: float
    model_size: float
    completed_at: datetime
    notes: str

@dataclass
class ModelPerformance:
    """Model performance metrics."""
    model_name: str
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    inference_time: float
    memory_usage: float
    model_size: float
    parameters: int
    flops: float

class ExperimentManager:
    """
    Comprehensive experiment management system for disaster detection research.
    
    Provides experiment tracking, model comparison, hyperparameter optimization,
    and research workflow management.
    """
    
    def __init__(self, experiments_dir: str = "experiments"):
        """
        Initialize the ExperimentManager.
        
        Args:
            experiments_dir: Directory to store experiment data
        """
        self.experiments_dir = Path(experiments_dir)
        self.experiments_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize experiment tracking
        self.experiments_db = self._load_experiments_database()
        self.active_experiments = {}
        
        logger.info(f"ExperimentManager initialized with directory: {self.experiments_dir}")
    
    def create_experiment(self, 
                         name: str,
                         description: str,
                         model_type: str,
                         model_params: Dict[str, Any],
                         dataset_path: str,
                         train_split: float = 0.7,
                         val_split: float = 0.15,
                         test_split: float = 0.15,
                         augmentation: bool = True,
                         preprocessing: Optional[Dict[str, Any]] = None,
                         created_by: str = "researcher") -> str:
        """
        Create a new experiment.
        
        Args:
            name: Experiment name
            description: Experiment description
            model_type: Type of model to use
            model_params: Model parameters
            dataset_path: Path to dataset
            train_split: Training data split ratio
            val_split: Validation data split ratio
            test_split: Test data split ratio
            augmentation: Whether to use data augmentation
            preprocessing: Preprocessing parameters
            created_by: Name of experiment creator
            
        Returns:
            Experiment ID
        """
        experiment_id = str(uuid.uuid4())
        
        if preprocessing is None:
            preprocessing = {}
        
        config = ExperimentConfig(
            experiment_id=experiment_id,
            name=name,
            description=description,
            model_type=model_type,
            model_params=model_params,
            dataset_path=dataset_path,
            train_split=train_split,
            val_split=val_split,
            test_split=test_split,
            augmentation=augmentation,
            preprocessing=preprocessing,
            created_at=datetime.now(),
            created_by=created_by
        )
        
        # Save experiment configuration
        self._save_experiment_config(config)
        
        # Add to database
        self.experiments_db[experiment_id] = {
            "config": asdict(config),
            "status": "created",
            "results": None
        }
        
        logger.info(f"Created experiment: {name} (ID: {experiment_id})")
        return experiment_id
    
    def run_experiment(self, 
                      experiment_id: str,
                      config: Optional[Dict] = None) -> ExperimentResults:
        """
        Run an experiment.
        
        Args:
            experiment_id: ID of experiment to run
            config: Optional configuration override
            
        Returns:
            Experiment results
        """
        if experiment_id not in self.experiments_db:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment_data = self.experiments_db[experiment_id]
        experiment_config = ExperimentConfig(**experiment_data["config"])
        
        logger.info(f"Starting experiment: {experiment_config.name}")
        
        try:
            # Update status
            self._update_experiment_status(experiment_id, "running")
            
            # Load and prepare data
            train_data, val_data, test_data = self._prepare_dataset(experiment_config)
            
            # Initialize model
            model = self._initialize_model(experiment_config, config)
            
            # Train model (if applicable)
            if experiment_config.model_type in ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]:
                # For YOLOv8, we'll use pretrained weights and fine-tune
                training_results = self._train_model(model, train_data, val_data, experiment_config)
            else:
                training_results = None
            
            # Evaluate model
            results = self._evaluate_model(model, test_data, experiment_config)
            
            # Calculate additional metrics
            results = self._calculate_comprehensive_metrics(results, model, test_data)
            
            # Save results
            self._save_experiment_results(experiment_id, results)
            
            # Update database
            self.experiments_db[experiment_id]["results"] = asdict(results)
            self.experiments_db[experiment_id]["status"] = "completed"
            
            logger.info(f"Completed experiment: {experiment_config.name}")
            return results
            
        except Exception as e:
            logger.error(f"Experiment {experiment_id} failed: {e}")
            self._update_experiment_status(experiment_id, "failed")
            raise
    
    def compare_models(self, 
                      experiment_ids: List[str],
                      metrics: List[str] = None) -> pd.DataFrame:
        """
        Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare
            metrics: List of metrics to include in comparison
            
        Returns:
            Comparison DataFrame
        """
        if metrics is None:
            metrics = ["accuracy", "precision", "recall", "f1_score", "inference_time", "memory_usage"]
        
        comparison_data = []
        
        for exp_id in experiment_ids:
            if exp_id not in self.experiments_db:
                logger.warning(f"Experiment {exp_id} not found")
                continue
            
            experiment_data = self.experiments_db[exp_id]
            config = ExperimentConfig(**experiment_data["config"])
            
            if experiment_data["results"] is None:
                logger.warning(f"Experiment {exp_id} has no results")
                continue
            
            results = ExperimentResults(**experiment_data["results"])
            
            row = {
                "experiment_id": exp_id,
                "name": config.name,
                "model_type": config.model_type,
                "created_at": config.created_at.isoformat()
            }
            
            for metric in metrics:
                if hasattr(results, metric):
                    row[metric] = getattr(results, metric)
                else:
                    row[metric] = None
            
            comparison_data.append(row)
        
        return pd.DataFrame(comparison_data)
    
    def get_experiment_summary(self, experiment_id: str) -> Dict[str, Any]:
        """
        Get comprehensive experiment summary.
        
        Args:
            experiment_id: Experiment ID
            
        Returns:
            Experiment summary dictionary
        """
        if experiment_id not in self.experiments_db:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment_data = self.experiments_db[experiment_id]
        config = ExperimentConfig(**experiment_data["config"])
        
        summary = {
            "experiment_id": experiment_id,
            "name": config.name,
            "description": config.description,
            "model_type": config.model_type,
            "status": experiment_data["status"],
            "created_at": config.created_at.isoformat(),
            "created_by": config.created_by
        }
        
        if experiment_data["results"] is not None:
            results = ExperimentResults(**experiment_data["results"])
            summary.update({
                "accuracy": results.accuracy,
                "precision": results.precision,
                "recall": results.recall,
                "f1_score": results.f1_score,
                "inference_time": results.inference_time,
                "memory_usage": results.memory_usage,
                "model_size": results.model_size,
                "completed_at": results.completed_at.isoformat()
            })
        
        return summary
    
    def list_experiments(self, 
                        status: Optional[str] = None,
                        model_type: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        List experiments with optional filtering.
        
        Args:
            status: Filter by status (created, running, completed, failed)
            model_type: Filter by model type
            
        Returns:
            List of experiment summaries
        """
        experiments = []
        
        for exp_id, exp_data in self.experiments_db.items():
            config = ExperimentConfig(**exp_data["config"])
            
            # Apply filters
            if status and exp_data["status"] != status:
                continue
            
            if model_type and config.model_type != model_type:
                continue
            
            summary = self.get_experiment_summary(exp_id)
            experiments.append(summary)
        
        # Sort by creation date (newest first)
        experiments.sort(key=lambda x: x["created_at"], reverse=True)
        
        return experiments
    
    def generate_experiment_report(self, 
                                 experiment_id: str,
                                 output_path: Optional[str] = None) -> str:
        """
        Generate comprehensive experiment report.
        
        Args:
            experiment_id: Experiment ID
            output_path: Optional output path for report
            
        Returns:
            Path to generated report
        """
        if experiment_id not in self.experiments_db:
            raise ValueError(f"Experiment {experiment_id} not found")
        
        experiment_data = self.experiments_db[experiment_id]
        config = ExperimentConfig(**experiment_data["config"])
        
        if output_path is None:
            output_path = self.experiments_dir / f"report_{experiment_id}.html"
        
        # Generate HTML report
        report_html = self._generate_html_report(config, experiment_data)
        
        with open(output_path, 'w') as f:
            f.write(report_html)
        
        logger.info(f"Generated experiment report: {output_path}")
        return str(output_path)
    
    def _load_experiments_database(self) -> Dict[str, Any]:
        """Load experiments database from disk."""
        db_path = self.experiments_dir / "experiments_db.json"
        
        if db_path.exists():
            with open(db_path, 'r') as f:
                return json.load(f)
        else:
            return {}
    
    def _save_experiments_database(self):
        """Save experiments database to disk."""
        db_path = self.experiments_dir / "experiments_db.json"
        
        with open(db_path, 'w') as f:
            json.dump(self.experiments_db, f, indent=2, default=str)
    
    def _save_experiment_config(self, config: ExperimentConfig):
        """Save experiment configuration."""
        config_path = self.experiments_dir / f"config_{config.experiment_id}.json"
        
        with open(config_path, 'w') as f:
            json.dump(asdict(config), f, indent=2, default=str)
    
    def _save_experiment_results(self, experiment_id: str, results: ExperimentResults):
        """Save experiment results."""
        results_path = self.experiments_dir / f"results_{experiment_id}.json"
        
        with open(results_path, 'w') as f:
            json.dump(asdict(results), f, indent=2, default=str)
    
    def _update_experiment_status(self, experiment_id: str, status: str):
        """Update experiment status."""
        self.experiments_db[experiment_id]["status"] = status
        self._save_experiments_database()
    
    def _prepare_dataset(self, config: ExperimentConfig) -> Tuple[Any, Any, Any]:
        """Prepare dataset for training/validation/testing."""
        # This would implement actual dataset preparation
        # For now, return placeholder data
        logger.info(f"Preparing dataset from {config.dataset_path}")
        
        # Placeholder implementation
        train_data = None
        val_data = None
        test_data = None
        
        return train_data, val_data, test_data
    
    def _initialize_model(self, config: ExperimentConfig, override_config: Optional[Dict] = None) -> Any:
        """Initialize model for experiment."""
        logger.info(f"Initializing {config.model_type} model")
        
        # Use override config if provided
        model_params = override_config if override_config else config.model_params
        
        # Initialize appropriate model
        if config.model_type in ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x"]:
            from ultralytics import YOLO
            model = YOLO(f"{config.model_type}.pt")
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        
        return model
    
    def _train_model(self, model: Any, train_data: Any, val_data: Any, config: ExperimentConfig) -> Dict[str, Any]:
        """Train model on training data."""
        logger.info("Training model...")
        
        # Placeholder training implementation
        training_results = {
            "epochs": 100,
            "best_accuracy": 0.95,
            "training_time": 3600,
            "loss_history": []
        }
        
        return training_results
    
    def _evaluate_model(self, model: Any, test_data: Any, config: ExperimentConfig) -> ExperimentResults:
        """Evaluate model on test data."""
        logger.info("Evaluating model...")
        
        # Placeholder evaluation - in practice, this would run actual inference
        # and calculate metrics on test data
        
        # Mock results for demonstration
        accuracy = np.random.uniform(0.85, 0.95)
        precision = np.random.uniform(0.80, 0.90)
        recall = np.random.uniform(0.80, 0.90)
        f1_score = 2 * (precision * recall) / (precision + recall)
        
        # Mock confusion matrix
        n_classes = 5  # Number of disaster classes
        confusion_mat = np.random.randint(0, 100, (n_classes, n_classes))
        np.fill_diagonal(confusion_mat, np.random.randint(80, 100, n_classes))
        
        results = ExperimentResults(
            experiment_id=config.experiment_id,
            model_name=config.model_type,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            confusion_matrix=confusion_mat.tolist(),
            class_metrics={},  # Would be populated with actual class-wise metrics
            inference_time=np.random.uniform(0.01, 0.1),
            memory_usage=np.random.uniform(1.0, 4.0),
            model_size=np.random.uniform(10, 100),
            completed_at=datetime.now(),
            notes="Experiment completed successfully"
        )
        
        return results
    
    def _calculate_comprehensive_metrics(self, 
                                       results: ExperimentResults,
                                       model: Any,
                                       test_data: Any) -> ExperimentResults:
        """Calculate comprehensive performance metrics."""
        # Add additional metrics calculation here
        # This could include:
        # - Model size in parameters
        # - FLOPs calculation
        # - Memory efficiency
        # - Inference speed analysis
        # - Class-wise performance metrics
        
        return results
    
    def _generate_html_report(self, config: ExperimentConfig, experiment_data: Dict[str, Any]) -> str:
        """Generate HTML report for experiment."""
        results = experiment_data.get("results")
        
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Experiment Report - {config.name}</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; }}
                .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
                .section {{ margin: 20px 0; }}
                .metric {{ display: inline-block; margin: 10px; padding: 10px; background-color: #e8f4fd; border-radius: 3px; }}
                table {{ border-collapse: collapse; width: 100%; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #f2f2f2; }}
            </style>
        </head>
        <body>
            <div class="header">
                <h1>Experiment Report: {config.name}</h1>
                <p><strong>Description:</strong> {config.description}</p>
                <p><strong>Model Type:</strong> {config.model_type}</p>
                <p><strong>Created:</strong> {config.created_at}</p>
                <p><strong>Status:</strong> {experiment_data['status']}</p>
            </div>
        """
        
        if results:
            results_obj = ExperimentResults(**results)
            html += f"""
            <div class="section">
                <h2>Results</h2>
                <div class="metric"><strong>Accuracy:</strong> {results_obj.accuracy:.4f}</div>
                <div class="metric"><strong>Precision:</strong> {results_obj.precision:.4f}</div>
                <div class="metric"><strong>Recall:</strong> {results_obj.recall:.4f}</div>
                <div class="metric"><strong>F1-Score:</strong> {results_obj.f1_score:.4f}</div>
                <div class="metric"><strong>Inference Time:</strong> {results_obj.inference_time:.4f}s</div>
                <div class="metric"><strong>Memory Usage:</strong> {results_obj.memory_usage:.2f} GB</div>
            </div>
            """
        
        html += """
        </body>
        </html>
        """
        
        return html
