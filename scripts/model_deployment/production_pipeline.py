"""
Production ML Pipeline for Fashion Recommendation System

Components:
1. MLflow experiment tracking
2. Automated daily retraining
3. TorchServe deployment
4. Model drift monitoring
5. Performance dashboards
"""

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from pathlib import Path
import json
import schedule
import time
from collections import deque
import os
from collections import defaultdict
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('production_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class MLflowTracker:
    """MLflow experiment tracking and model registry"""
    
    def __init__(self, experiment_name="fashion-recommendation", tracking_uri="./mlruns"):
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment_name)
        
        self.experiment_name = experiment_name
        logger.info(f"MLflow tracking initialized: {experiment_name}")
    
    def start_run(self, run_name=None):
        """Start a new MLflow run"""
        if run_name is None:
            run_name = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.run = mlflow.start_run(run_name=run_name)
        logger.info(f"Started MLflow run: {run_name}")
        return self.run
    
    def log_params(self, params):
        """Log hyperparameters"""
        for key, value in params.items():
            mlflow.log_param(key, value)
    
    def log_metrics(self, metrics, step=None):
        """Log metrics"""
        for key, value in metrics.items():
            mlflow.log_metric(key, value, step=step)
    
    def log_model(self, model, artifact_path="model"):
        """Log PyTorch model"""
        mlflow.pytorch.log_model(model, artifact_path)
        logger.info(f"Model logged to MLflow: {artifact_path}")
    
    def register_model(self, model_name, run_id=None):
        """Register model in MLflow Model Registry"""
        if run_id is None:
            run_id = self.run.info.run_id
        
        model_uri = f"runs:/{run_id}/model"
        
        result = mlflow.register_model(model_uri, model_name)
        logger.info(f"Model registered: {model_name} version {result.version}")
        return result
    
    def transition_model_stage(self, model_name, version, stage="Production"):
        """Transition model to Production/Staging/Archived"""
        from mlflow.tracking import MlflowClient
        
        client = MlflowClient()
        client.transition_model_version_stage(
            name=model_name,
            version=version,
            stage=stage
        )
        logger.info(f"Model {model_name} v{version} → {stage}")
    
    def end_run(self):
        """End current run"""
        mlflow.end_run()


class ModelDriftMonitor:
    """Monitor model performance for drift detection"""
    
    def __init__(self, window_size=7, drift_threshold=0.15):
        self.window_size = window_size  # Days to monitor
        self.drift_threshold = drift_threshold  # 15% degradation triggers alert
        
        self.performance_history = deque(maxlen=window_size)
        self.baseline_performance = None
        
        logger.info(f"Drift monitor initialized: window={window_size}d, threshold={drift_threshold}")
    
    def record_performance(self, metrics):
        """Record daily performance metrics"""
        daily_metrics = {
            'timestamp': datetime.now().isoformat(),
            'hit@10': metrics.get('hit@10', 0),
            'ndcg@10': metrics.get('ndcg@10', 0),
            'inference_time': metrics.get('inference_ms', 0),
            'num_requests': metrics.get('num_requests', 0)
        }
        
        self.performance_history.append(daily_metrics)
        
        # Set baseline if not set
        if self.baseline_performance is None:
            self.baseline_performance = daily_metrics['hit@10']
        
        logger.info(f"Performance recorded: Hit@10={daily_metrics['hit@10']:.2f}%")
    
    def check_drift(self):
        """Check for model drift"""
        if len(self.performance_history) < 3:
            return False, "Insufficient data"
        
        # Calculate recent average
        recent_performance = np.mean([m['hit@10'] for m in self.performance_history])
        
        # Compare to baseline
        degradation = (self.baseline_performance - recent_performance) / self.baseline_performance
        
        is_drifting = degradation > self.drift_threshold
        
        if is_drifting:
            logger.warning(f"🚨 MODEL DRIFT DETECTED: {degradation*100:.1f}% degradation")
            return True, f"Performance degraded by {degradation*100:.1f}%"
        
        logger.info(f"✅ No drift detected: {degradation*100:.1f}% change")
        return False, f"Within threshold ({degradation*100:.1f}%)"
    
    def save_history(self, path="drift_history.json"):
        """Save performance history"""
        with open(path, 'w') as f:
            json.dump(list(self.performance_history), f, indent=2)


class AutomatedRetraining:
    """Automated model retraining pipeline"""
    
    def __init__(self, 
                 data_path,
                 model_config,
                 mlflow_tracker,
                 drift_monitor,
                 retrain_schedule="daily"):
        
        self.data_path = data_path
        self.model_config = model_config
        self.mlflow_tracker = mlflow_tracker
        self.drift_monitor = drift_monitor
        self.retrain_schedule = retrain_schedule
        
        logger.info(f"Automated retraining initialized: {retrain_schedule}")
    
    def load_new_data(self):
        """Load new interaction data from last 24h"""
        logger.info("Loading new interaction data...")
        
        # In production, this would query from database
        # For now, simulate new data
        cutoff_time = datetime.now() - timedelta(hours=24)
        
        df = pd.read_csv(self.data_path)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        new_data = df[df['timestamp'] > cutoff_time]
        
        logger.info(f"Loaded {len(new_data)} new interactions from last 24h")
        return new_data
    
    def retrain_model(self, new_data=None):
        """Retrain model with new data"""
        logger.info("="*70)
        logger.info("STARTING AUTOMATED RETRAINING")
        logger.info("="*70)
        
        # Start MLflow run
        self.mlflow_tracker.start_run(
            run_name=f"auto_retrain_{datetime.now().strftime('%Y%m%d_%H%M')}"
        )
        
        try:
            # Log retraining trigger
            self.mlflow_tracker.log_params({
                'retrain_trigger': 'scheduled_24h',
                'timestamp': datetime.now().isoformat(),
                'new_data_size': len(new_data) if new_data is not None else 0
            })
            
            # Train model (simplified - use your actual training code)
            logger.info("Training model...")
            
            # Import your actual training function
            try:
                # Prefer a normal package import if the workspace is installed as a package
                from gnn.gnn_training import EnhancedGraphSAGE  # type: ignore
            except Exception:
                # Fallback: load the script by file path relative to this file
                import importlib.util
                from pathlib import Path

                gnn_path = Path(__file__).resolve().parents[1] / 'gnn' / 'gnn_training.py'
                if not gnn_path.exists():
                    raise ImportError(f"Could not locate gnn_training.py at {gnn_path}")

                spec = importlib.util.spec_from_file_location('gnn_training', str(gnn_path))
                gnn_mod = importlib.util.module_from_spec(spec)
                assert spec.loader is not None
                spec.loader.exec_module(gnn_mod)  # type: ignore
                EnhancedGraphSAGE = getattr(gnn_mod, 'EnhancedGraphSAGE')

            model, metrics = EnhancedGraphSAGE(
                data_path=self.data_path,
                **self.model_config
            )
            
            # Log metrics
            self.mlflow_tracker.log_metrics(metrics)
            
            # Log model
            self.mlflow_tracker.log_model(model)
            
            # Register model
            model_version = self.mlflow_tracker.register_model("fashion-graphsage")
            
            # Check if better than production
            if metrics['hit@10'] > self.drift_monitor.baseline_performance:
                logger.info("✅ New model outperforms production")
                self.mlflow_tracker.transition_model_stage(
                    "fashion-graphsage",
                    model_version.version,
                    "Production"
                )
            else:
                logger.warning("⚠️ New model underperforms, keeping in Staging")
                self.mlflow_tracker.transition_model_stage(
                    "fashion-graphsage",
                    model_version.version,
                    "Staging"
                )
            
            # Record performance
            self.drift_monitor.record_performance(metrics)
            
            logger.info(f"✅ Retraining complete: Hit@10={metrics['hit@10']:.2f}%")
            
        except Exception as e:
            logger.error(f"❌ Retraining failed: {e}")
            raise
        
        finally:
            self.mlflow_tracker.end_run()
    
    def schedule_retraining(self):
        """Schedule automatic retraining"""
        if self.retrain_schedule == "daily":
            # Run every day at 2 AM
            schedule.every().day.at("02:00").do(self.retrain_model)
            logger.info("Scheduled daily retraining at 02:00")
        elif self.retrain_schedule == "hourly":
            schedule.every().hour.do(self.retrain_model)
            logger.info("Scheduled hourly retraining")
        
        # Run scheduler loop
        logger.info("Starting scheduler loop...")
        while True:
            schedule.run_pending()
            time.sleep(60)  # Check every minute


class TorchServeDeployment:
    """TorchServe deployment configuration and management"""
    
    def __init__(self, model_store="model_store", port=8080):
        self.model_store = Path(model_store)
        self.model_store.mkdir(exist_ok=True)
        self.port = port
        
        logger.info(f"TorchServe deployment initialized: port={port}")
    
    def create_handler(self, output_path="handler.py"):
        """Create TorchServe handler for inference"""
        
        handler_code = '''
import torch
import json
import numpy as np
from ts.torch_handler.base_handler import BaseHandler


class FashionRecommendationHandler(BaseHandler):
    """Custom handler for fashion recommendations"""
    
    def initialize(self, context):
        """Load model and metadata"""
        super().initialize(context)
        
        # Load embeddings
        self.embeddings = np.load('embeddings.npz', allow_pickle=True)
        self.item_ids = self.embeddings['item_ids']
        
        # Load user/item mappings
        with open('mappings.json', 'r') as f:
            self.mappings = json.load(f)
        
        self.initialized = True
    
    def preprocess(self, data):
        """Preprocess request"""
        # Extract user_id and k from request
        request = data[0]
        body = request.get("body") or request.get("data")
        
        if isinstance(body, (bytes, bytearray)):
            body = body.decode('utf-8')
        
        request_data = json.loads(body)
        
        user_id = request_data.get('user_id')
        k = request_data.get('k', 10)
        
        return {'user_id': user_id, 'k': k}
    
    def inference(self, data):
        """Run inference"""
        user_id = data['user_id']
        k = data['k']
        
        # Get user index
        user_idx = self.mappings['user_to_idx'].get(user_id)
        
        if user_idx is None:
            return {'error': 'User not found'}
        
        # Get embeddings (cached)
        user_emb = self.user_embeddings[user_idx]
        
        # Compute scores
        scores = np.dot(user_emb, self.item_embeddings.T)
        
        # Get top-K
        top_k_idx = np.argsort(scores)[-k:][::-1]
        top_k_items = [self.item_ids[idx] for idx in top_k_idx]
        top_k_scores = scores[top_k_idx].tolist()
        
        return {
            'user_id': user_id,
            'recommendations': [
                {'item_id': item, 'score': score}
                for item, score in zip(top_k_items, top_k_scores)
            ],
            'timestamp': datetime.now().isoformat()
        }
    
    def postprocess(self, inference_output):
        """Format response"""
        return [json.dumps(inference_output)]
'''
        
        with open(output_path, 'w') as f:
            f.write(handler_code)
        
        logger.info(f"Handler created: {output_path}")
    
    def export_model_for_torchserve(self, model, model_name="fashion_rec"):
        """Export model to TorchServe format"""
        
        # Save model
        model_path = self.model_store / f"{model_name}.pt"
        torch.save(model.state_dict(), model_path)
        
        # Create model archive
        import subprocess
        
        cmd = f"""
        torch-model-archiver \
          --model-name {model_name} \
          --version 1.0 \
          --model-file src/graphsage/model.py \
          --serialized-file {model_path} \
          --handler handler.py \
          --export-path {self.model_store}
        """
        
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            logger.info(f"✅ Model archive created: {model_name}.mar")
        else:
            logger.error(f"❌ Archive creation failed: {result.stderr}")
        
        return f"{model_name}.mar"
    
    def start_torchserve(self):
        """Start TorchServe"""
        import subprocess
        
        cmd = f"""
        torchserve \
          --start \
          --model-store {self.model_store} \
          --models fashion_rec=fashion_rec.mar \
          --ncs
        """
        
        subprocess.run(cmd, shell=True)
        logger.info(f"TorchServe started on port {self.port}")
    
    def health_check(self):
        """Check TorchServe health"""
        import requests
        
        try:
            response = requests.get(f"http://localhost:{self.port}/ping")
            if response.status_code == 200:
                logger.info("✅ TorchServe healthy")
                return True
        except:
            logger.error("❌ TorchServe not responding")
            return False


class PerformanceMonitor:
    """Monitor production model performance"""
    
    def __init__(self, metrics_file="production_metrics.json"):
        self.metrics_file = Path(metrics_file)
        self.daily_metrics = defaultdict(list)
        
        # Load existing metrics
        if self.metrics_file.exists():
            with open(self.metrics_file, 'r') as f:
                self.daily_metrics = json.load(f)
        
        logger.info("Performance monitor initialized")
    
    def record_request(self, user_id, recommendations, latency_ms, success=True):
        """Record individual request metrics"""
        today = datetime.now().strftime('%Y-%m-%d')
        
        self.daily_metrics[today].append({
            'timestamp': datetime.now().isoformat(),
            'user_id': user_id,
            'num_recommendations': len(recommendations),
            'latency_ms': latency_ms,
            'success': success
        })
    
    def get_daily_summary(self, date=None):
        """Get summary metrics for a day"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        day_data = self.daily_metrics.get(date, [])
        
        if not day_data:
            return None
        
        latencies = [r['latency_ms'] for r in day_data]
        
        return {
            'date': date,
            'total_requests': len(day_data),
            'successful_requests': sum(1 for r in day_data if r['success']),
            'avg_latency_ms': np.mean(latencies),
            'p95_latency_ms': np.percentile(latencies, 95),
            'p99_latency_ms': np.percentile(latencies, 99),
            'max_latency_ms': np.max(latencies)
        }
    
    def save_metrics(self):
        """Save metrics to file"""
        with open(self.metrics_file, 'w') as f:
            json.dump(dict(self.daily_metrics), f, indent=2)
    
    def generate_dashboard_data(self, days=30):
        """Generate data for monitoring dashboard"""
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        dashboard_data = []
        
        current_date = start_date
        while current_date <= end_date:
            date_str = current_date.strftime('%Y-%m-%d')
            summary = self.get_daily_summary(date_str)
            
            if summary:
                dashboard_data.append(summary)
            
            current_date += timedelta(days=1)
        
        return dashboard_data


class ProductionPipeline:
    """Complete production ML pipeline"""
    
    def __init__(self, config_path="pipeline_config.json"):
        # Load configuration
        with open(config_path, 'r') as f:
            self.config = json.load(f)
        
        # Initialize components
        self.mlflow_tracker = MLflowTracker(
            experiment_name=self.config['experiment_name']
        )
        
        self.drift_monitor = ModelDriftMonitor(
            window_size=self.config['drift_window_days'],
            drift_threshold=self.config['drift_threshold']
        )
        
        self.performance_monitor = PerformanceMonitor()
        
        self.torchserve = TorchServeDeployment(
            model_store=self.config['model_store'],
            port=self.config['serve_port']
        )
        
        self.automated_retraining = AutomatedRetraining(
            data_path=self.config['data_path'],
            model_config=self.config['model_params'],
            mlflow_tracker=self.mlflow_tracker,
            drift_monitor=self.drift_monitor
        )
        
        logger.info("Production pipeline initialized")
    
    def daily_health_check(self):
        """Daily health check and monitoring"""
        logger.info("Running daily health check...")
        
        # 1. Check TorchServe
        is_healthy = self.torchserve.health_check()
        
        # 2. Check for drift
        is_drifting, drift_msg = self.drift_monitor.check_drift()
        
        # 3. Get performance summary
        summary = self.performance_monitor.get_daily_summary()
        
        # 4. Log to MLflow
        if summary:
            self.mlflow_tracker.start_run(run_name="daily_monitoring")
            self.mlflow_tracker.log_metrics(summary)
            self.mlflow_tracker.end_run()
        
        # 5. Trigger retraining if drift detected
        if is_drifting:
            logger.warning(f"Drift detected: {drift_msg}")
            logger.info("Triggering emergency retraining...")
            self.automated_retraining.retrain_model()
        
        # 6. Save metrics
        self.drift_monitor.save_history()
        self.performance_monitor.save_metrics()
        
        logger.info("Daily health check complete")
    
    def start(self):
        """Start the production pipeline"""
        logger.info("="*70)
        logger.info("STARTING PRODUCTION ML PIPELINE")
        logger.info("="*70)
        
        # 1. Deploy current model
        logger.info("Deploying model to TorchServe...")
        self.torchserve.start_torchserve()
        
        # 2. Schedule daily retraining
        schedule.every().day.at("02:00").do(self.automated_retraining.retrain_model)
        
        # 3. Schedule daily health check
        schedule.every().day.at("08:00").do(self.daily_health_check)
        
        # 4. Start monitoring loop
        logger.info("Pipeline running. Press Ctrl+C to stop.")
        
        try:
            while True:
                schedule.run_pending()
                time.sleep(60)
        except KeyboardInterrupt:
            logger.info("Pipeline stopped by user")


# Configuration file template
def create_config_template():
    """Create pipeline configuration file"""
    
    config = {
        "experiment_name": "fashion-recommendation-production",
        "model_store": "./model_store",
        "serve_port": 8080,
        "data_path": os.environ.get('DATA_PATH'),
        "drift_window_days": 7,
        "drift_threshold": 0.15,
        "model_params": {
            "embedding_dim": 128,
            "num_layers": 3,
            "learning_rate": 0.001,
            "batch_size": 2048,
            "epochs": 30
        },
        "monitoring": {
            "enable_logging": True,
            "alert_email": "your.email@example.com",
            "slack_webhook": "https://hooks.slack.com/..."
        }
    }
    
    with open('pipeline_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    
    print("✅ Configuration template created: pipeline_config.json")


if __name__ == "__main__":
    # Create config if not exists
    if not Path('pipeline_config.json').exists():
        create_config_template()
        print("\n⚠️  Please update pipeline_config.json with your settings")
        print("Then run: python production_pipeline.py")
    else:
        # Start production pipeline
        pipeline = ProductionPipeline('pipeline_config.json')
        pipeline.start()