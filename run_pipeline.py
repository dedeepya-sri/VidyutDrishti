#!/usr/bin/env python3
"""
VidyutDrishti Pipeline Runner
Complete pipeline execution for Karnataka renewable energy forecasting
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class PipelineRunner:
    """Manages execution of the complete VidyutDrishti pipeline"""

    def __init__(self, base_dir: str = None):
        self.base_dir = Path(base_dir) if base_dir else Path(__file__).parent
        self.python_cmd = sys.executable

    def run_command(self, cmd: list, cwd: Path = None, description: str = "") -> bool:
        """Run a command with logging"""
        if description:
            logger.info(f"Starting: {description}")

        try:
            result = subprocess.run(
                cmd,
                cwd=cwd or self.base_dir,
                capture_output=True,
                text=True,
                check=True
            )
            logger.info(f"Completed: {description}")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed: {description}")
            logger.error(f"Error: {e.stderr}")
            return False

    def step_data_ingestion(self) -> bool:
        """Step 1: Data ingestion from real sources"""
        logger.info("=== STEP 1: Data Ingestion ===")

        # Create data directories
        data_dirs = [
            self.base_dir / "data" / "raw",
            self.base_dir / "data" / "processed"
        ]
        for dir_path in data_dirs:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Download NASA POWER data
        success = self.run_command(
            [self.python_cmd, "data_pipeline/ingestion/download_nasa_power.py"],
            description="Downloading NASA POWER data"
        )

        if not success:
            logger.warning("NASA POWER download failed, using synthetic data")

        # Generate SCADA-like data
        success &= self.run_command(
            [self.python_cmd, "data_pipeline/ingestion/generate_scada_data.py"],
            description="Generating SCADA-like generation data"
        )

        return success

    def step_data_preprocessing(self) -> bool:
        """Step 2: Data preprocessing and feature engineering"""
        logger.info("=== STEP 2: Data Preprocessing ===")

        success = self.run_command(
            [self.python_cmd, "data_pipeline/preprocessing/feature_engineering.py"],
            description="Feature engineering and preprocessing"
        )

        # Data validation
        success &= self.run_command(
            [self.python_cmd, "data_pipeline/validation/data_quality_checks.py"],
            description="Data quality validation"
        )

        return success

    def step_model_training(self) -> bool:
        """Step 3: Model training"""
        logger.info("=== STEP 3: Model Training ===")

        # Create models directory
        models_dir = self.base_dir / "models"
        models_dir.mkdir(exist_ok=True)

        # Train LightGBM quantile model
        success = self.run_command(
            [self.python_cmd, "-c", "from models.lightgbm.quantile_regression import train_karnataka_model; train_karnataka_model()"],
            description="Training LightGBM quantile model"
        )

        # Train LSTM model (if TensorFlow available)
        try:
            success &= self.run_command(
                [self.python_cmd, "-c", "from models.deep.lstm_model import train_karnataka_lstm; train_karnataka_lstm()"],
                description="Training LSTM model"
            )
        except Exception as e:
            logger.warning(f"LSTM training failed: {e}")

        # Create ensemble
        try:
            success &= self.run_command(
                [self.python_cmd, "-c", "from models.ensemble.hybrid_ensemble import create_karnataka_ensemble; create_karnataka_ensemble()"],
                description="Creating hybrid ensemble"
            )
        except Exception as e:
            logger.warning(f"Ensemble creation failed: {e}")

        return success

    def step_model_evaluation(self) -> bool:
        """Step 4: Model evaluation"""
        logger.info("=== STEP 4: Model Evaluation ===")

        success = self.run_command(
            [self.python_cmd, "-c", "from models.evaluation.model_evaluation import evaluate_karnataka_models; evaluate_karnataka_models()"],
            description="Model evaluation and comparison"
        )

        return success

    def step_api_startup(self) -> bool:
        """Step 5: Start API server"""
        logger.info("=== STEP 5: API Server ===")

        logger.info("Starting FastAPI server...")
        logger.info("API will be available at: http://localhost:8000")
        logger.info("API documentation at: http://localhost:8000/docs")
        logger.info("Press Ctrl+C to stop the server")

        try:
            subprocess.run([
                self.python_cmd, "-m", "uvicorn",
                "api.main:app",
                "--host", "0.0.0.0",
                "--port", "8000",
                "--reload"
            ], cwd=self.base_dir)
            return True
        except KeyboardInterrupt:
            logger.info("API server stopped by user")
            return True
        except Exception as e:
            logger.error(f"API startup failed: {e}")
            return False

    def run_full_pipeline(self, skip_api: bool = False) -> bool:
        """Run the complete pipeline"""
        logger.info("🚀 Starting VidyutDrishti Pipeline")
        logger.info("=" * 50)

        start_time = datetime.now()

        # Execute pipeline steps
        steps = [
            ("Data Ingestion", self.step_data_ingestion),
            ("Data Preprocessing", self.step_data_preprocessing),
            ("Model Training", self.step_model_training),
            ("Model Evaluation", self.step_model_evaluation),
        ]

        if not skip_api:
            steps.append(("API Server", self.step_api_startup))

        success = True
        for step_name, step_func in steps:
            step_start = datetime.now()

            if step_func():
                step_duration = datetime.now() - step_start
                logger.info(f"✅ {step_name} completed in {step_duration}")
            else:
                logger.error(f"❌ {step_name} failed")
                success = False
                break

        total_duration = datetime.now() - start_time

        if success:
            logger.info(f"🎉 Pipeline completed successfully in {total_duration}")
            logger.info("\nNext steps:")
            logger.info("1. Open dashboard: python dashboard/app.py")
            logger.info("2. API docs: http://localhost:8000/docs")
            logger.info("3. View evaluation: docs/model_evaluation_report.txt")
        else:
            logger.error(f"💥 Pipeline failed after {total_duration}")

        return success

    def run_single_step(self, step: str) -> bool:
        """Run a single pipeline step"""
        step_map = {
            "ingestion": self.step_data_ingestion,
            "preprocessing": self.step_data_preprocessing,
            "training": self.step_model_training,
            "evaluation": self.step_model_evaluation,
            "api": self.step_api_startup
        }

        if step not in step_map:
            logger.error(f"Unknown step: {step}")
            return False

        return step_map[step]()

def main():
    parser = argparse.ArgumentParser(description="VidyutDrishti Pipeline Runner")
    parser.add_argument(
        "command",
        choices=["full", "ingestion", "preprocessing", "training", "evaluation", "api"],
        help="Pipeline command to run"
    )
    parser.add_argument(
        "--skip-api",
        action="store_true",
        help="Skip API server startup in full pipeline"
    )
    parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Base directory for the project"
    )

    args = parser.parse_args()

    runner = PipelineRunner(args.base_dir)

    if args.command == "full":
        success = runner.run_full_pipeline(skip_api=args.skip_api)
    else:
        success = runner.run_single_step(args.command)

    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()