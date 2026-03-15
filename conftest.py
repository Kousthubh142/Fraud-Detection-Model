import os
import sys
from pathlib import Path

# Set before any MLflow imports happen
os.environ.setdefault("MLFLOW_TRACKING_URI", "sqlite:///tmp/mlflow_test.db")

sys.path.insert(0, str(Path(__file__).parent / "src"))