"""Configuration centrale du projet."""

from pathlib import Path
import torch

SEED = 42
HORIZONS = [1, 3, 6]
COVERAGE = 0.90
TEST_START_YEAR = 2021
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

ROOT = Path(__file__).resolve().parents[1]
DATA_RAW = ROOT / "data" / "raw"
DATA_INTERIM = ROOT / "data" / "interim"
DATA_PROCESSED = ROOT / "data" / "processed"
CONFIGS_DIR = ROOT / "configs"
MODELS_DIR = ROOT / "models"
MLRUNS_DIR = ROOT / "mlruns"
REPORTS_DIR = ROOT / "reports"
NOTEBOOKS_DIR = ROOT / "notebooks"
TESTS_DIR = ROOT / "tests"