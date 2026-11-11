"""Configuration settings for the 10-Class AI Object Recognition System."""

from pathlib import Path

# Paths
SRC_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SRC_DIR.parent
DATASET_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
SCRIPTS_DIR = PROJECT_ROOT / "scripts"

ATTRIBUTION_CSV = DATASET_DIR / "attribution.csv"
BEST_MODEL_PATH = MODELS_DIR / "mobilenetv2_best.keras"
METRICS_JSON_PATH = REPORTS_DIR / "metrics.json"
CONFUSION_MATRIX_PATH = REPORTS_DIR / "confusion_matrix.png"
TRAINING_HISTORY_PATH = REPORTS_DIR / "training_history.png"
CLASSIFICATION_REPORT_PATH = REPORTS_DIR / "classification_report.txt"

# Ensure runtime directories exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
DATASET_DIR.mkdir(parents=True, exist_ok=True)

# 10 Object Classes (Alphabetical order for deterministic label indexing)
CLASS_NAMES = sorted([
    "water_bottle",
    "coffee_mug",
    "smartphone",
    "book",
    "computer_mouse",
    "keyboard",
    "tv_remote",
    "shoe",
    "scissors",
    "wallet",
])

# Human-readable labels
DISPLAY_NAMES = {
    "book": "Book",
    "coffee_mug": "Coffee Mug",
    "computer_mouse": "Computer Mouse",
    "keyboard": "Keyboard",
    "scissors": "Scissors",
    "shoe": "Shoe",
    "smartphone": "Smartphone",
    "tv_remote": "TV Remote",
    "wallet": "Wallet",
    "water_bottle": "Water Bottle",
}

# Image Parameters
IMG_HEIGHT = 224
IMG_WIDTH = 224
IMG_SIZE = (IMG_HEIGHT, IMG_WIDTH)
CHANNELS = 3
INPUT_SHAPE = (IMG_HEIGHT, IMG_WIDTH, CHANNELS)

# Dataset Splits & Target
TARGET_PER_CLASS = 50
TOTAL_TARGET = len(CLASS_NAMES) * TARGET_PER_CLASS  # 500
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
RANDOM_SEED = 42

# Training Hyperparameters
BATCH_SIZE = 16
INITIAL_EPOCHS = 15
INITIAL_LR = 1e-3
FINE_TUNE_EPOCHS = 10
FINE_TUNE_LR = 1e-5
PATIENCE_EARLY_STOP = 5
PATIENCE_REDUCE_LR = 3

# Inference Parameters
CONFIDENCE_THRESHOLD = 0.45  # Below this, flag as low-confidence
TOP_K = 3
