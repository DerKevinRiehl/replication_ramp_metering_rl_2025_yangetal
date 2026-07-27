import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

MODELS_DIR = os.path.join(REPO_ROOT, "1_code_produce", "single_ramp", "models")
DATA_DIR = os.path.join(REPO_ROOT, "2_data_produced")
PLOTS_DIR = os.path.join(REPO_ROOT, "3_data_visualization")

REPLACEMENT_HISTORY_PATH = os.path.join(
    MODELS_DIR, "training_history_replacement_seed42.pkl"
)
BASELINE_HISTORY_PATH = os.path.join(
    MODELS_DIR, "training_history_baseline_seed42.pkl"
)

os.makedirs(PLOTS_DIR, exist_ok=True)