from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
DATA_DIR = PROJECT_ROOT / "data"

# Real production value. Never lower this to make a model "qualify".
APPLICABILITY_THRESHOLD = 0.60

# Credit each provenance tag earns toward applicability.
TAG_WEIGHTS = {"real": 1.0, "real_approx": 0.5, "imputed": 0.0}

# Calibration-quality discount on a qualifying model's vote weight.
CALIBRATION_WEIGHT = {
    "german_credit": 1.0,
    "gmsc": 0.9,        # mild mid-range calibration wobble
    "homecredit": 0.6,  # overconfident after class-weighting
    "africa": 0.0,      # advisory only, never votes
}

# Only these models can influence the decision. Africa Synth (n=101,
# near-perfect test AUC = overfitting evidence) is explanation-only.
VOTING_MODELS = ["german_credit", "gmsc", "homecredit"]
ADVISORY_MODELS = ["africa"]

MODEL_PATHS = {
    "german_credit": MODELS_DIR / "CatBoost.pkl",
    "gmsc": MODELS_DIR / "gmsc" / "CatBoost.pkl",
    "homecredit": MODELS_DIR / "homecredit" / "CatBoost.pkl",
    "africa": MODELS_DIR / "africa" / "CatBoost.pkl",
}

# Optional: {dataset: {feature: mean_abs_shap}}. Generated in Step 2.
# If absent, the gate falls back to equal weights (current behaviour).
SHAP_WEIGHTS_PATH = MODELS_DIR / "shap_weights.pkl"