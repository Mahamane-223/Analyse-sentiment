from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "penguins.csv"
ARTIFACTS = ROOT / "artifacts"
MODEL = ARTIFACTS / "model.joblib"

TARGET = "species"
FEATURES = [
    "bill_length_mm",
    "bill_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]