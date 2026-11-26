"""Chemins du projet et correspondance des labels, définis en un seul endroit."""

from pathlib import Path

# Racine du dépôt : src/config.py -> src/ -> racine
ROOT = Path(__file__).resolve().parent.parent

# Données
DATA_DIR = ROOT / "data"
IMDB_DIR = DATA_DIR / "imdb"
IMDB_RAW_PATH = IMDB_DIR / "imdb_labelled.txt"
IMDB_CLEAN_PATH = IMDB_DIR / "imdb_clean.csv"
TRAIN_PATH = IMDB_DIR / "train.csv"
TEST_PATH = IMDB_DIR / "test.csv"
ALLOCINE_DIR = DATA_DIR / "allocine"

# Modèle entraîné
MODEL_DIR = ROOT / "model"
MODEL_PATH = MODEL_DIR / "sentiment.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"

# Classes du modèle (0/1) -> libellés renvoyés par l'API
LABELS = {0: "négatif", 1: "positif"}
