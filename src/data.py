from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
 
def load_imdb(path=DATA_DIR / "imdb" / "imdb_labelled.txt"):
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            text, label = line.rstrip("\n").rsplit("\t", 1)
            rows.append((text.strip(), int(label)))
    return pd.DataFrame(rows, columns=["review", "label"])

def load_allocine(split="train"):
    path = next((DATA_DIR / "allocine").glob(f"{split}-*.parquet"))
    return pd.read_parquet(path)