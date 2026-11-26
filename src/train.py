#import et reglages

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline

RANDOM_STATE = 42
ROOT = Path(__file__).resolve().parent.parent      # racine du projet
TRAIN_PATH = ROOT / "data" / "imdb" / "train.csv"
TEST_PATH = ROOT / "data" / "imdb" / "test.csv"
MODEL_PATH = ROOT / "model" / "sentiment.joblib"
METRICS_PATH = ROOT / "model" / "metrics.json"

#chargeùent des données netoyées

train = pd.read_csv(TRAIN_PATH)
test = pd.read_csv(TEST_PATH)
X_train, y_train = train["review"], train["label"]
X_test, y_test = test["review"], test["label"]
print(f"Train : {len(X_train)} avis | Test : {len(X_test)} avis")

#baseline et comparaison des deux modeles

baseline = DummyClassifier(strategy="most_frequent")
baseline.fit(X_train, y_train)
f1_baseline = f1_score(y_test, baseline.predict(X_test), average="macro")
print(f"\nBaseline (classe majoritaire) - F1 macro test : {f1_baseline:.3f}")


def build_pipeline(ngram_range):
    return Pipeline([
        ("tfidf", TfidfVectorizer(ngram_range=ngram_range)),
        ("clf", LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)),
    ])


scores = {}
for ngram in [(1, 1), (1, 2)]:
    cv = cross_val_score(build_pipeline(ngram), X_train, y_train, cv=5, scoring="f1_macro")
    scores[ngram] = cv.mean()
    print(f"TF-IDF ngram_range={ngram} - F1 macro (validation croisée) : {cv.mean():.3f} (+/- {cv.std():.3f})")

best_ngram = max(scores, key=scores.get)
print(f"\nModèle retenu : ngram_range={best_ngram}")

# entrainement du modèle final score sauvegarde

model = build_pipeline(best_ngram)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
f1_test = f1_score(y_test, y_pred, average="macro")
print("\nRapport sur le jeu de test :")
print(classification_report(y_test, y_pred, target_names=["négatif", "positif"], digits=3))

MODEL_PATH.parent.mkdir(exist_ok=True)
joblib.dump(model, MODEL_PATH)

metrics = {
    "dataset": "IMDb (UCI Sentiment Labelled Sentences)",
    "baseline_f1_macro": round(f1_baseline, 3),
    "cv_f1_macro": {str(k): round(v, 3) for k, v in scores.items()},
    "best_ngram_range": str(best_ngram),
    "test_f1_macro": round(f1_test, 3),
}
METRICS_PATH.write_text(json.dumps(metrics, indent=2, ensure_ascii=False))
print(f"Modèle sauvegardé dans {MODEL_PATH}")

for phrase in ["I loved the ending.", "The movie was boring."]:
    proba = model.predict_proba([phrase])[0]
    label = "positif" if proba[1] >= 0.5 else "négatif"
    print(f"  « {phrase} » -> {label} ({max(proba):.0%})")




