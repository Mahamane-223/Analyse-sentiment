# Analyse de sentiments : avis de films IMDb
 
Ce projet classe un avis de film (en anglais) comme **positif** ou **négatif**.
Le modèle est un TF-IDF suivi d'une régression logistique (scikit-learn). Il est exposé par une API **FastAPI** et utilisé depuis une interface **Streamlit**.
 
Projet réalisé pendant le mini-hackathon Data & IA de HETIC (novembre 2026).
 
## Architecture
 
```
Streamlit (port 8501)  ──POST /predict──▶  FastAPI (port 8000)  ──▶  model/sentiment.joblib
```
 
## Structure du dépôt
 
```
.
├── api/                 # API FastAPI (main.py + Dockerfile)
├── app/                 # Interface Streamlit (streamlit_app.py + Dockerfile)
├── src/                 # Préparation des données et entraînement
├── data/imdb/           # Jeu de données IMDb
├── model/               # Modèle entraîné (sentiment.joblib)
├── notebooks/           # Exploration des données
├── tests/               # Tests pytest
├── docker-compose.yml
└── .github/workflows/   # CI GitHub Actions
```
 
## Données
 
Jeu utilisé : [Sentiment Labelled Sentences (UCI)](https://archive.ics.uci.edu/dataset/331/sentiment+labelled+sentences), fichier `imdb_labelled.txt` : 1 000 phrases issues d'avis IMDb, étiquetées 1 (positif) ou 0 (négatif).
 
> **À compléter :** licence, nettoyage effectué (vides, doublons), répartition des classes.
 
## Prérequis
 
- Git
- Docker et Docker Compose
- Python 3.12 (uniquement pour lancer le projet sans Docker)
## Lancer le projet avec Docker (recommandé)
 
```bash
git clone https://github.com/Mahamane-223/Analyse-sentiment.git
cd Analyse-sentiment
docker compose up --build
```
 
- Interface : http://localhost:8501
- API et documentation Swagger : http://localhost:8000/docs
Pour arrêter : `Ctrl+C`, puis `docker compose down`.
 
## Lancer le projet sans Docker
 
```bash
python -m venv .venv
source .venv/bin/activate        # Windows : .venv\Scripts\activate
pip install -r requirements.txt -r app/requirements.txt
```
 
Ensuite, ouvrez deux terminaux à la racine du projet :
 
```bash
# Terminal 1 : l'API
uvicorn api.main:app --reload
 
# Terminal 2 : l'interface
streamlit run app/streamlit_app.py
```
 
## Utiliser l'API
 
| Route | Entrée | Sortie |
|---|---|---|
| `GET /health` | — | `{"status": "ok"}` |
| `POST /predict` | `{"text": "I loved this movie"}` | `{"label": "positif", "proba": 0.91}` |
 
Un texte vide ou composé d'espaces renvoie une erreur **HTTP 422**.
 
Exemple d'avis valide :
 
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "I loved the ending."}'
```
 
Exemple d'avis invalide, qui renvoie 422 :
 
```bash
curl -X POST http://localhost:8000/predict \
     -H "Content-Type: application/json" \
     -d '{"text": "   "}'
```
 
## Captures d'écran
 
| Avis positif | Avis négatif | Avis vide (422) |
|---|---|---|
| ![positif](/home/sarah/Downloads/positif.png) | ![négatif](/home/sarah/Downloads/negatif.png) | ![vide](/home/sarah/Downloads/neutre.png) |
 
## Tests et intégration continue
 
```bash
pytest -v
```
 
À chaque push, GitHub Actions (`.github/workflows/ci.yml`) lance `pytest` et vérifie que les images Docker se construisent.
 
## Résultats et limites
 
> **À compléter :** F1 macro du baseline et du modèle, matrice de confusion, exemples d'erreurs (ironie, négations, avis très courts).
 
## Équipe
 
| Rôle | Responsabilités |
|---|---|
| Data et cheffe de projet | données, évaluation, gestion du repo |
| Data 2 | entraînement du modèle, limites |
| Dev 1 | API FastAPI, tests pytest |
| Dev 2 | interface Streamlit, Docker, CI, README |