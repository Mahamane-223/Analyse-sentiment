import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://localhost:8000")

st.title("🎬 Analyse de sentiments")
st.write("Saisis un avis de film, le modèle dira s'il est positif ou négatif.")

avis = st.text_area("Avis", placeholder="Ex. : Le film m'a ennuyé.")

if st.button("Analyser"):
    try:
        r = requests.post(f"{API_URL}/predict", json={"text": avis}, timeout=10)
    except requests.RequestException:
        st.error(f"Impossible de joindre l'API ({API_URL}). Est-elle lancée ?")
        st.stop()

    if r.status_code == 200:
        data = r.json()
        if data["label"] == "positif":
            st.success(f"😀 Sentiment : **{data['label']}**")
        else:
            st.error(f"🙁 Sentiment : **{data['label']}**")
        st.progress(data["proba"], text=f"Confiance : {data['proba']:.0%}")
    elif r.status_code == 422:
        st.warning("Avis invalide : le texte est vide ou ne contient que des espaces.")
    else:
        st.error(f"Erreur de l'API (HTTP {r.status_code})")
