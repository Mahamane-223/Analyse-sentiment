import streamlit as st

st.title("🎬 Analyse de sentiments")
st.write("Saisis un avis de film, le modèle dira s'il est positif ou négatif.")

avis = st.text_area("Avis", placeholder="Ex. : Le film m'a ennuyé.")

if st.button("Analyser"):
    st.write("Tu as écrit :", avis)
