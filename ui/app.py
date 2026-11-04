"""Interface du service qualité — Escale Atlantique (mission FastIA)."""

import os

import streamlit as st
import httpx


# Dans docker compose, l'API est joignable par le NOM de son service (cf. .env)
API_URL = os.getenv("API_URL", "http://localhost:8008")

st.set_page_config(page_title="Escale Atlantique — Avis clients", page_icon="🏨")
st.title("🏨 Sentiment des avis clients.")
st.caption("Escale Atlantique · outil du service qualité")

text = st.text_area(
    "Avis client",
    height=160,
    placeholder="Collez ici un avis Booking, Google, TripAdvisor ou un questionnaire de fin de séjour…",
)

sentiments = {
    "négatif": ("😠", "red"),
    "neutre":  ("😐", "orange"),
    "positif": ("😊", "green"),
}

if st.button("Analyser", type="primary"):
    with st.spinner("Analyse par camembert…"):
        try:
            response = httpx.post(f"{API_URL}/predict", json={"text": text}, timeout=10)
            if response.status_code == 200:
                result = response.json()
                emoji, couleur = sentiments.get(result["sentiment"], ("❓", "gray"))
                st.badge(f"Sentiment : {result['sentiment']}", icon=emoji, color=couleur)
                st.write(f"Score : {result['score']:.2f}")
            elif response.status_code == 422:
                st.error("Le texte fourni est vide ou invalide.")
            elif response.status_code == 500:
                st.error("Erreur interne du serveur. Veuillez réessayer plus tard.")                
            else:
                st.error("Erreur lors de l'analyse du sentiment.")
        except httpx.HTTPError as e:
            st.error(f"Erreur de communication avec l'API : {e}")

