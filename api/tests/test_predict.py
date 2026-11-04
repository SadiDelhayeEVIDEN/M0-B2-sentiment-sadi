import csv
from pathlib import Path

import pytest

from app.inference import etoiles_vers_sentiment
from app.main import LONGUEUR_LOG_TEXTE, tronquer

# Le CSV est à la racine du dépôt : /app/data dans le conteneur (volume compose), ../data en local
CSV_CANDIDATS = [
    Path(__file__).resolve().parents[1] / "data" / "avis_annotes.csv",
    Path(__file__).resolve().parents[2] / "data" / "avis_annotes.csv",
]


AVIS_SANS_AMBIGUITE = {
    1, 4, 14, 20,   # positif
    2, 6, 15, 29,   # négatif
    5, 13,          # neutre
}


def charger_avis_sans_ambiguite() -> list[dict]:
    for chemin in CSV_CANDIDATS:
        if chemin.exists():
            with chemin.open(encoding="utf-8", newline="") as f:
                return [r for r in csv.DictReader(f) if int(r["id"]) in AVIS_SANS_AMBIGUITE]
    return []


AVIS = charger_avis_sans_ambiguite()


# --- Cas valide -----------------------------------------------------------------

def test_predict_positive_review(client):
    response = client.post("/predict", json={"text": "Chambre propre et personnel adorable."})
    assert response.status_code == 200
    body = response.json()
    assert body["sentiment"] == "positif"
    assert set(body["scores_etoiles"]) == {"1 star", "2 stars", "3 stars", "4 stars", "5 stars"}
    assert body["latence_ms"] > 0


def test_predict_negative_review(client):
    response = client.post("/predict", json={"text": "Chambre sale, personnel désagréable, à fuir."})
    assert response.json()["sentiment"] == "négatif"


# --- Entrées refusées (422) ------------------------------------------------------

@pytest.mark.parametrize("text", ["", "   ", "x" * 2001], ids=["vide", "espaces", "trop-long"])
def test_predict_rejects_invalid_text(client, text):
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 422


def test_predict_rejects_missing_text(client):
    response = client.post("/predict", json={})
    assert response.status_code == 422


# --- Avis annotés du CSV ---------------------------------------------------------

@pytest.mark.skipif(not AVIS, reason="data/avis_annotes.csv introuvable")
@pytest.mark.parametrize("avis", AVIS, ids=[f"avis{a['id']}-{a['sentiment_attendu']}" for a in AVIS])
def test_predict_matches_annotation_on_unambiguous_reviews(client, avis):
    response = client.post("/predict", json={"text": avis["texte"]})
    assert response.status_code == 200
    assert response.json()["sentiment"] == avis["sentiment_attendu"]


def test_all_selected_reviews_found_in_csv():
    if not AVIS:
        pytest.skip("data/avis_annotes.csv introuvable")
    assert len(AVIS) == len(AVIS_SANS_AMBIGUITE)


# --- Conversion étoiles -> sentiment ---------------------------------------------

def test_conversion_uses_weighted_mean_not_argmax():
    # "4 stars" est l'étoile la plus probable, mais la note moyenne vaut 2.45
    scores = {"1 star": 0.35, "2 stars": 0.25, "3 stars": 0.0, "4 stars": 0.4, "5 stars": 0.0}
    assert etoiles_vers_sentiment(scores) == "négatif"


# --- Troncature du texte dans les logs ------------------------------------------

def test_tronquer_limits_logged_text_to_80_chars():
    long_text = "Chambre 214, contactez-moi : j.lefevre@exemple.fr. " * 5
    tronque = tronquer(long_text)
    assert len(tronque) == LONGUEUR_LOG_TEXTE == 80
    assert tronque.endswith("…")
    assert "@" not in tronque[60:]  # la fin du texte (et l'e-mail répété) n'est pas loggée


def test_tronquer_keeps_short_text_and_flattens_newlines():
    assert tronquer("Très bien.\n\nJe reviendrai.") == "Très bien. Je reviendrai."
