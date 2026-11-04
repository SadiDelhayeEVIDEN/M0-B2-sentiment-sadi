import pytest

from app.inference import etoiles_vers_sentiment


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


@pytest.mark.parametrize("text", ["", "   ", "x" * 2001])
def test_predict_rejects_invalid_text(client, text):
    response = client.post("/predict", json={"text": text})
    assert response.status_code == 422


def test_conversion_uses_weighted_mean_not_argmax():
    # "4 stars" est l'étoile la plus probable, mais la note moyenne vaut 2.45
    scores = {"1 star": 0.35, "2 stars": 0.25, "3 stars": 0.0, "4 stars": 0.4, "5 stars": 0.0}
    assert etoiles_vers_sentiment(scores) == "négatif"
