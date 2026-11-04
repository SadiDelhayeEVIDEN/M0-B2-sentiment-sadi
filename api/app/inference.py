"""Chargement du modèle Hugging Face et inférence."""

import time

from transformers import pipeline

# Note moyenne pondérée -> classe métier
SEUIL_NEGATIF = 2.5
SEUIL_POSITIF = 3.5


def load_pipeline(model_name: str):
    """Charge le pipeline de classification (téléchargé au 1er lancement, puis en cache).

    truncation=True : un avis plus long que la limite du modèle (512 tokens) est coupé
    au lieu de faire planter la requête.
    """
    return pipeline("text-classification", model=model_name, top_k=None, truncation=True)


def etoiles_vers_sentiment(scores: dict[str, float]) -> str:
    """Convertit les scores des 5 étoiles en négatif / neutre / positif.

    Choix : note moyenne pondérée par les probabilités (espérance de la note).
    - Prendre l'étoile la plus probable ignore la répartition : 0.40 sur "4 stars"
      et 0.60 réparti sur 1-2 étoiles serait classé positif, alors que la note
      moyenne vaut 2.45.
    - La moyenne pondérée exploite toute la distribution et donne une seule valeur
      lisible (de 1.0 à 5.0), que l'on découpe en 3 zones de largeur égale autour
      de la note neutre 3 : < 2.5 négatif, > 3.5 positif, entre les deux neutre.
    """
    # "3 stars" -> 3
    note = sum(int(label.split()[0]) * score for label, score in scores.items())

    if note < SEUIL_NEGATIF:
        return "négatif"
    if note > SEUIL_POSITIF:
        return "positif"
    return "neutre"


def predict_sentiment(classifier, text: str) -> dict:
    """Prédit le sentiment métier (négatif / neutre / positif) d'un avis."""
    debut = time.perf_counter()

    # top_k=None -> [[{"label": "1 star", "score": 0.02}, ...]]
    resultats = classifier(text)[0]
    scores = {r["label"]: round(r["score"], 4) for r in resultats}

    return {
        "sentiment": etoiles_vers_sentiment(scores),
        "scores_etoiles": scores,
        "latence_ms": round((time.perf_counter() - debut) * 1000, 1),
    }
