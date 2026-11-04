"""Schémas Pydantic : format des entrées et des sorties de l'API."""

from typing import Literal

from pydantic import BaseModel, Field, field_validator


class ReviewIn(BaseModel):
    """Un avis client à analyser."""

    text: str = Field(min_length=1, max_length=2000, examples=["Chambre propre et personnel adorable."])

    @field_validator("text")
    @classmethod
    def non_vide(cls, value: str) -> str:
        # Refuse les textes composés uniquement d'espaces
        if not value.strip():
            raise ValueError("Le texte ne peut pas être vide")
        return value.strip()


class SentimentOut(BaseModel):
    """Réponse de /predict."""

    sentiment: Literal["négatif", "neutre", "positif"]
    # Scores bruts du modèle, pour la traçabilité : {"1 star": 0.02, ..., "5 stars": 0.71}
    scores_etoiles: dict[str, float]
    latence_ms: float
